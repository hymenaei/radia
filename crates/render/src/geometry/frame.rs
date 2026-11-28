use std::{
    cmp::Ordering,
    collections::BTreeSet,
    mem,
    ops::Bound::{Excluded, Unbounded},
    time::{Duration, Instant},
};

use assets::Priority;
use math::{DVec3, Frustum, Mat4};
use uuid::Uuid;
use wgpu::{CommandEncoder, Device};
use world::{Kind, WorldPose};

use super::{FramePreparation, FramePreparationKey, GeometryRenderer, PrepareTimes, ShadingMode, Stats, mesh::MeshUpload};
use crate::{
    FrameInput,
    environment::Evaluated,
    geometry::{
        cache::{CacheUpdate, MeshDemandOrder, item_is_live, range_moved},
        draw_list::LAYOUT_TIME_PER_FRAME,
        instance::{instance_base, parameter_part, scene_origin},
        items::{CachedFace, CachedItem, ItemIndex, ItemSlab},
        visibility::WorldSphere,
    },
    material::FaceParams,
    pipeline,
    pipeline::{TexturePair, blend_keys},
    scene::{AgentPose, Camera, Object, Placement, SceneView, motion, texture_animation},
    texture::Textures,
};

pub(super) const BUDGET_REPORT_INTERVAL: Duration = Duration::from_secs(5);
pub(super) const SLOW_PREPARE_REPORT_INTERVAL: Duration = Duration::from_secs(10);

pub(super) const MESH_DEMANDS_PER_FRAME: usize = 24;

#[derive(Default)]
pub(super) struct MeshAdmission {
    pub(super) revision: u64,
    pub(super) active: bool,
    after: Option<MeshDemandOrder>,
    pass_revision: u64,
    blocked: bool,
    retry: bool,
    pub(super) restart_pending: bool,
    pub(super) retry_after_layout: bool,
    pub(super) capacity_blocked: bool,
    pub(super) visible_ready: bool,
    pub(super) visits_this_frame: usize,
}

impl MeshAdmission {
    fn begin(&mut self, revision: u64) {
        *self = Self { revision, active: true, ..Self::default() };
        self.restart(revision);
    }

    fn restart(&mut self, revision: u64) {
        self.after = None;
        self.pass_revision = revision;
        self.blocked = false;
        self.retry = false;
        self.capacity_blocked = false;
        self.visible_ready = false;
    }

    fn pass_incomplete(&self, revision: u64, retired: bool) -> bool {
        self.after.is_some() && (self.pass_revision != revision || self.blocked && (self.retry || retired))
    }
}

fn next_mesh_demand(order: &BTreeSet<MeshDemandOrder>, after: Option<MeshDemandOrder>) -> Option<MeshDemandOrder> {
    match after {
        Some(after) => order.range((Excluded(after), Unbounded)).next().copied(),
        None => order.first().copied(),
    }
}

fn membership_changes(visible: &[ItemIndex], next_visible: &[ItemIndex], items: &ItemSlab) -> (bool, bool) {
    let id = |index: &ItemIndex| items.get(*index).map(|item| item.id);
    let mut old = visible.iter().map(id).peekable();
    let mut new = next_visible.iter().map(id).peekable();
    let (mut entered, mut left) = (false, false);
    loop {
        match (old.peek(), new.peek()) {
            (None, None) => return (entered, left),
            (Some(_), None) => return (entered, true),
            (None, Some(_)) => return (true, left),
            (Some(a), Some(b)) => match a.cmp(b) {
                Ordering::Less => {
                    left = true;
                    old.next();
                }
                Ordering::Greater => {
                    entered = true;
                    new.next();
                }
                Ordering::Equal => {
                    old.next();
                    new.next();
                }
            },
        }
    }
}

#[derive(Clone, Copy, Default)]
struct EnvironmentPairs {
    sun: TexturePair,
    moon: TexturePair,
    cloud: TexturePair,
    bloom: TexturePair,
    rainbow: TexturePair,
    halo: TexturePair,
    water_normal: TexturePair,
}

impl GeometryRenderer {
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        view: &SceneView,
        agent_pose: Option<AgentPose>,
        camera: Camera,
        size: [u32; 2],
        water_height: Option<f64>,
    ) {
        self.prepare_with_environment(
            device,
            encoder,
            view,
            agent_pose,
            camera,
            size,
            FrameInput { water_height, ..FrameInput::default() },
        );
    }

    pub(crate) fn prepare_with_environment(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        view: &SceneView,
        agent_pose: Option<AgentPose>,
        camera: Camera,
        size: [u32; 2],
        frame_input: FrameInput<'_>,
    ) {
        self.prepare_at(device, encoder, view, agent_pose, camera, size, frame_input, Instant::now());
    }

    pub(super) fn prepare_at(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        view: &SceneView,
        agent_pose: Option<AgentPose>,
        camera: Camera,
        size: [u32; 2],
        frame_input: FrameInput<'_>,
        started: Instant,
    ) {
        let mut times = PrepareTimes::default();
        let animation_now = started;
        self.stats.layouts = 0;
        self.stats.items_updated = 0;
        self.stats.faces_patched = 0;
        self.stats.blended_sorts = 0;
        self.stats.animated_faces_visited = 0;
        self.stats.animated_faces_changed = 0;
        self.epoch = self.epoch.checked_add(1).expect("render frame space exhausted");
        if let Some(environment) = frame_input.environment {
            request_environment_textures(view, environment, &mut self.textures, self.epoch);
        }
        let key = FramePreparationKey { version: view.geometry_version(), camera, agent_pose, size };
        let current_version = view.version();
        let same = self.prepared.as_ref().is_some_and(|prepared| prepared.key == key)
            && self.cache.version.as_ref().is_some_and(|version| version == &current_version);
        if same {
            for id in &self.visible_textures {
                self.textures.touch(*id, self.epoch);
            }
            for texture in &self.visible_texture_requests {
                self.textures.request(texture, self.epoch, Priority::Visible);
            }
        }
        let motion_was_active = self.motion_redraw;
        self.motion_redraw = false;
        if same {
            let texture_progress =
                self.prepared.as_ref().is_some_and(|prepared| prepared.textures != self.textures.state()) || self.textures.uploading();
            if texture_progress {
                let upload_started = Instant::now();
                self.textures.upload(device, encoder, self.epoch);
                times.textures = upload_started.elapsed();
            }
            if self.prepared.as_ref().is_some_and(|prepared| {
                !motion_was_active && prepared.reusable(&key, self.textures.geometry_state(), self.memory.mesh_state(), started)
            }) {
                let projection = self.pipelines.update_camera_uniform(&self.queue, camera, size, self.active_origin);
                let pairs = environment_pairs(view, &self.textures, frame_input.environment);
                self.update_environment_uniforms(device, camera, projection, size, frame_input, pairs);
                if let Some(prepared) = self.prepared.as_mut() {
                    prepared.textures = self.textures.state();
                    prepared.geometry_textures = self.textures.geometry_state();
                    prepared.meshes = self.memory.mesh_state();
                }
                self.cache.version = Some(view.version());
                self.stats.uploaded_bytes = 0;
                self.memory_stats();
                self.record_prepare(started, times, 0);
                return;
            }
        }

        self.stats = Stats::default();
        self.mesh_admission.visits_this_frame = 0;
        let previous_camera = self.cache.camera;
        let camera_changed = previous_camera != Some(camera) || self.cache.size != size;
        let origin = scene_origin(camera.position());
        if !self.draw_list.valid() {
            self.active_origin = origin;
        }
        if self.cache.items.slot_count() == 0 && !self.draw_list.valid() {
            self.cache.origin = origin;
        }

        let visibility_started = Instant::now();
        let noted_camera = self.note_rebuild_camera(camera);
        let cache_started = Instant::now();
        let mut update = self.update_cache_for_frame(view, self.recovery_publication_pending);
        if update.rebuild_completed {
            self.recovery_publication_pending = true;
        }
        if update.rebuild_started {
            self.moving_entities.clear();
            self.motion_entities.clear();
            self.mesh_admission.restart_pending = true;
            self.pending_meshes.clear();
            self.mesh_budget_blocked.clear();
            self.layout_pose_items.clear();
        }
        if update.immediate_visibility_change {
            self.moving_entities.retain(|entity| view.object_is_live(*entity) && self.cache.objects.contains_key(entity));
            self.motion_entities.clear();
        }
        let candidate_camera = self.note_rebuild_camera(camera).or(noted_camera).unwrap_or(camera);
        times.cache = cache_started.elapsed();
        let rebuilding = update.rebuild_pending;
        let stale_pass = update.rebuild_completed && range_moved(candidate_camera, camera);
        if self.draw_list.layout_pending() && (update.immediate_visibility_change || update.layout_invalidated) {
            self.layout_required = true;
        }
        let frame_origin = if rebuilding { self.cache.origin } else { origin };
        let origin_rebased = !rebuilding && self.cache.origin != origin;
        self.apply_agent_pose(agent_pose, frame_origin, &mut update);
        let motion_started = Instant::now();
        if rebuilding {
            let entities = update.background.clone();
            self.update_object_motion_page(view, started, frame_origin, &entities, agent_pose.map(AgentPose::agent_id), &mut update);
        } else if update.rebuild_completed {
            self.refresh_motion_membership(view, &update.background);
            self.motion_entities.clear();
        } else {
            self.update_object_motion(view, started, frame_origin, agent_pose.map(AgentPose::agent_id), &mut update);
        }
        times.motion += motion_started.elapsed();
        self.stats.items_updated = update.items_updated;
        update.pose_items.sort_unstable();
        update.pose_items.dedup();
        let projection = self.pipelines.update_camera_uniform(&self.queue, camera, size, self.active_origin);
        let frustum = Frustum::from_clip_matrix(projection);
        let mut range_items = update.visibility_items.clone();
        range_items.extend(update.pose_items.iter().copied());
        range_items.sort_unstable();
        range_items.dedup();
        let candidates_started = Instant::now();
        let mut mesh_demand_missing = update.mesh_demand_missing;
        let full_range = update.full_visibility || stale_pass || self.cache.range_camera.is_none_or(|anchor| range_moved(anchor, camera));
        let mut range_changed = if rebuilding || update.rebuild_completed && !stale_pass {
            if self.cache.range_camera.is_none_or(|anchor| range_moved(anchor, candidate_camera)) {
                self.cache.range_camera = Some(candidate_camera);
            }
            let candidate_frustum = Frustum::from_clip_matrix(pipeline::projection(candidate_camera, size));
            let (changed, missing) = self.update_range(view, &range_items, Some((candidate_camera, &candidate_frustum)));
            mesh_demand_missing |= missing;
            changed
        } else if full_range {
            mesh_demand_missing = true;
            self.rebuild_range(view, camera)
        } else if !range_items.is_empty() {
            let (changed, missing) = self.update_range(view, &range_items, None);
            mesh_demand_missing |= missing;
            changed
        } else {
            false
        };
        if rebuilding && let Some(rebuild) = self.cache.rebuild.as_mut() {
            rebuild.visibility_changed |= range_changed;
            rebuild.mesh_demand_missing |= mesh_demand_missing;
        }
        range_changed |= update.rebuild_visibility_changed;
        times.candidates += candidates_started.elapsed();
        times.visibility += visibility_started.elapsed().saturating_sub(times.cache);

        let previous_mesh_state = self.cache.mesh_state;
        if previous_mesh_state != self.memory.mesh_state() {
            self.draw_list.invalidate_batches();
        }
        let meshes_started = Instant::now();
        let demand_reclassified = camera_changed && self.cache.nonresident_demands;
        let mesh_demands_started = Instant::now();
        let candidate_mesh_missing = if !rebuilding
            && (!update.rebuild_completed || stale_pass)
            && (mesh_demand_missing || update.mesh_dirty || demand_reclassified)
        {
            self.rebuild_visible_meshes(view, camera, &frustum)
        } else {
            false
        };
        times.mesh_demands = mesh_demands_started.elapsed();
        let mesh_revision_changed = self.mesh_admission.revision != self.cache.mesh_demand_revision;
        let mesh_follow_up = self.prepared.as_ref().is_some_and(|prepared| prepared.more_meshes)
            && (self.mesh_admission.active || self.mesh_admission.restart_pending);
        let mesh_admission_requested = candidate_mesh_missing
            || mesh_demand_missing
            || update.mesh_dirty
            || demand_reclassified
            || mesh_revision_changed
            || self.mesh_admission.active
            || self.mesh_admission.restart_pending
            || self.mesh_admission.retry_after_layout && !self.draw_list.layout_pending()
            || self.cache.mesh_state != self.memory.mesh_state()
            || mesh_follow_up;
        let mut uploaded = 0;
        let mesh_uploads_started = Instant::now();
        if mesh_admission_requested {
            self.advance_mesh_admission(device, &mut uploaded, camera, &frustum);
            self.cache.mesh_state = self.memory.mesh_state();
        }
        let residency_changed = uploaded > 0 || self.stats.gpu_mesh_retirements > 0;
        times.mesh_uploads = mesh_uploads_started.elapsed();
        self.stats.items_updated = update.items_updated;
        if self.cache.mesh_state != previous_mesh_state {
            self.draw_list.invalidate_batches();
        }
        let appearances_started = Instant::now();
        let replacement_indices = if rebuilding { Vec::new() } else { self.cache.replacements.iter().copied().collect::<Vec<_>>() };
        let mut promotions = Vec::new();
        for index in replacement_indices {
            self.stats.appearance_candidates_checked += 1;
            let Some(replacement) = self.cache.items.get(index).and_then(|item| item.replacement.as_ref()) else { continue };
            if self.meshes.contains_key(&replacement.item.mesh.id()) {
                promotions.push(index);
            }
        }
        if self.draw_list.layout_pending() {
            self.replacement_retry_pending |= !promotions.is_empty();
            promotions.clear();
        } else {
            self.replacement_retry_pending = false;
        }
        let replacement_promoted = self.promote_replacements(&promotions, &mut update);
        if replacement_promoted {
            let (promoted_range, _) = self.update_range(view, &promotions, None);
            range_changed |= promoted_range;
            self.rebuild_visible_meshes(view, camera, &frustum);
            self.pending_meshes.retain(|mesh| self.cache.mesh_order_by_id.contains_key(mesh));
        }
        times.appearances = appearances_started.elapsed();
        let rebuild_visible = range_changed || residency_changed || replacement_promoted || !update.visibility_items.is_empty();
        if rebuild_visible {
            self.cache.next_visible.clear();
            self.mesh_omitted_faces = 0;
            for id in &self.cache.range {
                self.stats.visible_candidates_checked += 1;
                let Some(item) = self.cache.items.get(*id) else { continue };
                if !item_is_live(view, item) {
                    continue;
                }
                let mesh_ready = self.meshes.contains_key(&item.mesh.id());
                if !mesh_ready && self.mesh_budget_blocked.contains(&item.mesh.id()) {
                    self.mesh_omitted_faces = self.mesh_omitted_faces.saturating_add(item.faces.len());
                }
                if mesh_ready && !item.faces.is_empty() {
                    self.cache.next_visible.push(*id);
                }
            }
        }
        times.meshes = meshes_started.elapsed();
        let visible_changed = rebuild_visible && self.cache.visible != self.cache.next_visible;
        if visible_changed {
            let (entered, left) = membership_changes(&self.cache.visible, &self.cache.next_visible, &self.cache.items);
            self.layout_counts.entered += u64::from(entered);
            self.layout_counts.left += u64::from(left);
        }
        let agent_id = agent_pose.map(AgentPose::agent_id);
        self.motion_redraw = update.rebuild_completed && !self.moving_entities.is_empty();
        if !rebuilding && !update.rebuild_completed {
            for entity in &self.motion_entities {
                let Some(index) = self.cache.objects.get(entity) else { continue };
                let Some(item) = self.cache.items.get(*index) else { continue };
                if !viewed(item, camera, &frustum) {
                    continue;
                }
                self.stats.motion_redraw_items_checked += 1;
                let Some(object) = view.object(*entity).filter(|_| view.object_is_live(*entity)) else { continue };
                if is_agent_pose_override(object.detail().summary.kind, object.detail().summary.id, agent_id) {
                    continue;
                }
                if let (Some(current), Some(future)) = (self.placements.get(entity), self.future_placements.get(entity))
                    && motion::should_redraw(true, *current, *future)
                {
                    self.motion_redraw = true;
                    break;
                }
            }
        }
        let layout_requested = (!rebuilding && !self.draw_list.valid() && !self.draw_list.layout_pending())
            || self.layout_required
            || update.immediate_visibility_change
            || update.layout_invalidated
            || origin_rebased
            || visible_changed;
        if self.draw_list.layout_building() {
            for index in update.pose_items.iter().copied() {
                if let Some(entity) = self.cache.items.get(index).and_then(|item| item.entity) {
                    self.layout_pose_items.insert((index, entity));
                }
            }
        }
        if self.draw_list.layout_pending() {
            if origin_rebased {
                self.draw_list.mark_layout_stale();
            } else if update.layout_invalidated {
                self.layout_required = true;
            }
        }
        if layout_requested {
            self.layout_counts.visibility_resets += u64::from(update.immediate_visibility_change);
            self.layout_counts.invalidations += u64::from(update.layout_invalidated);
            self.layout_counts.rebases += u64::from(origin_rebased);
            let retry = self.layout_required;
            let visible = if visible_changed { &self.cache.next_visible } else { &self.cache.visible };
            self.layout_required = !self.draw_list.request_layout(visible, camera);
            if !self.layout_required {
                self.layout_counts.started += 1;
                self.layout_counts.retries += u64::from(retry);
            }
        }

        let draw_list_started = Instant::now();
        if origin_rebased {
            for (_, item) in self.cache.items.iter_mut() {
                item.transform = instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin);
                if let Some(replacement) = item.replacement.as_mut() {
                    replacement.item.transform = instance_base(
                        replacement.item.position,
                        replacement.item.rotation,
                        replacement.item.scale,
                        replacement.item.invert_normals,
                        origin,
                    );
                }
            }
        }
        let layout_progress = self.draw_list.advance_layout(&mut self.cache.items, LAYOUT_TIME_PER_FRAME);
        let layout_committed = layout_progress.as_ref().is_some_and(|progress| progress.complete);
        if layout_committed {
            self.active_origin = frame_origin;
            self.pipelines.update_camera_uniform(&self.queue, camera, size, self.active_origin);
        }
        if let Some(progress) = layout_progress {
            self.stats.layout_items_visited = progress.stats.items_visited;
            self.stats.layout_faces_visited = progress.stats.faces_visited;
            self.stats.prepare_layout_commit_ms = progress.commit_ms;
            self.layout_required |= progress.stale;
            self.layout_counts.stale += u64::from(progress.stale);
            if progress.complete {
                self.layout_counts.committed += 1;
                self.stats.layouts = 1;
                self.omitted_faces = progress.stats.omitted_faces;
            }
        }
        if layout_committed {
            update.pose_items.extend(
                mem::take(&mut self.layout_pose_items)
                    .into_iter()
                    .filter_map(|(index, entity)| (self.cache.objects.get(&entity) == Some(&index)).then_some(index)),
            );
            update.pose_items.sort_unstable();
            update.pose_items.dedup();
        }
        if layout_committed && self.mesh_admission.capacity_blocked {
            self.mesh_admission.restart_pending = true;
        }
        let layout_time = draw_list_started.elapsed();
        times.layout = layout_time;

        let draw_updates_started = Instant::now();
        self.prepare_face_patches(&update, camera, &frustum, animation_now);
        self.stats.faces_patched =
            self.draw_list.patch(&self.cache.items, &self.cache.patch_faces, &update.pose_items, self.active_origin, frame_origin);
        let draw_upload_started = Instant::now();
        if layout_committed {
            let grown = self.instances.fit(device, &self.queue, &self.memory, self.draw_list.slot_count());
            if grown {
                self.pipelines.set_faces(device, &self.textures, &self.instances.instances);
                self.main_view.clear();
            }
            let uploaded = self.draw_list.upload_layout(&self.queue, &self.instances.instances, grown);
            self.stats.draw_instance_bytes_uploaded += uploaded.bytes;
            self.stats.draw_instance_write_calls += uploaded.calls;
        } else if !self.cache.patch_faces.is_empty() || self.draw_list.has_dirty_slots() {
            let uploaded = self.draw_list.upload_patch(&self.queue, &self.instances.instances);
            self.stats.draw_instance_bytes_uploaded += uploaded.bytes;
            self.stats.draw_instance_write_calls += uploaded.calls;
        }
        times.draw_upload = draw_upload_started.elapsed();
        let draw_updates_time = draw_updates_started.elapsed();
        let draw_batches_started = Instant::now();
        if camera_changed || self.draw_list.cull_dirty() {
            let culled =
                self.draw_list.cull(&self.cache.items, camera, &frustum, &self.meshes, self.indirect_first_instance, &mut self.main_view);
            self.stats.draw_batch_commands_built = culled.commands;
            self.stats.blended_sorts += usize::from(culled.blended_sorted);
            let bytes = self.main_view.indirect_bytes();
            if culled.changed && !bytes.is_empty() {
                self.queue.write_buffer(&self.instances.indirect, 0, bytes);
                self.stats.draw_command_bytes_uploaded = bytes.len();
            }
        }
        let draw_batches_time = draw_batches_started.elapsed();
        times.draw_batches = draw_batches_time;
        times.draw_list = layout_time + draw_updates_time + draw_batches_time;
        let requests_started = Instant::now();
        self.request_frame_textures(view, &update.background);
        times.requests = requests_started.elapsed();
        let texture_started = Instant::now();
        if !same {
            self.textures.upload(device, encoder, self.epoch);
        }
        let texture_changes = self.textures.take_changed_ids();
        let binding_started = Instant::now();
        self.stats.texture_binding_entries =
            self.prepare_texture_bindings(device, layout_committed, &texture_changes, &update.terrain_regions);
        times.texture_bindings += binding_started.elapsed();
        times.textures += texture_started.elapsed();
        if visible_changed {
            mem::swap(&mut self.cache.visible, &mut self.cache.next_visible);
        }
        self.cache.next_visible.clear();
        if !rebuilding {
            self.cache.camera = Some(camera);
            self.cache.origin = origin;
            self.cache.size = size;
        } else {
            self.cache.camera = Some(camera);
            self.cache.size = size;
        }
        let pairs = environment_pairs(view, &self.textures, frame_input.environment);
        self.update_region_water_bounds(visible_changed);
        self.update_environment_uniforms(device, camera, projection, size, frame_input, pairs);
        let animation_deadline = (!rebuilding)
            .then(|| {
                self.cache
                    .animated_items
                    .iter()
                    .filter_map(|index| {
                        let item = self.cache.items.get(*index)?;
                        viewed(item, camera, &frustum).then_some(item.animation_deadline).flatten()
                    })
                    .min()
            })
            .flatten();
        let redraw_deadline = animation_deadline;
        let instances_started = Instant::now();
        let mesh_omitted_faces = if rebuilding || self.mesh_budget_blocked.is_empty() { 0 } else { self.mesh_omitted_faces };
        self.stats.mesh_budget_blocked = self.mesh_budget_blocked.len();
        self.stats.omitted_faces = self.omitted_faces.saturating_add(mesh_omitted_faces);
        self.stats.omission_reason = match (self.omitted_faces > 0, mesh_omitted_faces > 0) {
            (true, true) => Some(super::OmissionReason::CombinedBudgets),
            (true, false) => Some(super::OmissionReason::InstanceBufferBudget),
            (false, true) => Some(super::OmissionReason::MeshMemoryBudget),
            (false, false) => None,
        };
        self.stats.instances = self.draw_list.visible_items();
        let draw_submissions = if self.indirect_first_instance { self.main_view.batches().len() } else { self.main_view.command_count() };
        self.stats.draw_calls = draw_submissions + usize::from(self.void_water.active);
        self.stats.draw_commands = self.main_view.command_count() + usize::from(self.void_water.active);
        self.stats.triangles = self.main_view.triangles();
        self.stats.pending_meshes = self.pending_meshes.len();
        self.stats.pending_textures = self.pending_textures.len();
        if self.recovery_publication_pending
            && self.draw_list.valid()
            && !self.draw_list.layout_pending()
            && !self.layout_required
            && (self.mesh_admission.visible_ready || self.cache.mesh_order.is_empty())
        {
            self.recovery_publication_pending = false;
        }
        self.stats.rebuild_active = self.cache.rebuild.is_some();
        self.stats.rescan_pending = self.cache.rescan_pending;
        self.stats.layout_pending = self.draw_list.layout_pending() || self.layout_required;
        self.stats.layout_counts = self.layout_counts;
        self.stats.rebuild_pending =
            update.rebuild_pending || self.stats.rebuild_active || self.stats.rescan_pending || self.stats.layout_pending;
        if self.stats.omitted_faces > 0 && self.budget_reports.record(Instant::now(), |_| {}).is_some() {
            tracing::warn!(
                omitted_faces = self.stats.omitted_faces,
                mesh_budget_blocked = self.stats.mesh_budget_blocked,
                instance_capacity = self.instances.cap(),
                instance_buffer_bytes = self.instances.bytes(),
                draw_submissions = self.stats.draw_calls,
                "scene geometry exceeded its GPU instance buffer budget"
            );
        }
        self.memory_stats();
        self.prepared = Some(FramePreparation {
            key,
            textures: self.textures.state(),
            geometry_textures: self.textures.geometry_state(),
            meshes: self.memory.mesh_state(),
            more_meshes: update.rebuild_pending
                || self.cache.rebuild.is_some()
                || (self.cache.rescan_pending && !self.recovery_publication_pending)
                || self.draw_list.layout_pending()
                || self.layout_required
                || self.mesh_admission.active
                || self.mesh_admission.restart_pending
                || self.mesh_admission.retry_after_layout
                || self.replacement_retry_pending
                || self.cache.mesh_state != self.memory.mesh_state()
                || self.mesh_admission.revision != self.cache.mesh_demand_revision,
            redraw_deadline,
        });
        times.finalize = instances_started.elapsed();
        self.record_prepare(started, times, uploaded);
    }

    fn advance_mesh_admission(&mut self, device: &Device, uploaded: &mut usize, camera: Camera, frustum: &Frustum) {
        let revision = self.cache.mesh_demand_revision;
        self.pending_meshes.clear();
        if !self.mesh_admission.active || self.mesh_admission.restart_pending {
            self.mesh_admission.begin(revision);
            self.mesh_budget_blocked.clear();
        } else if self.mesh_admission.revision != revision && self.mesh_admission.visible_ready {
            self.mesh_admission.restart(revision);
        }
        self.mesh_admission.visits_this_frame = 0;
        let mut budget = MESH_DEMANDS_PER_FRAME;
        while budget > 0 {
            let next = next_mesh_demand(&self.cache.mesh_order, self.mesh_admission.after);
            let viewed_done = next.is_none_or(|demand| !demand.needs_upload());
            if viewed_done && !self.mesh_admission.visible_ready {
                if self.mesh_admission.pass_incomplete(revision, !self.retired.is_empty()) {
                    self.mesh_admission.restart(revision);
                    continue;
                }
                self.mesh_admission.visible_ready = true;
                self.mesh_admission.blocked = false;
                self.mesh_admission.retry = false;
            }
            let Some(demand) = next else {
                if self.mesh_admission.pass_incomplete(revision, !self.retired.is_empty()) {
                    self.mesh_admission.restart(revision);
                    continue;
                }
                self.mesh_admission.active = false;
                break;
            };
            self.mesh_admission.after = Some(demand);
            budget -= 1;
            self.mesh_admission.visits_this_frame += 1;
            self.stats.mesh_admission_items_visited += 1;
            if self.meshes.contains_key(&demand.mesh) {
                if demand.in_view {
                    self.cache.admission_protected.insert(demand.mesh);
                }
                continue;
            }
            match self.admit_mesh_demand(device, demand, uploaded, camera, frustum) {
                MeshUpload::Ready => {}
                MeshUpload::Deferred => {
                    self.mesh_admission.blocked = true;
                    self.mesh_admission.retry = true;
                }
                MeshUpload::CapacityBlocked => {
                    self.mesh_admission.blocked = true;
                    self.mesh_admission.capacity_blocked = true;
                    self.mesh_admission.retry_after_layout |= self.draw_list.layout_pending();
                }
            }
        }
        self.mesh_admission.revision = revision;
    }

    fn admit_mesh_demand(
        &mut self,
        device: &Device,
        demand: MeshDemandOrder,
        uploaded: &mut usize,
        camera: Camera,
        frustum: &Frustum,
    ) -> MeshUpload {
        let mesh_id = demand.mesh;
        let Some(mesh) = self.cache.mesh_representatives.get(&mesh_id).cloned() else { return MeshUpload::Ready };
        let composition = self.cache.mesh_compositions.get(&mesh_id).cloned().flatten();
        self.stats.mesh_upload_attempts += 1;
        self.stats.mesh_upload_attempt_bytes += mesh.mesh().owned_bytes();
        let eviction = demand.in_view.then_some((camera, frustum));
        let outcome = self.upload_mesh(device, &mesh, composition.as_deref(), uploaded, eviction);
        if demand.in_view {
            match outcome {
                MeshUpload::Ready => {
                    self.cache.admission_protected.insert(mesh_id);
                }
                MeshUpload::Deferred => {
                    self.pending_meshes.insert(mesh_id);
                }
                MeshUpload::CapacityBlocked => {
                    self.pending_meshes.insert(mesh_id);
                    self.mesh_budget_blocked.insert(mesh_id);
                }
            }
        }
        outcome
    }

    fn update_environment_uniforms(
        &mut self,
        device: &Device,
        camera: Camera,
        projection: Mat4,
        size: [u32; 2],
        frame_input: FrameInput<'_>,
        pairs: EnvironmentPairs,
    ) {
        let normal = pairs.water_normal;
        self.pipelines.update_environment(
            device,
            &self.textures,
            &self.queue,
            camera.position(),
            camera.position() - self.active_origin,
            frame_input.environment,
            normal,
            frame_input.water_height,
            frame_input.water_scroll,
        );
        self.void_water.update(
            device,
            &self.textures,
            &self.queue,
            camera,
            projection.inverse(),
            frame_input.water_height,
            frame_input.environment,
            normal,
            frame_input.water_scroll,
            &self.region_water_bounds,
            self.region_water_count,
            (camera.position() - self.active_origin).as_vec3(),
        );
        self.sky.update(
            device,
            &self.textures,
            &self.queue,
            projection,
            projection.inverse(),
            size,
            frame_input.environment,
            frame_input.cloud_scroll,
            frame_input.animation_time,
            pairs.sun,
            pairs.moon,
            pairs.cloud,
            pairs.bloom,
            pairs.rainbow,
            pairs.halo,
        );
    }

    fn update_region_water_bounds(&mut self, visible_changed: bool) {
        if !visible_changed && !self.cache.region_water_changed && self.region_water_origin == Some(self.active_origin) {
            return;
        }
        self.cache.region_water_changed = false;
        self.region_water_origin = Some(self.active_origin);
        self.region_water_count = 0;
        for index in self.cache.region_water.values().copied() {
            if !self.cache.contains(&self.cache.visible, index) {
                continue;
            }
            let Some(item) = self.cache.items.get(index) else { continue };
            let relative = (item.position - self.active_origin).as_vec3();
            let Some(bounds) = self.region_water_bounds.get_mut(self.region_water_count).filter(|_| relative.is_finite()) else {
                continue;
            };
            *bounds = math::Vec4::new(relative.x, relative.y, 256.0, 256.0);
            self.region_water_count += 1;
        }
    }

    fn apply_agent_pose(&mut self, agent_pose: Option<AgentPose>, origin: DVec3, update: &mut CacheUpdate) {
        let agent_id = agent_pose.map(AgentPose::agent_id);
        if let Some(previous) = self.applied_agent.filter(|previous| Some(*previous) != agent_id) {
            self.set_agent_item_pose(previous, None, origin, Some(&mut *update));
        }
        if let Some(agent) = agent_pose {
            self.set_agent_item_pose(agent.agent_id(), Some(agent.pose()), origin, Some(&mut *update));
        }
        self.applied_agent = agent_id;
    }

    pub(super) fn restore_agent_pose(&mut self, agent_id: Uuid) {
        let origin = self.cache.origin;
        self.set_agent_item_pose(agent_id, None, origin, None);
    }

    fn set_agent_item_pose(&mut self, agent_id: Uuid, pose_override: Option<WorldPose>, origin: DVec3, update: Option<&mut CacheUpdate>) {
        let Some(index) = self.cache.index((2, agent_id.as_u128(), 0)) else { return };
        let Some(item) = self.cache.items.get(index) else { return };
        let Some(object) = item.source.as_ref() else { return };
        if object.detail().summary.kind != Kind::Avatar {
            return;
        }
        let Placement::Ready(transform) = object.placement() else { return };
        let pose = pose_override.unwrap_or(transform.pose);
        self.set_item_pose(index, pose, origin, update);
    }

    fn set_item_pose(&mut self, index: ItemIndex, pose: WorldPose, origin: DVec3, update: Option<&mut CacheUpdate>) {
        let Some(item) = self.cache.items.get_mut(index) else { return };
        let changed = set_cached_item_pose(item, pose, origin);
        let replacement_changed =
            item.replacement.as_mut().is_some_and(|replacement| set_cached_item_pose(&mut replacement.item, pose, origin));
        if !changed && !replacement_changed {
            return;
        }
        if let Some(update) = update {
            update.pose_items.push(index);
            update.items_updated += 1;
        }
    }

    fn update_object_motion(&mut self, view: &SceneView, now: Instant, origin: DVec3, agent_id: Option<Uuid>, update: &mut CacheUpdate) {
        self.motion_entities.clear();
        let mut newly_moving = false;
        for entity in &update.background {
            let was_moving = self.moving_entities.contains(entity);
            let index = self.cache.objects.get(entity);
            self.stats.motion_cache_items_scanned += usize::from(index.is_some());
            let is_moving = index.is_some() && view.object_is_live(*entity) && motion::needs_evaluation(view, *entity);
            if is_moving {
                self.moving_entities.insert(*entity);
                newly_moving |= !was_moving;
            } else {
                self.moving_entities.remove(entity);
            }
            if was_moving && !is_moving {
                self.motion_entities.push(*entity);
            }
        }
        if !update.motion_changed.is_empty() || newly_moving {
            self.moving_entities.clear();
            self.motion_entities.clear();
            for (_, item) in self.cache.items.iter() {
                self.stats.motion_cache_items_scanned += 1;
                let Some(entity) = item.entity.filter(|entity| view.object_is_live(*entity)) else { continue };
                let needs_evaluation = motion::needs_evaluation(view, entity);
                if needs_evaluation {
                    self.moving_entities.insert(entity);
                }
                if needs_evaluation || motion::has_changed_ancestor(view, entity, &update.motion_changed) {
                    self.motion_entities.push(entity);
                }
            }
        } else {
            self.motion_entities.extend(self.moving_entities.iter().copied());
        }
        motion::fill_placement_pair(
            view,
            now,
            now + Duration::from_millis(16),
            &self.motion_entities,
            &mut self.placements,
            &mut self.future_placements,
            &mut self.regions,
        );
        self.stats.motion_entities_evaluated += self.motion_entities.len();
        let motion_entities = mem::take(&mut self.motion_entities);
        for &entity in &motion_entities {
            let Some(index) = self.cache.objects.get(&entity).copied() else { continue };
            let Some(object) = view.object(entity).filter(|_| view.object_is_live(entity)) else { continue };
            if is_agent_pose_override(object.detail().summary.kind, object.detail().summary.id, agent_id) {
                continue;
            }
            let Some(pose) = self.placements.get(&entity).map(|transform| transform.pose) else { continue };
            self.set_item_pose(index, pose, origin, Some(update));
        }
        self.motion_entities = motion_entities;
    }

    fn update_object_motion_page(
        &mut self,
        view: &SceneView,
        now: Instant,
        origin: DVec3,
        entities: &[world::Entity],
        agent_id: Option<Uuid>,
        update: &mut CacheUpdate,
    ) {
        self.motion_entities.clear();
        self.motion_entities.extend_from_slice(entities);
        self.stats.motion_entities_evaluated += entities.len();
        motion::fill_placement_pair(
            view,
            now,
            now + Duration::from_millis(16),
            &self.motion_entities,
            &mut self.placements,
            &mut self.future_placements,
            &mut self.regions,
        );
        for entity in entities {
            let Some(index) = self.cache.objects.get(entity).copied() else {
                self.moving_entities.remove(entity);
                continue;
            };
            self.stats.motion_cache_items_scanned += 1;
            if view.object_is_live(*entity) && motion::needs_evaluation(view, *entity) {
                self.moving_entities.insert(*entity);
            } else {
                self.moving_entities.remove(entity);
            }
            let Some(object) = view.object(*entity) else { continue };
            if is_agent_pose_override(object.detail().summary.kind, object.detail().summary.id, agent_id) {
                continue;
            }
            let Some(pose) = self.placements.get(entity).map(|transform| transform.pose) else { continue };
            self.set_item_pose(index, pose, origin, Some(update));
        }
    }

    fn refresh_motion_membership(&mut self, view: &SceneView, entities: &[world::Entity]) {
        for entity in entities {
            if !self.cache.objects.contains_key(entity) {
                self.moving_entities.remove(entity);
                continue;
            }
            self.stats.motion_cache_items_scanned += 1;
            if view.object_is_live(*entity) && motion::needs_evaluation(view, *entity) {
                self.moving_entities.insert(*entity);
            } else {
                self.moving_entities.remove(entity);
            }
        }
    }

    fn prepare_face_patches(&mut self, update: &CacheUpdate, camera: Camera, frustum: &Frustum, now: Instant) {
        let items = &mut self.cache.items;
        let patch_items = &mut self.cache.patch_items;
        patch_items.clear();
        patch_items.extend(update.pose_items.iter().copied());
        patch_items.extend(update.parameter_items.iter().copied());
        for id in self.cache.animated_items.iter().copied() {
            let Some(item) = items.get(id) else { continue };
            if viewed(item, camera, frustum) && item.animation_deadline.is_some_and(|deadline| deadline <= now) {
                patch_items.push(id);
            }
        }
        patch_items.sort_unstable();
        patch_items.dedup();
        self.cache.patch_faces.clear();
        for id in patch_items.iter().copied() {
            let pose_changed = update.pose_items.binary_search(&id).is_ok();
            let Some(item) = items.get_mut(id) else { continue };
            let parameters_dirty = item.parameters_dirty;
            let animation_due = item.animation_deadline.is_some_and(|deadline| deadline <= now);
            let animation = item
                .source
                .as_ref()
                .and_then(|object| object.detail().texture_animation)
                .filter(|animation| texture_animation::is_enabled(*animation));
            let full_faces = pose_changed || parameters_dirty;
            let animated_faces = &item.animated_faces;
            let source = item.source.as_ref();
            let shading = item.shading;
            let scale = item.scale;
            let faces = &mut item.faces;
            let count = if full_faces { faces.len() } else { animated_faces.len() };
            for cursor in 0..count {
                let face_index = if full_faces { cursor } else { animated_faces.get(cursor).copied().unwrap_or(usize::MAX) };
                let Some(face) = faces.get_mut(face_index) else { continue };
                let animated = animation_due
                    && face.base_params.textures.has_images()
                    && animation.is_some_and(|animation| animation.targets_face(face.order.3));
                let refresh_params = parameters_dirty || animated;
                let params_changed = if refresh_params {
                    let params = current_params(source, shading, face, now);
                    let changed = params != face.params;
                    if changed {
                        face.params = params;
                        face.parameter_bytes = parameter_part(&params, scale);
                    }
                    changed
                } else {
                    false
                };
                if animated {
                    self.stats.animated_faces_visited += 1;
                    self.stats.animated_faces_changed += usize::from(params_changed);
                }
                if pose_changed || params_changed {
                    self.cache.patch_faces.push((id, face_index));
                }
            }
            item.parameters_dirty = false;
            if animation_due || parameters_dirty {
                item.animation_deadline = next_animation_deadline(item, now);
            }
        }
    }

    fn record_prepare(&mut self, started: Instant, times: PrepareTimes, uploaded: usize) {
        self.stats.prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.stats.prepare_cache_ms = times.cache.as_secs_f64() * 1000.0;
        self.stats.prepare_visibility_ms = times.visibility.as_secs_f64() * 1000.0;
        self.stats.prepare_motion_ms = times.motion.as_secs_f64() * 1000.0;
        self.stats.prepare_candidates_ms = times.candidates.as_secs_f64() * 1000.0;
        self.stats.prepare_requests_ms = times.requests.as_secs_f64() * 1000.0;
        self.stats.prepare_textures_ms = times.textures.as_secs_f64() * 1000.0;
        self.stats.prepare_texture_bindings_ms = times.texture_bindings.as_secs_f64() * 1000.0;
        self.stats.prepare_meshes_ms = times.meshes.as_secs_f64() * 1000.0;
        self.stats.prepare_mesh_demands_ms = times.mesh_demands.as_secs_f64() * 1000.0;
        self.stats.prepare_mesh_uploads_ms = times.mesh_uploads.as_secs_f64() * 1000.0;
        self.stats.prepare_appearances_ms = times.appearances.as_secs_f64() * 1000.0;
        self.stats.prepare_draw_list_ms = times.draw_list.as_secs_f64() * 1000.0;
        self.stats.prepare_layout_ms = times.layout.as_secs_f64() * 1000.0;
        self.stats.prepare_draw_upload_ms = times.draw_upload.as_secs_f64() * 1000.0;
        self.stats.prepare_draw_batches_ms = times.draw_batches.as_secs_f64() * 1000.0;
        self.stats.prepare_finalize_ms = times.finalize.as_secs_f64() * 1000.0;
        if self.stats.prepare_ms >= 16.0 {
            let frame = SlowPrepare {
                total_ms: self.stats.prepare_ms,
                visibility_ms: times.visibility.as_secs_f64() * 1000.0,
                texture_requests_ms: times.requests.as_secs_f64() * 1000.0,
                texture_upload_ms: times.textures.as_secs_f64() * 1000.0,
                mesh_upload_ms: times.meshes.as_secs_f64() * 1000.0,
                draw_list_ms: times.draw_list.as_secs_f64() * 1000.0,
                mesh_uploads: uploaded,
                mesh_upload_bytes: self.stats.uploaded_bytes,
                draw_submissions: self.stats.draw_calls,
                draw_commands: self.stats.draw_commands,
                instances: self.stats.instances,
                pending_meshes: self.stats.pending_meshes,
                pending_textures: self.stats.pending_textures,
                layouts: self.stats.layouts,
                items_updated: self.stats.items_updated,
                faces_patched: self.stats.faces_patched,
                blended_sorts: self.stats.blended_sorts,
                animated_faces_visited: self.stats.animated_faces_visited,
                animated_faces_changed: self.stats.animated_faces_changed,
                omitted_faces: self.stats.omitted_faces,
            };
            let due = self.slow_prepares.record(Instant::now(), |slow| {
                slow.frames = slow.frames.saturating_add(1);
                if slow.worst.as_ref().is_none_or(|worst| frame.total_ms > worst.total_ms) {
                    slow.worst = Some(frame);
                }
            });
            if let Some(due) = due {
                due.report();
            }
        }
    }
}

#[derive(Default)]
pub(super) struct SlowPrepares {
    frames: u64,
    worst: Option<SlowPrepare>,
}

impl SlowPrepares {
    pub(super) fn report(self) {
        let Some(worst) = self.worst else { return };
        tracing::debug!(
            slow_frames = self.frames,
            worst_ms = worst.total_ms,
            visibility_ms = worst.visibility_ms,
            texture_requests_ms = worst.texture_requests_ms,
            texture_upload_ms = worst.texture_upload_ms,
            mesh_upload_ms = worst.mesh_upload_ms,
            draw_list_ms = worst.draw_list_ms,
            mesh_uploads = worst.mesh_uploads,
            mesh_upload_bytes = worst.mesh_upload_bytes,
            draw_submissions = worst.draw_submissions,
            draw_commands = worst.draw_commands,
            instances = worst.instances,
            pending_meshes = worst.pending_meshes,
            pending_textures = worst.pending_textures,
            layouts = worst.layouts,
            items_updated = worst.items_updated,
            faces_patched = worst.faces_patched,
            blended_sorts = worst.blended_sorts,
            animated_faces_visited = worst.animated_faces_visited,
            animated_faces_changed = worst.animated_faces_changed,
            omitted_faces = worst.omitted_faces,
            "slow scene preparation"
        );
    }
}

struct SlowPrepare {
    total_ms: f64,
    visibility_ms: f64,
    texture_requests_ms: f64,
    texture_upload_ms: f64,
    mesh_upload_ms: f64,
    draw_list_ms: f64,
    mesh_uploads: usize,
    mesh_upload_bytes: usize,
    draw_submissions: usize,
    draw_commands: usize,
    instances: usize,
    pending_meshes: usize,
    pending_textures: usize,
    layouts: usize,
    items_updated: usize,
    faces_patched: usize,
    blended_sorts: usize,
    animated_faces_visited: usize,
    animated_faces_changed: usize,
    omitted_faces: usize,
}

fn set_cached_item_pose(item: &mut CachedItem, pose: WorldPose, origin: DVec3) -> bool {
    if item.position == pose.position && item.rotation == pose.rotation {
        return false;
    }
    item.position = pose.position;
    item.rotation = pose.rotation;
    item.transform = instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin);
    true
}

fn is_agent_pose_override(kind: Kind, object_id: Uuid, agent_id: Option<Uuid>) -> bool {
    kind == Kind::Avatar && agent_id == Some(object_id)
}

fn current_params(source: Option<&Object>, shading: ShadingMode, face: &CachedFace, now: Instant) -> FaceParams {
    let Some(object) = source else { return face.base_params };
    let Some(animation) = object.detail().texture_animation else { return face.base_params };
    if !texture_animation::is_enabled(animation) || !animation.targets_face(face.order.3) {
        return face.base_params;
    }
    let Some((elapsed, offset)) = object.animation_timing(now) else { return face.base_params };
    let transform = texture_animation::transform_at(animation, elapsed, offset, face.texture_entry_transform);
    let material = object.materials().and_then(|materials| face.material_index.and_then(|index| materials.get(index)));
    let mut params = FaceParams::with_texture_entry_transform(material, Some(transform));
    match shading {
        ShadingMode::Terrain => {
            params.color = [0.18, 0.23, 0.13, 1.0];
            params.params[0] = 1.0;
        }
        ShadingMode::RegionWater | ShadingMode::Surface => {}
    }
    params
}

fn viewed(item: &CachedItem, camera: Camera, frustum: &Frustum) -> bool {
    WorldSphere::new(item.position, item.rotation, item.scale, item.info.bounds, camera)
        .is_some_and(|sphere| sphere.visible(item.shading, camera, frustum))
}

fn next_animation_deadline(item: &CachedItem, now: Instant) -> Option<Instant> {
    if !item.animated_faces.iter().any(|index| item.faces.get(*index).is_some_and(|face| face.slot.is_some())) {
        return None;
    }
    let source = item.source.as_ref()?;
    let animation = source.detail().texture_animation?;
    let (elapsed, offset) = source.animation_timing(now)?;
    texture_animation::next_update_after(animation, elapsed, offset).and_then(|delay| now.checked_add(delay))
}

fn request_environment_textures(view: &SceneView, environment: &Evaluated, textures: &mut Textures, epoch: u64) {
    for blend in [
        environment.sky.sun.texture,
        environment.sky.moon.texture,
        environment.sky.clouds.texture,
        environment.sky.bloom,
        environment.sky.rainbow,
        environment.sky.halo,
        environment.water.normal_map,
    ] {
        for id in [blend.current, blend.next].into_iter().flatten() {
            if let Some(handle) = view.environment_texture(id) {
                textures.request_environment(handle, epoch);
            }
        }
    }
}

fn environment_pairs(view: &SceneView, textures: &Textures, environment: Option<&Evaluated>) -> EnvironmentPairs {
    let Some(environment) = environment else { return EnvironmentPairs::default() };
    EnvironmentPairs {
        sun: blend_keys(environment.sky.sun.texture, view, textures),
        moon: blend_keys(environment.sky.moon.texture, view, textures),
        cloud: blend_keys(environment.sky.clouds.texture, view, textures),
        bloom: blend_keys(environment.sky.bloom, view, textures),
        rainbow: blend_keys(environment.sky.rainbow, view, textures),
        halo: blend_keys(environment.sky.halo, view, textures),
        water_normal: blend_keys(environment.water.normal_map, view, textures),
    }
}

#[cfg(test)]
mod tests {
    use math::{DVec3, Vec3};

    use super::*;
    use crate::{
        geometry::{
            MeshState,
            cache::object_item_id,
            draw_list::DrawList,
            fixtures,
            items::{CachedFaceContext, cached_faces, mesh_info, surface_pipeline},
        },
        scene::{Budget, Placement, Reconciler},
        texture::{GeometryTextureState, TextureState},
    };

    fn prepare_key(view: &SceneView) -> FramePreparationKey {
        FramePreparationKey {
            version: view.geometry_version(),
            camera: Camera::new(DVec3::ZERO, Vec3::X, 1.0, [720, 720]).expect("fixture camera"),
            agent_pose: None,
            size: [1280, 720],
        }
    }

    fn texture_state() -> TextureState {
        TextureState { revision: 0, staging: 0, jobs: 0, lifecycle: 0, cap: 1024 * 1024 * 1024 }
    }

    #[test]
    fn preparation_tracks_scene_inputs() {
        let (mut scene, view) = Reconciler::new(Budget::default());
        let (_other_scene, other) = Reconciler::new(Budget::default());
        let prepared = FramePreparation {
            key: prepare_key(&view),
            textures: texture_state(),
            geometry_textures: GeometryTextureState::default(),
            meshes: MeshState::default(),
            more_meshes: false,
            redraw_deadline: None,
        };
        let now = Instant::now();
        assert!(prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), now));
        let mut moved = prepare_key(&view);
        moved.camera = Camera::new(DVec3::X, Vec3::X, 1.0, [720, 720]).expect("moved camera");
        assert!(!prepared.reusable(&moved, GeometryTextureState::default(), MeshState::default(), now));
        let mut resized = prepare_key(&view);
        resized.size[0] = 1600;
        assert!(!prepared.reusable(&resized, GeometryTextureState::default(), MeshState::default(), now));
        let mut quality = prepare_key(&view);
        quality.camera = quality.camera.with_quality(128.0, 2.0).expect("changed quality");
        assert!(!prepared.reusable(&quality, GeometryTextureState::default(), MeshState::default(), now));
        assert!(!prepared.reusable(&prepare_key(&other), GeometryTextureState::default(), MeshState::default(), now));
        scene.close();
        assert!(!prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), now));
    }

    #[test]
    fn preparation_tracks_gpu_progress() {
        let (_scene, view) = Reconciler::new(Budget::default());
        let state = texture_state();
        let mut prepared = FramePreparation {
            key: prepare_key(&view),
            textures: state,
            geometry_textures: GeometryTextureState::default(),
            meshes: MeshState::default(),
            more_meshes: false,
            redraw_deadline: None,
        };
        let now = Instant::now();
        for (reason, changed) in [
            ("image readiness or cancellation", TextureState { revision: 1, ..state }),
            ("handle release or asset change", TextureState { lifecycle: 1, ..state }),
            ("staging completion", TextureState { staging: 256, ..state }),
            ("conversion admission", TextureState { jobs: 1, ..state }),
            ("soft cap change", TextureState { cap: 64 * 1024 * 1024, ..state }),
        ] {
            assert!(
                prepared.needs_frame(changed, GeometryTextureState::default(), MeshState::default(), now),
                "{reason} must allow texture progress to schedule a frame"
            );
            assert!(prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), now));
        }
        assert!(!prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState { bytes: 4, revision: 0 }, now));
        assert!(!prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState { bytes: 0, revision: 1 }, now));
        assert!(prepared.needs_frame(state, GeometryTextureState(1), MeshState::default(), now));
        prepared.more_meshes = true;
        assert!(!prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), now));
    }

    #[test]
    fn stepped_animation_waits_for_deadline() {
        let (_scene, view) = Reconciler::new(Budget::default());
        let state = texture_state();
        let now = Instant::now();
        let deadline = now + Duration::from_millis(500);
        let mut prepared = FramePreparation {
            key: prepare_key(&view),
            textures: state,
            geometry_textures: GeometryTextureState::default(),
            meshes: MeshState::default(),
            more_meshes: false,
            redraw_deadline: Some(deadline),
        };
        assert!(!prepared.needs_frame(state, GeometryTextureState::default(), MeshState::default(), now));
        assert!(prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), now));
        assert!(prepared.needs_frame(state, GeometryTextureState::default(), MeshState::default(), deadline));
        assert!(!prepared.reusable(&prepare_key(&view), GeometryTextureState::default(), MeshState::default(), deadline));
        prepared.redraw_deadline = None;
        assert!(!prepared.needs_frame(state, GeometryTextureState::default(), MeshState::default(), deadline));
    }

    #[test]
    fn scale_winding_rekeys_faces() {
        let (_scene, view, _) = fixtures::scene(255);
        let object = view.objects().next().expect("fixture object");
        let (_, mesh) = object.mesh().expect("fixture mesh");
        let Placement::Ready(transform) = object.placement() else { panic!("fixture placement") };
        let info = mesh_info(mesh);
        let mut draw_list = DrawList::new(usize::MAX);
        let mut faces = cached_faces(
            CachedFaceContext {
                id: object_item_id(object),
                mesh,
                info: &info,
                materials: object.materials(),
                appearance: object.detail().surface_appearance.as_ref(),
                scale: transform.scale,
                invert_normals: false,
                shading: ShadingMode::Surface,
                terrain: None,
                terrain_textures: None,
            },
            &mut draw_list,
        );
        let original = draw_list.face_key(faces[0].group).expect("fixture draw group").pipeline;
        let flipped = Vec3::new(-transform.scale.x, transform.scale.y, transform.scale.z);
        assert!(draw_list.refresh_winding(&mut faces, transform.scale, false, flipped, false));
        let key = draw_list.face_key(faces[0].group).expect("flipped draw group");
        assert_ne!(key.pipeline, original);
        assert_eq!(key.pipeline, surface_pipeline(&faces[0].base_params, flipped, false));
        assert!(!draw_list.refresh_winding(&mut faces, flipped, false, flipped, false));
    }
}
