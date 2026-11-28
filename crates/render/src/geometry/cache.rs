use std::{
    array,
    cmp::Ordering,
    collections::{BTreeSet, HashMap, HashSet},
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use math::{Frustum, Quat, Vec3};
use uuid::Uuid;
use world::{Entity, Region};

use super::{
    GeometryRenderer, MeshState, ShadingMode,
    instance::{instance_base, parameter_part},
    items::{
        CachedFace, CachedFaceContext, CachedItem, CachedReplacement, ItemId, ItemIndex, ItemSlab, MeshInfo, cached_faces, draw_priority,
        draw_scale, mesh_info, object_draw_scale, same_surface_geometry,
    },
    visibility::WorldSphere,
};
use crate::scene::{
    Camera, DrawnMeshKey, MeshId, Object, Placement, PreparedMesh, RANGE_MARGIN, SceneView, SharedMesh, Surface, SurfaceKind, ViewVersion,
    VisibilityVersion, texture_animation,
};

const REBUILD_ITEMS_PER_FRAME: usize = 1_024;
const REBUILD_BYTES_PER_FRAME: usize = 8 * 1024 * 1024;
const REBUILD_TIME_PER_FRAME: Duration = Duration::from_micros(1_000);
const REBUILD_CHANGE_TIME_PER_FRAME: Duration = Duration::from_micros(250);

#[derive(Clone, Copy)]
enum RebuildPhase {
    Objects,
    Surfaces,
    Sweep,
    PruneMeshInfo,
    VisibilityObjects,
    VisibilitySurfaces,
    Finalize,
}

pub(super) struct CacheRebuild {
    phase: RebuildPhase,
    sweep_needed: bool,
    object_after: Option<Uuid>,
    surface_after: Option<(Region, SurfaceKind)>,
    visibility_object_after: Option<Uuid>,
    visibility_surface_after: Option<(Region, SurfaceKind)>,
    sweep_cursor: usize,
    sweep_limit: usize,
    object_change_since: u64,
    object_change_after: Option<(u64, Uuid)>,
    added_object_after: Option<(u64, Uuid)>,
    object_change_page_sequence: u64,
    object_change_page_complete: bool,
    added_object_page_complete: bool,
    objects_done: bool,
    surface_rescan_pending: bool,
    mesh_info_cursor: usize,
    retained_meshes: HashSet<MeshId>,
    mesh_info_cleanup: Option<MeshInfo>,
    layout_changed: bool,
    mesh_dirty: bool,
    pub(super) visibility_changed: bool,
    pub(super) mesh_demand_missing: bool,
    observed_version: ViewVersion,
    observed_visibility: VisibilityVersion,
    visibility_restart: bool,
    visibility_restarted: bool,
    candidate_camera: Option<Camera>,
    mesh_info_build: Option<MeshInfoBuild>,
    #[cfg(test)]
    scanned_objects: usize,
}

impl CacheRebuild {
    fn new(view: &SceneView, sweep_limit: usize, sweep_needed: bool) -> Self {
        Self {
            phase: RebuildPhase::Surfaces,
            sweep_needed,
            object_after: None,
            surface_after: None,
            visibility_object_after: None,
            visibility_surface_after: None,
            sweep_cursor: 0,
            sweep_limit,
            object_change_since: view.sequence(),
            object_change_after: None,
            added_object_after: None,
            object_change_page_sequence: view.sequence(),
            object_change_page_complete: false,
            added_object_page_complete: false,
            objects_done: false,
            surface_rescan_pending: false,
            mesh_info_cursor: 0,
            retained_meshes: HashSet::new(),
            mesh_info_cleanup: None,
            layout_changed: false,
            mesh_dirty: false,
            visibility_changed: false,
            mesh_demand_missing: false,
            observed_version: view.version(),
            observed_visibility: view.visibility_version(),
            visibility_restart: false,
            visibility_restarted: false,
            candidate_camera: None,
            mesh_info_build: None,
            #[cfg(test)]
            scanned_objects: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn progress(&self) -> String {
        let phase = match self.phase {
            RebuildPhase::Objects => "objects",
            RebuildPhase::Surfaces => "surfaces",
            RebuildPhase::Sweep => "sweep",
            RebuildPhase::PruneMeshInfo => "prune_mesh_info",
            RebuildPhase::VisibilityObjects => "visibility_objects",
            RebuildPhase::VisibilitySurfaces => "visibility_surfaces",
            RebuildPhase::Finalize => "finalize",
        };
        format!(
            "phase={phase},object_cursor={},surface_cursor={},visibility_object_cursor={},visibility_surface_cursor={},sweep_cursor={},mesh_info_cursor={},surface_rescan_pending={},visibility_restart={},object_change_since={},object_change_page_sequence={},object_change_cursor={},added_object_cursor={},object_change_page_complete={},added_object_page_complete={},scanned_objects={}",
            self.object_after.is_some(),
            self.surface_after.is_some(),
            self.visibility_object_after.is_some(),
            self.visibility_surface_after.is_some(),
            self.sweep_cursor,
            self.mesh_info_cursor,
            self.surface_rescan_pending,
            self.visibility_restart,
            self.object_change_since,
            self.object_change_page_sequence,
            self.object_change_after.is_some(),
            self.added_object_after.is_some(),
            self.object_change_page_complete,
            self.added_object_page_complete,
            self.scanned_objects,
        )
    }

    fn finish_objects(&mut self, slot_count: usize) {
        self.objects_done = true;
        self.mesh_info_build = None;
        if self.surface_rescan_pending {
            self.surface_rescan_pending = false;
            self.surface_after = None;
            self.phase = RebuildPhase::Surfaces;
        } else {
            self.finish_surfaces(slot_count);
        }
    }

    fn finish_surfaces(&mut self, slot_count: usize) {
        self.mesh_info_build = None;
        if !self.objects_done {
            self.phase = RebuildPhase::Objects;
        } else if self.surface_rescan_pending {
            self.surface_rescan_pending = false;
            self.surface_after = None;
            self.phase = RebuildPhase::Surfaces;
        } else if self.sweep_needed {
            self.phase = RebuildPhase::Sweep;
            self.sweep_limit = slot_count;
        } else {
            self.phase = RebuildPhase::PruneMeshInfo;
        }
    }

    fn finish_mesh_info(&mut self) {
        if self.visibility_restart {
            self.phase = RebuildPhase::VisibilityObjects;
            self.visibility_restart = false;
            self.visibility_object_after = None;
            self.visibility_surface_after = None;
        } else {
            self.phase = RebuildPhase::Finalize;
        }
    }

    fn note_camera(&mut self, camera: Camera) -> Camera {
        let Some(previous) = self.candidate_camera else {
            self.candidate_camera = Some(camera);
            return camera;
        };
        if !range_moved(previous, camera) {
            return previous;
        }
        self.layout_changed = true;
        if matches!(self.phase, RebuildPhase::VisibilityObjects | RebuildPhase::VisibilitySurfaces | RebuildPhase::Finalize) {
            return previous;
        }
        self.visibility_restart = true;
        self.candidate_camera = Some(camera);
        camera
    }
}

struct MeshInfoBuild {
    mesh: SharedMesh,
    face: usize,
    index: usize,
    min: Vec3,
    max: Vec3,
    centers: Vec<Vec3>,
}

#[derive(Clone, Copy)]
pub(super) struct MeshDemandOrder {
    pub(super) mesh: MeshId,
    resident: bool,
    pub(super) in_view: bool,
    priority: (f64, f64),
}

impl MeshDemandOrder {
    pub(super) fn needs_upload(self) -> bool {
        !self.resident && self.in_view
    }
}

impl PartialEq for MeshDemandOrder {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for MeshDemandOrder {}

impl PartialOrd for MeshDemandOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeshDemandOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .needs_upload()
            .cmp(&self.needs_upload())
            .then_with(|| other.resident.cmp(&self.resident))
            .then_with(|| other.priority.0.total_cmp(&self.priority.0))
            .then_with(|| self.priority.1.total_cmp(&other.priority.1))
            .then_with(|| self.mesh.cmp(&other.mesh))
    }
}

impl MeshInfoBuild {
    fn new(mesh: SharedMesh) -> Self {
        let centers = Vec::new();
        Self { mesh, face: 0, index: 0, min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY), centers }
    }

    fn advance(&mut self, byte_limit: usize, face_limit: usize, time_limit: Duration) -> (usize, usize) {
        let source = self.mesh.mesh();
        let started = Instant::now();
        let mut bytes: usize = 0;
        let mut faces = 0;
        let bytes_per_index = size_of::<u32>() + size_of::<Vec3>();
        while let Some(face) = source.faces().get(self.face) {
            if self.index.is_multiple_of(64) && started.elapsed() >= time_limit {
                break;
            }
            let range = usize::try_from(face.indices.start)
                .ok()
                .zip(usize::try_from(face.indices.end).ok())
                .filter(|(start, end)| start <= end)
                .and_then(|(start, end)| source.indices().get(start..end));
            let Some(range) = range else {
                if faces >= face_limit {
                    break;
                }
                self.centers.push(Vec3::ZERO);
                self.face += 1;
                faces += 1;
                self.index = 0;
                self.min = Vec3::splat(f32::INFINITY);
                self.max = Vec3::splat(f32::NEG_INFINITY);
                continue;
            };
            let Some(&vertex_index) = range.get(self.index) else {
                if faces >= face_limit {
                    break;
                }
                self.centers.push(if self.min.is_finite() && self.max.is_finite() { self.min * 0.5 + self.max * 0.5 } else { Vec3::ZERO });
                self.face += 1;
                faces += 1;
                self.index = 0;
                self.min = Vec3::splat(f32::INFINITY);
                self.max = Vec3::splat(f32::NEG_INFINITY);
                continue;
            };
            if bytes.saturating_add(bytes_per_index) > byte_limit {
                break;
            }
            if let Some(vertex) = usize::try_from(vertex_index).ok().and_then(|index| source.vertices().get(index)) {
                self.min = self.min.min(vertex.position);
                self.max = self.max.max(vertex.position);
            }
            self.index += 1;
            bytes += bytes_per_index;
        }
        (bytes, faces)
    }

    fn is_complete(&self) -> bool {
        self.face == self.mesh.mesh().faces().len()
    }
}

#[derive(Default)]
pub(super) struct FrameCache {
    pub(super) items: ItemSlab,
    item_indices: HashMap<ItemId, ItemIndex>,
    pub(super) objects: HashMap<Entity, ItemIndex>,
    pub(super) replacements: BTreeSet<ItemIndex>,
    pub(super) mesh_info: HashMap<MeshId, Arc<MeshInfo>>,
    mesh_info_order: Vec<MeshId>,
    item_order: Vec<ItemIndex>,
    item_order_dirty: bool,
    pub(super) range: Vec<ItemIndex>,
    next_range: Vec<ItemIndex>,
    pub(super) range_camera: Option<Camera>,
    pub(super) nonresident_demands: bool,
    pub(super) visible: Vec<ItemIndex>,
    pub(super) next_visible: Vec<ItemIndex>,
    pub(super) region_water: HashMap<Region, ItemIndex>,
    pub(super) region_water_changed: bool,
    pub(super) mesh_order: BTreeSet<MeshDemandOrder>,
    pub(super) mesh_order_by_id: HashMap<MeshId, MeshDemandOrder>,
    pub(super) mesh_demand_revision: u64,
    pub(super) admission_protected: HashSet<MeshId>,
    pub(super) mesh_representatives: HashMap<MeshId, SharedMesh>,
    pub(super) mesh_compositions: HashMap<MeshId, Option<Arc<[f32]>>>,
    pub(super) retained_meshes: HashSet<MeshId>,
    pub(super) mesh_state: MeshState,
    pub(super) animated_items: HashSet<ItemIndex>,
    pub(super) patch_items: Vec<ItemIndex>,
    pub(super) patch_faces: Vec<(ItemIndex, usize)>,
    pub(super) changed_objects: HashSet<Entity>,
    pub(super) changed_surfaces: HashSet<(Region, SurfaceKind)>,
    pub(super) changed_regions: HashSet<Region>,
    pub(super) rebuild: Option<CacheRebuild>,
    pub(super) version: Option<ViewVersion>,
    pub(super) rescan_pending: bool,
    visibility_version: Option<VisibilityVersion>,
    filtered_visibility_version: Option<VisibilityVersion>,
    pub(super) camera: Option<Camera>,
    pub(super) origin: math::DVec3,
    pub(super) size: [u32; 2],
}

impl FrameCache {
    pub(super) fn clear_mesh_demands(&mut self) {
        let changed = !self.mesh_order.is_empty();
        self.mesh_order.clear();
        self.mesh_order_by_id.clear();
        self.admission_protected.clear();
        self.mesh_representatives.clear();
        self.mesh_compositions.clear();
        if changed {
            self.mesh_demand_revision = self.mesh_demand_revision.wrapping_add(1);
        }
    }

    pub(super) fn index(&self, id: ItemId) -> Option<ItemIndex> {
        self.item_indices.get(&id).copied()
    }

    pub(super) fn get_by_id_mut(&mut self, id: ItemId) -> Option<&mut CachedItem> {
        let index = self.index(id)?;
        self.items.get_mut(index)
    }

    pub(super) fn insert_item(&mut self, item: CachedItem) -> Option<ItemIndex> {
        let id = item.id;
        if let Some(index) = self.index(id) {
            if let Some(slot) = self.items.get_mut(index) {
                let water = (item.surface_kind == Some(SurfaceKind::RegionWater)).then_some(item.region);
                *slot = item;
                self.note_region_water(index, water);
                return Some(index);
            }
            self.item_indices.remove(&id);
        }
        let water = (item.surface_kind == Some(SurfaceKind::RegionWater)).then_some(item.region);
        let index = self.items.insert(item)?;
        self.item_indices.insert(id, index);
        self.item_order_dirty = true;
        self.note_region_water(index, water);
        Some(index)
    }

    fn note_region_water(&mut self, index: ItemIndex, region: Option<Region>) {
        if let Some(region) = region {
            self.region_water.insert(region, index);
            self.region_water_changed = true;
        }
    }

    pub(super) fn contains(&self, set: &[ItemIndex], index: ItemIndex) -> bool {
        self.items.get(index).is_some_and(|item| contains_id(set, &self.items, item.id))
    }

    pub(super) fn slot_count(&self) -> usize {
        self.items.slot_count()
    }

    pub(super) fn remove_item(&mut self, index: ItemIndex) -> Option<(CachedItem, bool)> {
        let id = self.items.get(index)?.id;
        let was_visible = contains_id(&self.visible, &self.items, id);
        update_membership(&mut self.range, &self.items, id, index, false);
        update_membership(&mut self.visible, &self.items, id, index, false);
        update_membership(&mut self.next_visible, &self.items, id, index, false);
        let item = self.items.remove(index)?;
        self.item_indices.remove(&item.id);
        self.replacements.remove(&index);
        if self.region_water.get(&item.region) == Some(&index) {
            self.region_water.remove(&item.region);
            self.region_water_changed = true;
        }
        self.item_order_dirty = true;
        Some((item, was_visible))
    }

    pub(super) fn promote_replacement(&mut self, index: ItemIndex) -> Option<CachedItem> {
        let item = self.items.get_mut(index)?;
        let replacement = item.replacement.take()?;
        let previous = mem::replace(item, replacement.item);
        self.replacements.remove(&index);
        Some(previous)
    }

    #[cfg(test)]
    pub(super) fn clear_items(&mut self) {
        self.items.clear();
        self.item_indices.clear();
        self.item_order.clear();
        self.item_order_dirty = true;
        self.replacements.clear();
        self.range.clear();
        self.range_camera = None;
        self.region_water.clear();
        self.region_water_changed = true;
        self.clear_mesh_demands();
        self.changed_objects.clear();
        self.changed_surfaces.clear();
        self.changed_regions.clear();
    }

    fn rebuild_item_order(&mut self) -> usize {
        if !self.item_order_dirty {
            return 0;
        }
        self.item_order.clear();
        self.item_order.extend(self.items.iter().map(|(index, _)| index));
        self.item_order.sort_unstable_by_key(|index| self.items.get(*index).map(|item| item.id));
        self.item_order_dirty = false;
        self.item_order.len()
    }

    fn advance_visibility(
        &mut self,
        view: &SceneView,
        view_sequence: u64,
        started: Instant,
        items: &mut usize,
        update: &mut CacheUpdate,
    ) -> bool {
        let Some(rebuild) = self.rebuild.as_mut() else { return false };
        match rebuild.phase {
            RebuildPhase::VisibilityObjects => {
                let after = rebuild.visibility_object_after;
                let page = view.object_page(after, REBUILD_ITEMS_PER_FRAME - *items);
                if page.entries.is_empty() && page.done {
                    rebuild.phase = RebuildPhase::VisibilitySurfaces;
                    return true;
                }
                let page_len = page.entries.len();
                let mut last = after;
                let mut processed = 0;
                for (id, object) in page.entries {
                    if *items >= REBUILD_ITEMS_PER_FRAME || started.elapsed() >= REBUILD_TIME_PER_FRAME {
                        break;
                    }
                    if let Some(index) = object.and_then(|object| self.objects.get(&object.detail().summary.entity)) {
                        update.visibility_items.push(*index);
                    }
                    last = Some(id);
                    *items += 1;
                    processed += 1;
                }
                rebuild.visibility_object_after = if processed == page_len { page.cursor } else { last };
                if processed == page_len && page.done {
                    rebuild.phase = RebuildPhase::VisibilitySurfaces;
                }
                *items < REBUILD_ITEMS_PER_FRAME && processed == page_len && page.done
            }
            RebuildPhase::VisibilitySurfaces => {
                let after = rebuild.visibility_surface_after;
                let (surfaces, _, done) = view.surface_page(after, REBUILD_ITEMS_PER_FRAME - *items);
                if surfaces.is_empty() && done {
                    rebuild.phase = RebuildPhase::Finalize;
                    return true;
                }
                let page_len = surfaces.len();
                let mut last = after;
                for surface in surfaces {
                    if let Some(index) = self.item_indices.get(&surface_item_id(surface.region, surface.kind)) {
                        update.visibility_items.push(*index);
                    }
                    last = Some((surface.region, surface.kind));
                    *items += 1;
                }
                rebuild.visibility_surface_after = last;
                if done {
                    rebuild.phase = RebuildPhase::Finalize;
                }
                *items < REBUILD_ITEMS_PER_FRAME && done && page_len > 0
            }
            RebuildPhase::Finalize => {
                if !rebuild.object_change_page_complete
                    || !rebuild.added_object_page_complete
                    || view_sequence > rebuild.object_change_page_sequence
                {
                    return false;
                }
                update.full_visibility = false;
                update.layout_changed |= rebuild.layout_changed;
                update.mesh_dirty |= rebuild.mesh_dirty;
                update.rebuild_visibility_changed |= rebuild.visibility_changed;
                update.mesh_demand_missing |= rebuild.mesh_demand_missing;
                self.rebuild = None;
                update.rebuild_pending = false;
                update.rebuild_completed = true;
                self.rescan_pending = false;
                self.version = Some(view.version());
                self.visibility_version = Some(view.visibility_version());
                false
            }
            RebuildPhase::Objects | RebuildPhase::Surfaces | RebuildPhase::Sweep | RebuildPhase::PruneMeshInfo => false,
        }
    }
}

impl GeometryRenderer {
    pub(super) fn note_rebuild_camera(&mut self, camera: Camera) -> Option<Camera> {
        self.cache.rebuild.as_mut().map(|rebuild| rebuild.note_camera(camera))
    }

    pub(super) fn update_cache_for_frame(&mut self, view: &SceneView, defer_recovery: bool) -> CacheUpdate {
        self.update_cache_inner(view, defer_recovery)
    }

    fn update_cache_inner(&mut self, view: &SceneView, defer_recovery: bool) -> CacheUpdate {
        if self.cache.rebuild.is_some() {
            let observed_version = self.cache.rebuild.as_ref().map(|rebuild| rebuild.observed_version.clone());
            let changes = observed_version.as_ref().and_then(|version| view.changes_since(version));
            let mut history_gap = changes.is_none();
            if let Some(changes) = changes {
                for change in changes {
                    history_gap |= change.full || change.reconcile_all || !change.retired.is_empty();
                    self.cache.changed_surfaces.extend(change.surfaces.iter().copied());
                    self.cache.changed_regions.extend(change.regions.iter().copied());
                }
            }
            let version = view.version();
            let visibility = view.visibility_version();
            if let Some(rebuild) = self.cache.rebuild.as_mut() {
                rebuild.observed_version = version;
                rebuild.surface_rescan_pending |= history_gap;
            }
            let mut update = CacheUpdate::default();
            let visibility_changed = self.cache.rebuild.as_ref().is_some_and(|rebuild| rebuild.observed_visibility != visibility);
            if visibility_changed {
                let slot_count = self.cache.slot_count();
                if let Some(rebuild) = self.cache.rebuild.as_mut() {
                    rebuild.observed_visibility = visibility;
                    rebuild.sweep_needed = true;
                    match rebuild.phase {
                        RebuildPhase::PruneMeshInfo
                        | RebuildPhase::VisibilityObjects
                        | RebuildPhase::VisibilitySurfaces
                        | RebuildPhase::Finalize => {
                            if !rebuild.visibility_restarted {
                                rebuild.phase = RebuildPhase::Sweep;
                                rebuild.sweep_cursor = 0;
                                rebuild.sweep_limit = slot_count;
                                rebuild.visibility_restart = true;
                                rebuild.visibility_restarted = true;
                            }
                        }
                        RebuildPhase::Sweep => rebuild.sweep_limit = rebuild.sweep_limit.max(slot_count),
                        RebuildPhase::Objects | RebuildPhase::Surfaces => {}
                    }
                }
                update.immediate_visibility_change = self.filter_inactive(view);
                self.cache.filtered_visibility_version = Some(visibility);
                update.layout_changed = false;
            }
            self.advance_rebuild(view, &mut update);
            return update;
        }

        if defer_recovery {
            if self.cache.version.as_ref() != Some(&view.version()) {
                self.cache.rescan_pending = true;
                self.cache.version = None;
            }
            if self.cache.visibility_version != Some(view.visibility_version()) {
                return self.begin_rebuild(view);
            }
            return self.defer_recovery(view);
        }

        if self.cache.rescan_pending {
            self.cache.rescan_pending = false;
            return self.begin_rebuild(view);
        }

        let version = self.cache.version.clone();
        let changes = version.as_ref().and_then(|version| view.changes_since(version));
        let mut full = changes.is_none();
        let mut reconcile_all = false;
        let mut retired = false;
        let mut regions = Vec::new();
        let mut motion_changed = HashSet::new();
        self.cache.changed_objects.clear();
        self.cache.changed_surfaces.clear();
        if let Some(changes) = changes {
            for change in changes {
                full |= change.full;
                reconcile_all |= change.reconcile_all;
                if full || reconcile_all {
                    break;
                }
                regions.extend(change.regions.iter().copied());
                motion_changed.extend(change.motion_changed.iter().copied());
                self.cache.changed_objects.extend(change.objects.iter().copied());
                self.cache.changed_surfaces.extend(change.surfaces.iter().copied());
                retired |= !change.retired.is_empty();
            }
        }
        if full || reconcile_all || retired {
            return self.begin_rebuild(view);
        }
        let mut update = CacheUpdate { motion_changed, ..CacheUpdate::default() };
        regions.sort_unstable();
        regions.dedup();
        for region in regions {
            let textures = view.region(region).map(|region| region.terrain_textures().clone()).unwrap_or_else(|| array::from_fn(|_| None));
            for surface in view.surfaces().filter(|surface| surface.region == region && matches!(surface.kind, SurfaceKind::Terrain(_))) {
                let id = surface_item_id(region, surface.kind);
                if let Some(item) = self.cache.get_by_id_mut(id)
                    && item.terrain_textures.as_ref() != Some(&textures)
                {
                    item.terrain_textures = Some(textures.clone());
                    self.draw_list.update_terrain(region, &textures);
                    update.items_updated += 1;
                    if !update.terrain_regions.contains(&region) {
                        update.terrain_regions.push(region);
                    }
                }
            }
        }
        while let Some(entity) = self.cache.changed_objects.iter().next().copied() {
            self.cache.changed_objects.remove(&entity);
            update.background.push(entity);
            self.update_object(view.object(entity), entity, &mut update);
        }
        while let Some((region, kind)) = self.cache.changed_surfaces.iter().next().copied() {
            self.cache.changed_surfaces.remove(&(region, kind));
            self.update_surface(view, view.surface(region, kind), region, kind, &mut update);
        }
        if update.full_visibility || update.mesh_dirty {
            self.prune_mesh_info();
        }
        self.cache.version = Some(view.version());
        self.cache.visibility_version = Some(view.visibility_version());
        self.cache.filtered_visibility_version = Some(view.visibility_version());
        update
    }

    fn begin_rebuild(&mut self, view: &SceneView) -> CacheUpdate {
        self.cache.changed_objects.clear();
        self.cache.changed_surfaces.clear();
        self.cache.changed_regions.clear();
        let visibility = view.visibility_version();
        let sweep_needed = self.cache.visibility_version != Some(visibility);
        self.cache.clear_mesh_demands();
        self.cache.rebuild = Some(CacheRebuild::new(view, self.cache.slot_count(), sweep_needed));
        let mut update = CacheUpdate { rebuild_pending: true, rebuild_started: true, ..CacheUpdate::default() };
        if sweep_needed {
            update.immediate_visibility_change = self.filter_inactive(view);
            self.cache.filtered_visibility_version = Some(visibility);
        }
        update.layout_changed = false;
        self.advance_rebuild(view, &mut update);
        update
    }

    fn defer_recovery(&mut self, view: &SceneView) -> CacheUpdate {
        let visibility = view.visibility_version();
        if self.cache.filtered_visibility_version == Some(visibility) {
            return CacheUpdate::default();
        }

        let update = CacheUpdate {
            immediate_visibility_change: self.filter_inactive(view),
            full_visibility: true,
            layout_changed: true,
            ..CacheUpdate::default()
        };
        self.cache.filtered_visibility_version = Some(visibility);
        if self.draw_list.layout_pending() {
            self.draw_list.cancel_layout();
            self.layout_counts.cancelled += 1;
            self.layout_required = true;
            self.layout_pose_items.clear();
        }
        update
    }

    fn advance_rebuild(&mut self, view: &SceneView, update: &mut CacheUpdate) {
        let started = Instant::now();
        let mut items = 0;
        let mut bytes = 0;
        update.rebuild_pending = true;
        let view_sequence = view.sequence();
        if let Some(rebuild) = self.cache.rebuild.as_mut()
            && rebuild.object_change_page_complete
            && rebuild.added_object_page_complete
            && view_sequence > rebuild.object_change_page_sequence
        {
            rebuild.object_change_since = rebuild.object_change_page_sequence;
            rebuild.object_change_page_sequence = view_sequence;
            rebuild.object_change_after = None;
            rebuild.added_object_after = None;
            rebuild.object_change_page_complete = false;
            rebuild.added_object_page_complete = false;
        }
        let (object_change_since, page_after) =
            self.cache.rebuild.as_ref().map(|rebuild| (rebuild.object_change_since, rebuild.object_change_after)).unwrap_or_default();
        let (object_changes, object_change_page_cursor, object_change_page_done) =
            view.changed_objects_page(object_change_since, page_after, 128);
        let added_after = self.cache.rebuild.as_ref().and_then(|rebuild| rebuild.added_object_after);
        let (added_objects, added_object_page_cursor, added_object_page_done) =
            view.added_objects_page(object_change_since, added_after, 128);
        let mut object_change_index = 0;
        let mut added_object_index = 0;
        let mut object_change_page_advanced = false;
        let mut added_object_page_advanced = false;
        loop {
            if items >= REBUILD_ITEMS_PER_FRAME || bytes >= REBUILD_BYTES_PER_FRAME || started.elapsed() >= REBUILD_TIME_PER_FRAME {
                break;
            }
            let added_object = added_objects.get(added_object_index).copied();
            let changed_object = object_changes.get(object_change_index).copied();
            let changed_object = added_object.map(|(sequence, id, entity)| (sequence, id, entity, true)).or(changed_object);
            let next_object_change = changed_object.map(|(sequence, id, _, _)| (sequence, id));
            let next_added_object = added_object.map(|(sequence, id, _)| (sequence, id));
            let current_object_change_after = self.cache.rebuild.as_ref().map(|rebuild| rebuild.object_change_after).unwrap_or_default();
            let current_added_object_after = self.cache.rebuild.as_ref().and_then(|rebuild| rebuild.added_object_after);
            if added_object.is_none() && !added_object_page_advanced {
                if added_object_page_cursor != current_added_object_after
                    && let Some(rebuild) = self.cache.rebuild.as_mut()
                {
                    rebuild.added_object_after = added_object_page_cursor;
                    items += 1;
                    bytes += size_of::<Uuid>() + size_of::<u64>();
                    added_object_page_advanced = true;
                    continue;
                }
                if let Some(rebuild) = self.cache.rebuild.as_mut() {
                    rebuild.added_object_page_complete = added_object_page_done;
                }
                added_object_page_advanced = true;
            }
            if changed_object.is_none()
                && !object_change_page_advanced
                && object_change_page_cursor != current_object_change_after
                && let Some(rebuild) = self.cache.rebuild.as_mut()
            {
                rebuild.object_change_after = object_change_page_cursor;
                items += 1;
                bytes += size_of::<Entity>();
                object_change_page_advanced = true;
                continue;
            }
            if changed_object.is_none()
                && !object_change_page_advanced
                && object_change_page_cursor == current_object_change_after
                && let Some(rebuild) = self.cache.rebuild.as_mut()
            {
                rebuild.object_change_page_complete = object_change_page_done;
                object_change_page_advanced = true;
            }
            if let Some((_, _, entity, _)) = changed_object.filter(|_| started.elapsed() < REBUILD_CHANGE_TIME_PER_FRAME) {
                let object = view.object(entity);
                let rebuild_bytes = if let Some(object) = object {
                    let mesh = object
                        .mesh()
                        .filter(|_| object.mesh_matches_geometry() && matches!(object.placement(), Placement::Ready(_)))
                        .map(|(_, mesh)| mesh);
                    if let Some(mesh) = mesh {
                        let allowance = REBUILD_BYTES_PER_FRAME.saturating_sub(bytes);
                        let face_allowance = REBUILD_ITEMS_PER_FRAME.saturating_sub(items).saturating_sub(1);
                        let time_allowance = REBUILD_TIME_PER_FRAME.saturating_sub(started.elapsed());
                        let (used, faces, complete) = self.advance_rebuild_mesh_info(mesh, allowance, face_allowance, time_allowance);
                        self.stats.rebuild_mesh_info_bytes = self.stats.rebuild_mesh_info_bytes.saturating_add(used);
                        bytes = bytes.saturating_add(used);
                        items += faces;
                        if !complete {
                            break;
                        }
                        object_rebuild_bytes(object, &self.cache.mesh_info)
                    } else {
                        self.clear_rebuild_mesh_info();
                        1
                    }
                } else {
                    self.clear_rebuild_mesh_info();
                    1
                };
                if bytes.saturating_add(rebuild_bytes) > REBUILD_BYTES_PER_FRAME {
                    break;
                }
                if let Some(rebuild) = self.cache.rebuild.as_mut() {
                    if added_object.is_some() {
                        rebuild.added_object_after = next_added_object;
                        added_object_index += 1;
                    } else {
                        rebuild.object_change_after = next_object_change;
                        object_change_index += 1;
                    }
                }
                if let Some(object) = object {
                    update.background.push(entity);
                    self.update_object(Some(object), entity, update);
                    if let Some(index) = self.cache.objects.get(&entity).copied() {
                        update.visibility_items.push(index);
                        self.retain_rebuild_meshes(index);
                    }
                    self.stats.rebuild_objects += 1;
                } else {
                    self.update_object(None, entity, update);
                }
                items += 1;
                bytes = bytes.saturating_add(rebuild_bytes);
                continue;
            }
            if let Some((region, kind)) = self.cache.changed_surfaces.iter().next().copied() {
                let surface = view.surface(region, kind);
                let rebuild_bytes = if let Some(surface) = surface {
                    let mesh = match &surface.mesh {
                        PreparedMesh::Ready(mesh) => Some(mesh),
                        _ => None,
                    };
                    if let Some(mesh) = mesh {
                        let allowance = REBUILD_BYTES_PER_FRAME.saturating_sub(bytes);
                        let face_allowance = REBUILD_ITEMS_PER_FRAME.saturating_sub(items).saturating_sub(1);
                        let time_allowance = REBUILD_TIME_PER_FRAME.saturating_sub(started.elapsed());
                        let (used, faces, complete) = self.advance_rebuild_mesh_info(mesh, allowance, face_allowance, time_allowance);
                        self.stats.rebuild_mesh_info_bytes = self.stats.rebuild_mesh_info_bytes.saturating_add(used);
                        bytes = bytes.saturating_add(used);
                        items += faces;
                        if !complete {
                            break;
                        }
                        surface_rebuild_bytes(surface, &self.cache.mesh_info)
                    } else {
                        self.clear_rebuild_mesh_info();
                        1
                    }
                } else {
                    self.clear_rebuild_mesh_info();
                    1
                };
                if bytes.saturating_add(rebuild_bytes) > REBUILD_BYTES_PER_FRAME {
                    break;
                }
                self.cache.changed_surfaces.remove(&(region, kind));
                self.update_surface(view, surface, region, kind, update);
                if let Some(index) = self.cache.index(surface_item_id(region, kind)) {
                    update.visibility_items.push(index);
                    self.retain_rebuild_meshes(index);
                }
                self.stats.rebuild_surfaces += 1;
                items += 1;
                bytes = bytes.saturating_add(rebuild_bytes);
                continue;
            }
            if let Some(region) = self.cache.changed_regions.iter().next().copied() {
                self.cache.changed_regions.remove(&region);
                let textures =
                    view.region(region).map(|region| region.terrain_textures().clone()).unwrap_or_else(|| array::from_fn(|_| None));
                for surface in view.surfaces().filter(|surface| surface.region == region && matches!(surface.kind, SurfaceKind::Terrain(_)))
                {
                    let id = surface_item_id(region, surface.kind);
                    if let Some(item) = self.cache.get_by_id_mut(id)
                        && item.terrain_textures.as_ref() != Some(&textures)
                    {
                        item.terrain_textures = Some(textures.clone());
                        self.draw_list.update_terrain(region, &textures);
                        update.items_updated += 1;
                        if !update.terrain_regions.contains(&region) {
                            update.terrain_regions.push(region);
                        }
                    }
                }
                items += 1;
                bytes += size_of::<[Option<assets::TextureHandle>; 4]>();
                continue;
            }
            let phase = self.cache.rebuild.as_ref().map(|rebuild| rebuild.phase);
            match phase {
                Some(RebuildPhase::Objects) => {
                    let after = self.cache.rebuild.as_ref().and_then(|rebuild| rebuild.object_after);
                    let remaining = REBUILD_ITEMS_PER_FRAME - items;
                    let page = view.object_page(after, remaining);
                    if page.entries.is_empty() && page.done {
                        let slot_count = self.cache.slot_count();
                        if let Some(rebuild) = self.cache.rebuild.as_mut() {
                            rebuild.finish_objects(slot_count);
                        }
                        continue;
                    }
                    let page_len = page.entries.len();
                    let next = page.cursor;
                    let done = page.done;
                    let mut processed = 0;
                    let mut last = after;
                    for (id, object) in page.entries {
                        if items >= REBUILD_ITEMS_PER_FRAME || started.elapsed() >= REBUILD_TIME_PER_FRAME {
                            break;
                        }
                        if let Some(object) = object {
                            let entity = object.detail().summary.entity;
                            let mesh = object
                                .mesh()
                                .filter(|_| object.mesh_matches_geometry() && matches!(object.placement(), Placement::Ready(_)))
                                .map(|(_, mesh)| mesh);
                            let rebuild_bytes = if let Some(mesh) = mesh {
                                let allowance = REBUILD_BYTES_PER_FRAME.saturating_sub(bytes);
                                let face_allowance = REBUILD_ITEMS_PER_FRAME.saturating_sub(items).saturating_sub(1);
                                let time_allowance = REBUILD_TIME_PER_FRAME.saturating_sub(started.elapsed());
                                let (used, faces, complete) =
                                    self.advance_rebuild_mesh_info(mesh, allowance, face_allowance, time_allowance);
                                self.stats.rebuild_mesh_info_bytes = self.stats.rebuild_mesh_info_bytes.saturating_add(used);
                                bytes = bytes.saturating_add(used);
                                items += faces;
                                if !complete {
                                    break;
                                }
                                object_rebuild_bytes(object, &self.cache.mesh_info)
                            } else {
                                self.clear_rebuild_mesh_info();
                                1
                            };
                            if bytes.saturating_add(rebuild_bytes) > REBUILD_BYTES_PER_FRAME {
                                break;
                            }
                            self.update_object(Some(object), entity, update);
                            update.background.push(entity);
                            if let Some(index) = self.cache.objects.get(&entity).copied() {
                                update.visibility_items.push(index);
                                self.retain_rebuild_meshes(index);
                            }
                            self.stats.rebuild_objects += 1;
                            #[cfg(test)]
                            {
                                self.cache.rebuild.as_mut().expect("object rebuild exists").scanned_objects += 1;
                            }
                            bytes = bytes.saturating_add(rebuild_bytes);
                        } else {
                            self.clear_rebuild_mesh_info();
                        }
                        items += 1;
                        last = Some(id);
                        processed += 1;
                    }
                    let slot_count = self.cache.slot_count();
                    if let Some(rebuild) = self.cache.rebuild.as_mut() {
                        rebuild.object_after = if processed == page_len { next } else { last };
                        if processed == page_len && done {
                            rebuild.finish_objects(slot_count);
                        }
                    }
                    if processed == 0 || processed < page_len || !done {
                        break;
                    }
                }
                Some(RebuildPhase::Surfaces) => {
                    let after = self.cache.rebuild.as_ref().and_then(|rebuild| rebuild.surface_after);
                    let remaining = REBUILD_ITEMS_PER_FRAME - items;
                    let (surfaces, _, done) = view.surface_page(after, remaining);
                    if surfaces.is_empty() && done {
                        let slot_count = self.cache.slot_count();
                        if let Some(rebuild) = self.cache.rebuild.as_mut() {
                            rebuild.finish_surfaces(slot_count);
                        }
                        continue;
                    }
                    let page_len = surfaces.len();
                    let mut processed = 0;
                    let mut last = after;
                    for surface in surfaces {
                        if started.elapsed() >= REBUILD_TIME_PER_FRAME {
                            break;
                        }
                        let mesh = match &surface.mesh {
                            PreparedMesh::Ready(mesh) => Some(mesh),
                            _ => None,
                        };
                        let cost = if let Some(mesh) = mesh {
                            let allowance = REBUILD_BYTES_PER_FRAME.saturating_sub(bytes);
                            let face_allowance = remaining.saturating_sub(1);
                            let time_allowance = REBUILD_TIME_PER_FRAME.saturating_sub(started.elapsed());
                            let (used, faces, complete) = self.advance_rebuild_mesh_info(mesh, allowance, face_allowance, time_allowance);
                            self.stats.rebuild_mesh_info_bytes = self.stats.rebuild_mesh_info_bytes.saturating_add(used);
                            bytes = bytes.saturating_add(used);
                            items += faces;
                            if !complete {
                                break;
                            }
                            surface_rebuild_bytes(surface, &self.cache.mesh_info)
                        } else {
                            self.clear_rebuild_mesh_info();
                            1
                        };
                        if bytes.saturating_add(cost) > REBUILD_BYTES_PER_FRAME {
                            break;
                        }
                        last = Some((surface.region, surface.kind));
                        self.update_surface(view, Some(surface), surface.region, surface.kind, update);
                        self.stats.rebuild_surfaces += 1;
                        if let Some(index) = self.cache.index(surface_item_id(surface.region, surface.kind)) {
                            update.visibility_items.push(index);
                            self.retain_rebuild_meshes(index);
                        }
                        items += 1;
                        bytes = bytes.saturating_add(cost);
                        processed += 1;
                    }
                    let slot_count = self.cache.slot_count();
                    if let Some(rebuild) = self.cache.rebuild.as_mut() {
                        rebuild.surface_after = last;
                        if processed == page_len && done {
                            rebuild.finish_surfaces(slot_count);
                        }
                    }
                    if processed == 0 || processed < page_len || !done {
                        break;
                    }
                }
                Some(RebuildPhase::Sweep) => {
                    let Some(rebuild) = self.cache.rebuild.as_ref() else { break };
                    if rebuild.sweep_cursor >= rebuild.sweep_limit {
                        if let Some(rebuild) = self.cache.rebuild.as_mut() {
                            rebuild.phase = RebuildPhase::PruneMeshInfo;
                        }
                        continue;
                    }
                    let sweep_cursor = self.cache.rebuild.as_ref().map_or(0, |rebuild| rebuild.sweep_cursor);
                    self.stats.rebuild_sweep_slots += 1;
                    let item = self.cache.items.get_at(sweep_cursor).map(|(index, item)| {
                        (
                            index,
                            item.entity,
                            item.region,
                            item.surface_kind,
                            item.mesh.id(),
                            item.replacement.as_ref().map(|replacement| replacement.item.mesh.id()),
                        )
                    });
                    let cost = item.map_or(1, |_| size_of::<CachedFace>());
                    if bytes.saturating_add(cost) > REBUILD_BYTES_PER_FRAME {
                        break;
                    }
                    let mut cleanup_faces = 0;
                    let mut cleanup_pending = false;
                    if let Some((index, entity, region, surface_kind, mesh, replacement_mesh)) = item {
                        let live = self.cache.items.get(index).is_some_and(|item| item_is_live(view, item));
                        if !live {
                            loop {
                                let next_items = items.saturating_add(1).saturating_add(cleanup_faces);
                                let next_bytes =
                                    bytes.saturating_add(cost).saturating_add(cleanup_faces.saturating_mul(size_of::<CachedFace>()));
                                if next_items > REBUILD_ITEMS_PER_FRAME
                                    || next_bytes > REBUILD_BYTES_PER_FRAME
                                    || started.elapsed() >= REBUILD_TIME_PER_FRAME
                                {
                                    cleanup_pending = true;
                                    break;
                                }
                                if self.cleanup_stale_item(index) {
                                    self.stats.rebuild_cleanup_entries += 1;
                                    cleanup_faces += 1;
                                } else {
                                    if let Some(entity) = entity {
                                        self.update_object(None, entity, update);
                                    } else if let Some(kind) = surface_kind {
                                        self.update_surface(view, None, region, kind, update);
                                    }
                                    break;
                                }
                            }
                        } else if let Some(rebuild) = self.cache.rebuild.as_mut() {
                            rebuild.retained_meshes.insert(mesh);
                            if let Some(replacement_mesh) = replacement_mesh {
                                rebuild.retained_meshes.insert(replacement_mesh);
                            }
                        }
                    }
                    if cleanup_pending && cleanup_faces == 0 {
                        break;
                    }
                    let cleanup_extra = cleanup_faces.saturating_sub(1);
                    items = items.saturating_add(1 + cleanup_extra);
                    bytes = bytes.saturating_add(cost).saturating_add(cleanup_extra.saturating_mul(size_of::<CachedFace>()));
                    if cleanup_pending {
                        break;
                    }
                    if let Some(rebuild) = self.cache.rebuild.as_mut() {
                        rebuild.sweep_cursor += 1;
                    }
                }
                Some(RebuildPhase::PruneMeshInfo) => {
                    let Some(rebuild) = self.cache.rebuild.as_ref() else { break };
                    if let Some(info) = rebuild.mesh_info_cleanup.as_ref() {
                        if !info.centers.is_empty() {
                            let Some(rebuild) = self.cache.rebuild.as_mut() else { break };
                            if let Some(info) = rebuild.mesh_info_cleanup.as_mut() {
                                info.centers.pop();
                            }
                            self.stats.rebuild_mesh_info_centers_dropped += 1;
                            items += 1;
                            bytes += size_of::<Vec3>();
                            continue;
                        }
                        if let Some(rebuild) = self.cache.rebuild.as_mut() {
                            rebuild.mesh_info_cleanup = None;
                        }
                        self.stats.rebuild_mesh_info_entries += 1;
                        items += 1;
                        bytes += size_of::<MeshInfo>();
                        continue;
                    }
                    let Some(&id) = self.cache.mesh_info_order.get(rebuild.mesh_info_cursor) else {
                        let Some(rebuild) = self.cache.rebuild.as_mut() else { break };
                        rebuild.finish_mesh_info();
                        continue;
                    };
                    let Some(rebuild) = self.cache.rebuild.as_mut() else { break };
                    self.stats.rebuild_mesh_info_entries += 1;
                    if rebuild.retained_meshes.contains(&id) {
                        rebuild.mesh_info_cursor += 1;
                    } else {
                        let info = self.cache.mesh_info.remove(&id);
                        if let Some(info) = info {
                            match Arc::try_unwrap(info) {
                                Ok(info) => {
                                    rebuild.mesh_info_cleanup = Some(info);
                                    self.cache.mesh_info_order.swap_remove(rebuild.mesh_info_cursor);
                                }
                                Err(info) => {
                                    self.cache.mesh_info.insert(id, info);
                                    rebuild.mesh_info_cursor += 1;
                                }
                            }
                        } else {
                            self.cache.mesh_info_order.swap_remove(rebuild.mesh_info_cursor);
                        }
                    }
                    items += 1;
                    bytes += size_of::<MeshInfo>();
                }
                Some(RebuildPhase::VisibilityObjects | RebuildPhase::VisibilitySurfaces | RebuildPhase::Finalize) => {
                    if !self.cache.advance_visibility(view, view_sequence, started, &mut items, update) {
                        break;
                    }
                }
                None => {
                    break;
                }
            }
        }
        if self.cache.rebuild.is_some() {
            if let Some(rebuild) = self.cache.rebuild.as_mut() {
                rebuild.layout_changed |= update.layout_changed;
                rebuild.mesh_dirty |= update.mesh_dirty;
            }
            update.rebuild_pending = true;
            update.full_visibility = false;
            update.mesh_dirty = false;
            update.layout_changed = false;
        }
    }

    fn advance_rebuild_mesh_info(
        &mut self,
        mesh: &SharedMesh,
        byte_allowance: usize,
        face_allowance: usize,
        time_allowance: Duration,
    ) -> (usize, usize, bool) {
        let id = mesh.id();
        if self.cache.mesh_info.contains_key(&id) {
            self.clear_rebuild_mesh_info();
            return (0, 0, true);
        }
        let Some(rebuild) = self.cache.rebuild.as_mut() else { return (0, 0, false) };
        if rebuild.mesh_info_build.as_ref().is_none_or(|build| build.mesh.id() != id) {
            rebuild.mesh_info_build = Some(MeshInfoBuild::new(mesh.clone()));
        }
        let build = rebuild.mesh_info_build.as_mut().expect("mesh info build exists");
        let (bytes, faces) = build.advance(byte_allowance, face_allowance, time_allowance);
        if !build.is_complete() {
            return (bytes, faces, false);
        }
        let info = Arc::new(MeshInfo { bounds: build.mesh.mesh().bounds(), centers: mem::take(&mut build.centers) });
        rebuild.mesh_info_build = None;
        self.cache.mesh_info.insert(id, info);
        self.cache.mesh_info_order.push(id);
        (bytes, faces, true)
    }

    fn cleanup_stale_item(&mut self, index: ItemIndex) -> bool {
        let Some(item) = self.cache.items.get_mut(index) else { return false };
        if let Some(mut face) = item.faces.pop() {
            self.draw_list.hide_face(&mut face);
            self.draw_list.release_group(face.group);
            return true;
        }
        if let Some(replacement) = item.replacement.as_mut()
            && let Some(mut face) = replacement.item.faces.pop()
        {
            self.draw_list.hide_face(&mut face);
            self.draw_list.release_group(face.group);
            return true;
        }
        if item.animated_faces.pop().is_some() {
            return true;
        }
        if let Some(replacement) = item.replacement.as_mut()
            && replacement.item.animated_faces.pop().is_some()
        {
            return true;
        }
        false
    }

    fn clear_rebuild_mesh_info(&mut self) {
        if let Some(rebuild) = self.cache.rebuild.as_mut() {
            rebuild.mesh_info_build = None;
        }
    }

    fn retain_rebuild_meshes(&mut self, index: ItemIndex) {
        let Some(item) = self.cache.items.get(index) else { return };
        let mesh = item.mesh.id();
        let replacement = item.replacement.as_ref().map(|replacement| replacement.item.mesh.id());
        if let Some(rebuild) = self.cache.rebuild.as_mut() {
            rebuild.retained_meshes.insert(mesh);
            if let Some(replacement) = replacement {
                rebuild.retained_meshes.insert(replacement);
            }
        }
    }

    fn filter_inactive(&mut self, view: &SceneView) -> bool {
        let (visible, items) = (&mut self.cache.visible, &mut self.cache.items);
        let draw_list = &mut self.draw_list;
        let mut removed = false;
        let mut visible_items = 0;
        visible.retain(|index| {
            let Some(item) = items.get_mut(*index) else {
                removed = true;
                return false;
            };
            let live = item_is_live(view, item);
            if !live {
                draw_list.hide_item(*index, item);
            } else {
                visible_items += usize::from(item.faces.iter().any(|face| face.slot.is_some()));
            }
            removed |= !live;
            live
        });
        draw_list.set_visible_items(visible_items);
        removed
    }

    fn update_object(&mut self, object: Option<&Object>, entity: Entity, update: &mut CacheUpdate) {
        let mut old_index = self.cache.objects.get(&entity).copied();
        if let Some(object) = object
            && let Some(index) = self.cache.index(object_item_id(object))
            && Some(index) != old_index
        {
            let previous_entity = self.cache.items.get(index).and_then(|item| item.entity);
            if previous_entity == Some(entity) {
                self.cache.objects.insert(entity, index);
                old_index = Some(index);
            } else if let Some(previous_entity) = previous_entity {
                self.remove_item(index, update, true);
                self.cache.animated_items.remove(&index);
                self.cache.objects.remove(&previous_entity);
                update.full_visibility = true;
                update.layout_changed = true;
                update.mesh_dirty = true;
            }
        }
        let old_id = old_index.and_then(|index| self.cache.items.get(index).map(|item| item.id));
        let Some(object) = object else {
            if let Some(index) = old_index {
                update.items_updated += 1;
                self.remove_item(index, update, true);
                self.cache.animated_items.remove(&index);
                self.cache.objects.remove(&entity);
                update.full_visibility = true;
                update.layout_changed = true;
                update.mesh_dirty = true;
            }
            return;
        };
        let Some((_, mesh)) = object.mesh() else {
            if let Some(index) = old_index {
                update.items_updated += 1;
                self.remove_item(index, update, true);
                self.cache.animated_items.remove(&index);
                self.cache.objects.remove(&entity);
                update.full_visibility = true;
                update.layout_changed = true;
                update.mesh_dirty = true;
            }
            return;
        };
        let Placement::Ready(transform) = object.placement() else {
            if let Some(index) = old_index {
                update.items_updated += 1;
                self.remove_item(index, update, true);
                self.cache.animated_items.remove(&index);
                self.cache.objects.remove(&entity);
                update.full_visibility = true;
                update.layout_changed = true;
                update.mesh_dirty = true;
            }
            return;
        };
        if !object.mesh_matches_geometry() {
            if let Some(index) = old_index {
                self.discard_replacement(index);
                self.update_held_object_pose(index, transform, update);
            }
            return;
        }
        let previous_mesh = old_index.and_then(|index| self.cache.items.get(index).map(|cached| cached.mesh.id()));
        let unchanged = old_index.is_some_and(|index| {
            self.cache.items.get(index).is_some_and(|cached| {
                cached.id == object_item_id(object)
                    && cached.mesh.id() == mesh.id()
                    && cached.source.as_ref().is_some_and(|previous| previous.same_render_appearance(object))
            })
        });
        if unchanged && let Some(index) = old_index {
            self.discard_replacement(index);
            let Some(cached) = self.cache.items.get_mut(index) else { return };
            let flags = object.mesh_flags().unwrap_or_default();
            let scale = object_draw_scale(object, transform.scale);
            let pose_changed =
                cached.position != transform.pose.position || cached.rotation != transform.pose.rotation || cached.scale != scale;
            let parameters_dirty =
                cached.source.as_ref().and_then(|previous| previous.detail().texture_animation) != object.detail().texture_animation;
            update.items_updated += usize::from(pose_changed || parameters_dirty);
            if parameters_dirty {
                cached.animated_faces = animated_face_indices(object, &cached.faces);
                cached.animation_deadline = (!cached.animated_faces.is_empty()).then(Instant::now);
            }
            let scale_changed = cached.scale != scale;
            let winding_changed =
                self.draw_list.refresh_winding(&mut cached.faces, cached.scale, cached.invert_normals, scale, flags.invert);
            if pose_changed {
                cached.transform = instance_base(transform.pose.position, transform.pose.rotation, scale, flags.invert, self.cache.origin);
                if scale_changed {
                    for face in &mut cached.faces {
                        face.parameter_bytes = parameter_part(&face.params, scale);
                    }
                }
            }
            cached.position = transform.pose.position;
            cached.rotation = transform.pose.rotation;
            cached.scale = scale;
            cached.invert_normals = flags.invert;
            update.layout_changed |= winding_changed;
            cached.parameters_dirty |= parameters_dirty;
            cached.source = Some(object.clone());
            if !cached.animated_faces.is_empty() {
                self.cache.animated_items.insert(index);
            } else {
                self.cache.animated_items.remove(&index);
            }
            if pose_changed {
                update.pose_items.push(index);
            }
            if parameters_dirty {
                update.parameter_items.push(index);
            }
            return;
        }
        let Some(mut item) = self.object_item(object) else {
            if let Some(index) = old_index {
                self.remove_item(index, update, true);
                self.cache.animated_items.remove(&index);
                update.mesh_dirty = true;
            }
            self.cache.objects.remove(&entity);
            update.full_visibility = true;
            update.layout_changed = true;
            return;
        };
        let previous_asset = old_index
            .and_then(|index| self.cache.items.get(index))
            .and_then(|cached| cached.source.as_ref())
            .and_then(Object::mesh)
            .and_then(|(key, _)| match key {
                DrawnMeshKey::Asset { asset, .. } => Some(asset),
                DrawnMeshKey::Prim(_) | DrawnMeshKey::Sculpt(_) | DrawnMeshKey::SculptPlaceholder => None,
            });
        let current_asset = object.mesh().and_then(|(key, _)| match key {
            DrawnMeshKey::Asset { asset, .. } => Some(asset),
            DrawnMeshKey::Prim(_) | DrawnMeshKey::Sculpt(_) | DrawnMeshKey::SculptPlaceholder => None,
        });
        let asset_replaced = current_asset.is_some() && current_asset != previous_asset;
        if let Some(index) = old_index
            && (asset_replaced
                || previous_mesh != Some(item.mesh.id()) && !self.meshes.contains_key(&item.mesh.id())
                || self.draw_list.layout_pending())
        {
            self.discard_replacement(index);
            if let Some(cached) = self.cache.items.get_mut(index) {
                cached.replacement = Some(Box::new(CachedReplacement { item }));
                self.cache.replacements.insert(index);
            }
            self.update_held_object_pose(index, transform, update);
            update.items_updated += 1;
            update.mesh_dirty = true;
            update.full_visibility = true;
            return;
        }
        update.items_updated += 1;
        if let Some(index) = old_index {
            update.layout_invalidated = true;
            if let Some(previous) = self.remove_item(index, update, false)
                && self.cache.rebuild.is_some()
            {
                retain_draw_slots(&previous, &mut item);
            }
            self.cache.animated_items.remove(&index);
        }
        let id = item.id;
        update.mesh_dirty |= previous_mesh != Some(item.mesh.id());
        if let Some(index) = self.cache.insert_item(item) {
            if self.cache.items.get(index).is_some_and(|item| !item.animated_faces.is_empty()) {
                self.cache.animated_items.insert(index);
            }
            self.cache.objects.insert(entity, index);
            if old_id == Some(id) {
                update.pose_items.push(index);
            }
        }
        update.layout_changed = true;
        if old_id != Some(id) {
            update.full_visibility = true;
        }
    }

    fn discard_replacement(&mut self, index: ItemIndex) {
        let replacement = self.cache.items.get_mut(index).and_then(|item| item.replacement.take());
        if let Some(replacement) = replacement {
            self.cache.replacements.remove(&index);
            self.draw_list.release_faces(&replacement.item.faces);
        }
    }

    fn update_held_object_pose(&mut self, index: ItemIndex, transform: world::WorldTransform, update: &mut CacheUpdate) {
        let Some(cached) = self.cache.items.get_mut(index) else { return };
        let mirror = cached.source.as_ref().and_then(Object::mesh_flags).is_some_and(|flags| flags.mirror);
        let scale = draw_scale(transform.scale, mirror);
        let pose_changed =
            cached.position != transform.pose.position || cached.rotation != transform.pose.rotation || cached.scale != scale;
        let scale_changed = cached.scale != scale;
        let winding_changed =
            self.draw_list.refresh_winding(&mut cached.faces, cached.scale, cached.invert_normals, scale, cached.invert_normals);
        if pose_changed {
            cached.transform =
                instance_base(transform.pose.position, transform.pose.rotation, scale, cached.invert_normals, self.cache.origin);
            if scale_changed {
                for face in &mut cached.faces {
                    face.parameter_bytes = parameter_part(&face.params, scale);
                }
            }
        }
        cached.position = transform.pose.position;
        cached.rotation = transform.pose.rotation;
        cached.scale = scale;
        cached.parameters_dirty |= scale_changed;
        update.items_updated += usize::from(pose_changed);
        update.layout_changed |= winding_changed;
        if pose_changed {
            update.pose_items.push(index);
        }
    }

    pub(super) fn promote_replacements(&mut self, indices: &[ItemIndex], update: &mut CacheUpdate) -> bool {
        let mut changed = false;
        for index in indices.iter().copied() {
            let Some(previous) = self.cache.promote_replacement(index) else { continue };
            self.draw_list.release_faces(&previous.faces);
            self.cache.animated_items.remove(&index);
            if self.cache.items.get(index).is_some_and(|item| !item.animated_faces.is_empty()) {
                self.cache.animated_items.insert(index);
            }
            update.items_updated += 1;
            update.full_visibility = true;
            update.layout_changed = true;
            update.mesh_dirty = true;
            update.layout_invalidated = true;
            changed = true;
        }
        if changed {
            self.prune_mesh_info();
        }
        changed
    }
    fn update_surface(&mut self, view: &SceneView, surface: Option<&Surface>, region: Region, kind: SurfaceKind, update: &mut CacheUpdate) {
        let id = surface_item_id(region, kind);
        let old_index = self.cache.index(id);
        if let Some(surface) = surface
            && let Some(item) = self.surface_item(view, surface)
        {
            if let Some(previous) = old_index.and_then(|index| self.cache.items.get_mut(index))
                && same_surface_geometry(previous, &item)
            {
                if previous.terrain_textures != item.terrain_textures {
                    previous.terrain_textures = item.terrain_textures;
                    if let Some(textures) = previous.terrain_textures.as_ref() {
                        self.draw_list.update_terrain(region, textures);
                    }
                    update.items_updated += 1;
                    if !update.terrain_regions.contains(&region) {
                        update.terrain_regions.push(region);
                    }
                }
                self.draw_list.release_faces(&item.faces);
                return;
            }
            update.items_updated += 1;
            update.mesh_dirty |=
                old_index.is_none_or(|index| self.cache.items.get(index).is_none_or(|previous| previous.mesh.id() != item.mesh.id()));
            if let Some(index) = old_index {
                update.layout_invalidated = true;
                self.remove_item(index, update, false);
            }
            self.cache.insert_item(item);
            update.layout_changed = true;
            update.full_visibility = true;
            update.mesh_dirty = true;
        } else if let Some(index) = old_index {
            self.remove_item(index, update, true);
            update.items_updated += 1;
            update.full_visibility = true;
            update.layout_changed = true;
            update.mesh_dirty = true;
        }
    }

    fn object_item(&mut self, object: &Object) -> Option<CachedItem> {
        if !object.mesh_matches_geometry() {
            return None;
        }
        let (_, mesh) = object.mesh()?;
        let Placement::Ready(transform) = object.placement() else { return None };
        let flags = object.mesh_flags().unwrap_or_default();
        let scale = object_draw_scale(object, transform.scale);
        let id = object_item_id(object);
        let region = object.detail().summary.address.region();
        let info = self.cached_mesh_info(mesh);
        let materials = object.materials();
        let appearance = object.detail().surface_appearance.as_ref();
        let faces = cached_faces(
            CachedFaceContext {
                id,
                mesh,
                info: &info,
                materials,
                appearance,
                scale,
                invert_normals: flags.invert,
                shading: ShadingMode::Surface,
                terrain: None,
                terrain_textures: None,
            },
            &mut self.draw_list,
        );
        let animated_faces = animated_face_indices(object, &faces);
        Some(CachedItem {
            id,
            entity: Some(object.detail().summary.entity),
            region,
            mesh: mesh.clone(),
            info,
            position: transform.pose.position,
            rotation: transform.pose.rotation,
            scale,
            invert_normals: flags.invert,
            replacement: None,
            transform: instance_base(transform.pose.position, transform.pose.rotation, scale, flags.invert, self.cache.origin),
            pickable: true,
            shading: ShadingMode::Surface,
            terrain_textures: None,
            terrain_composition: None,
            surface_kind: None,
            source: Some(object.clone()),
            faces,
            animation_deadline: (!animated_faces.is_empty()).then(Instant::now),
            animated_faces,
            parameters_dirty: false,
        })
    }

    fn surface_item(&mut self, view: &SceneView, surface: &Surface) -> Option<CachedItem> {
        let PreparedMesh::Ready(mesh) = &surface.mesh else { return None };
        let id = surface_item_id(surface.region, surface.kind);
        let (scale, rotation, pickable, shading) = match surface.kind {
            SurfaceKind::RegionWater => (Vec3::ONE, Quat::IDENTITY, false, ShadingMode::RegionWater),
            SurfaceKind::Terrain(_) => (Vec3::ONE, Quat::IDENTITY, true, ShadingMode::Terrain),
        };
        let terrain_textures = (shading == ShadingMode::Terrain)
            .then(|| view.region(surface.region).map_or_else(|| array::from_fn(|_| None), |region| region.terrain_textures().clone()));
        let info = self.cached_mesh_info(mesh);
        let terrain_region = (shading == ShadingMode::Terrain).then_some(surface.region);
        let faces = cached_faces(
            CachedFaceContext {
                id,
                mesh,
                info: &info,
                materials: None,
                appearance: None,
                scale,
                invert_normals: false,
                shading,
                terrain: terrain_region,
                terrain_textures: terrain_textures.as_ref(),
            },
            &mut self.draw_list,
        );
        Some(CachedItem {
            id,
            entity: None,
            region: surface.region,
            mesh: mesh.clone(),
            info,
            position: surface.position,
            rotation,
            scale,
            invert_normals: false,
            replacement: None,
            transform: instance_base(surface.position, rotation, scale, false, self.cache.origin),
            pickable,
            shading,
            terrain_textures,
            terrain_composition: surface.terrain_composition.clone(),
            surface_kind: Some(surface.kind),
            source: None,
            faces,
            animated_faces: Vec::new(),
            animation_deadline: None,
            parameters_dirty: false,
        })
    }

    fn cached_mesh_info(&mut self, mesh: &SharedMesh) -> Arc<MeshInfo> {
        let id = mesh.id();
        if let Some(info) = self.cache.mesh_info.get(&id) {
            return info.clone();
        }
        let info = mesh_info(mesh);
        self.cache.mesh_info.insert(id, info.clone());
        self.cache.mesh_info_order.push(id);
        info
    }

    pub(super) fn prune_mesh_info(&mut self) {
        self.cache.retained_meshes.clear();
        for (_, item) in self.cache.items.iter() {
            self.cache.retained_meshes.insert(item.mesh.id());
            if let Some(replacement) = item.replacement.as_ref() {
                self.cache.retained_meshes.insert(replacement.item.mesh.id());
            }
        }
        self.cache.mesh_info.retain(|id, _| self.cache.retained_meshes.contains(id));
        self.cache.mesh_info_order.retain(|id| self.cache.mesh_info.contains_key(id));
    }

    fn remove_item(&mut self, index: ItemIndex, update: &mut CacheUpdate, immediate_visibility: bool) -> Option<CachedItem> {
        let (mut item, was_visible) = self.cache.remove_item(index)?;
        self.draw_list.forget_item(index);
        if was_visible && immediate_visibility {
            self.draw_list.hide_item(index, &mut item);
        }
        update.immediate_visibility_change |= was_visible && immediate_visibility;
        update.layout_invalidated |= was_visible && immediate_visibility;
        self.cache.replacements.remove(&index);
        self.draw_list.release_faces(&item.faces);
        if let Some(replacement) = item.replacement.as_ref() {
            self.draw_list.release_faces(&replacement.item.faces);
        }
        Some(item)
    }

    pub(super) fn rebuild_range(&mut self, view: &SceneView, camera: Camera) -> bool {
        self.stats.item_order_items += self.cache.rebuild_item_order();
        self.cache.range_camera = Some(camera);
        self.cache.next_range.clear();
        for index in self.cache.item_order.iter().copied() {
            self.stats.candidate_items_visited += 1;
            let Some(item) = self.cache.items.get(index) else { continue };
            if item_is_live(view, item) && in_range(item, camera) {
                self.cache.next_range.push(index);
            }
        }
        let changed = self.cache.range != self.cache.next_range;
        if changed {
            mem::swap(&mut self.cache.range, &mut self.cache.next_range);
        }
        self.cache.next_range.clear();
        changed
    }

    pub(super) fn update_range(&mut self, view: &SceneView, ids: &[ItemIndex], demands: Option<(Camera, &Frustum)>) -> (bool, bool) {
        let Some(anchor) = self.cache.range_camera else { return (false, false) };
        let mut changed = false;
        let mut missing = false;
        for index in ids.iter().copied() {
            self.stats.candidate_items_visited += 1;
            let Some(item) = self.cache.items.get(index) else { continue };
            let id = item.id;
            let member = item_is_live(view, item) && in_range(item, anchor);
            missing |= member && !self.meshes.contains_key(&item.mesh.id());
            changed |= update_membership(&mut self.cache.range, &self.cache.items, id, index, member);
            if member && let Some((camera, frustum)) = demands {
                missing |= self.add_item_demands(view, index, camera, frustum);
            }
        }
        (changed, missing)
    }

    pub(super) fn rebuild_visible_meshes(&mut self, view: &SceneView, camera: Camera, frustum: &Frustum) -> bool {
        self.cache.clear_mesh_demands();
        self.cache.nonresident_demands = false;
        let mut missing = false;
        for position in 0..self.cache.range.len() {
            let Some(index) = self.cache.range.get(position).copied() else { break };
            missing |= self.add_item_demands(view, index, camera, frustum);
        }
        missing
    }

    fn add_item_demands(&mut self, view: &SceneView, index: ItemIndex, camera: Camera, frustum: &Frustum) -> bool {
        let Some(item) = self.cache.items.get(index).filter(|item| item_is_live(view, item)) else { return false };
        let demands = [Some(item), item.replacement.as_ref().map(|replacement| &replacement.item)].map(|candidate| {
            candidate.map(|candidate| {
                let resident = self.meshes.contains_key(&candidate.mesh.id());
                let in_view = WorldSphere::new(candidate.position, candidate.rotation, candidate.scale, candidate.info.bounds, camera)
                    .is_some_and(|sphere| sphere.visible(candidate.shading, camera, frustum));
                (candidate.mesh.clone(), candidate.terrain_composition.clone(), draw_priority(candidate, camera), resident, in_view)
            })
        });
        let mut missing = false;
        for (mesh, composition, priority, resident, in_view) in demands.into_iter().flatten() {
            self.stats.mesh_demand_items_visited += 1;
            missing |= add_mesh_demand(&mut self.cache, &mut self.stats, &mesh, composition, priority, resident, in_view);
        }
        self.cache.nonresident_demands |= missing;
        missing
    }
}

fn add_mesh_demand(
    cache: &mut FrameCache,
    stats: &mut super::Stats,
    mesh: &SharedMesh,
    composition: Option<Arc<[f32]>>,
    priority: (f64, f64),
    resident: bool,
    in_view: bool,
) -> bool {
    let mesh_id = mesh.id();
    let Some(current) = cache.mesh_order_by_id.get(&mesh_id).copied() else {
        let order = MeshDemandOrder { mesh: mesh_id, resident, in_view, priority };
        cache.mesh_order.insert(order);
        cache.mesh_order_by_id.insert(mesh_id, order);
        cache.mesh_representatives.insert(mesh_id, mesh.clone());
        cache.mesh_compositions.insert(mesh_id, composition);
        cache.mesh_demand_revision = cache.mesh_demand_revision.wrapping_add(1);
        stats.mesh_demand_order_updates += 1;
        return !resident;
    };
    let priority_improved = priority.0 > current.priority.0 || (priority.0 == current.priority.0 && priority.1 < current.priority.1);
    let next = MeshDemandOrder {
        mesh: mesh_id,
        resident,
        in_view: current.in_view || in_view,
        priority: if priority_improved { priority } else { current.priority },
    };
    if next != current {
        cache.mesh_order.remove(&current);
        cache.mesh_order.insert(next);
        cache.mesh_order_by_id.insert(mesh_id, next);
        cache.mesh_demand_revision = cache.mesh_demand_revision.wrapping_add(1);
        stats.mesh_demand_order_updates += 1;
    }
    if priority_improved {
        cache.mesh_representatives.insert(mesh_id, mesh.clone());
        cache.mesh_compositions.insert(mesh_id, composition);
    }
    false
}

#[cfg(test)]
pub(super) fn add_mesh_demand_for_test(cache: &mut FrameCache, stats: &mut super::Stats, mesh: &SharedMesh, priority: (f64, f64)) {
    let _ = add_mesh_demand(cache, stats, mesh, None, priority, false, true);
}

pub(super) fn range_moved(anchor: Camera, camera: Camera) -> bool {
    anchor.draw_distance() != camera.draw_distance() || anchor.position().distance(camera.position()) > RANGE_MARGIN / 2.0
}

fn in_range(item: &CachedItem, camera: Camera) -> bool {
    WorldSphere::new(item.position, item.rotation, item.scale, item.info.bounds, camera)
        .is_some_and(|sphere| sphere.in_range(item.shading, camera.range()))
}

fn contains_id(indices: &[ItemIndex], items: &ItemSlab, id: ItemId) -> bool {
    indices.binary_search_by(|index| items.get(*index).map_or(Ordering::Less, |item| item.id.cmp(&id))).is_ok()
}

fn retain_draw_slots(previous: &CachedItem, next: &mut CachedItem) {
    if previous.mesh.id() != next.mesh.id() || previous.faces.len() != next.faces.len() {
        return;
    }
    for (previous, next) in previous.faces.iter().zip(&mut next.faces) {
        if previous.order == next.order && previous.indices == next.indices {
            next.slot = previous.slot;
            next.blended_index = previous.blended_index;
            next.layout = previous.layout;
            next.committed = previous.committed;
        }
    }
}

fn update_membership(indices: &mut Vec<ItemIndex>, items: &ItemSlab, id: ItemId, index: ItemIndex, visible: bool) -> bool {
    match indices.binary_search_by(|candidate| items.get(*candidate).map_or(Ordering::Less, |item| item.id.cmp(&id))) {
        Ok(position) if !visible => {
            indices.remove(position);
            true
        }
        Err(position) if visible => {
            indices.insert(position, index);
            true
        }
        _ => false,
    }
}

#[derive(Default)]
pub(super) struct CacheUpdate {
    pub(super) layout_changed: bool,
    pub(super) items_updated: usize,
    pub(super) terrain_regions: Vec<Region>,
    pub(super) full_visibility: bool,
    pub(super) mesh_dirty: bool,
    pub(super) background: Vec<Entity>,
    pub(super) motion_changed: HashSet<Entity>,
    pub(super) pose_items: Vec<ItemIndex>,
    pub(super) parameter_items: Vec<ItemIndex>,
    pub(super) visibility_items: Vec<ItemIndex>,
    pub(super) rebuild_pending: bool,
    pub(super) rebuild_started: bool,
    pub(super) rebuild_completed: bool,
    pub(super) mesh_demand_missing: bool,
    pub(super) immediate_visibility_change: bool,
    pub(super) rebuild_visibility_changed: bool,
    pub(super) layout_invalidated: bool,
}

pub(super) fn object_item_id(object: &Object) -> ItemId {
    (2, object.detail().summary.id.as_u128(), 0)
}

fn object_rebuild_bytes(object: &Object, cached: &HashMap<MeshId, Arc<MeshInfo>>) -> usize {
    let Some((_, mesh)) = object.mesh() else { return size_of::<Object>() };
    mesh_rebuild_bytes(mesh, cached).saturating_add(size_of::<Object>())
}

fn surface_rebuild_bytes(surface: &Surface, cached: &HashMap<MeshId, Arc<MeshInfo>>) -> usize {
    match &surface.mesh {
        PreparedMesh::Ready(mesh) => mesh_rebuild_bytes(mesh, cached),
        _ => 1,
    }
}

fn mesh_rebuild_bytes(mesh: &SharedMesh, cached: &HashMap<MeshId, Arc<MeshInfo>>) -> usize {
    let source = mesh.mesh();
    let faces = source.faces().len().saturating_mul(size_of::<CachedFace>());
    if cached.contains_key(&mesh.id()) {
        faces.max(1)
    } else {
        faces
            .saturating_add(source.indices().len().saturating_mul(size_of::<u32>()))
            .saturating_add(source.faces().len().saturating_mul(size_of::<Vec3>()))
            .max(1)
    }
}

pub(super) fn item_is_live(view: &SceneView, item: &CachedItem) -> bool {
    if let Some(entity) = item.entity {
        view.object(entity).is_some_and(|object| object_item_id(object) == item.id)
    } else {
        item.surface_kind.is_some_and(|kind| {
            view.surface(item.region, kind).is_some_and(|surface| surface_item_id(surface.region, surface.kind) == item.id)
        })
    }
}

fn animated_face_indices(object: &Object, faces: &[CachedFace]) -> Vec<usize> {
    let Some(animation) =
        object.detail().texture_animation.filter(|animation| texture_animation::is_enabled(*animation) && animation.rate() != 0.0)
    else {
        return Vec::new();
    };
    faces
        .iter()
        .enumerate()
        .filter_map(|(index, face)| (animation.targets_face(face.order.3) && face.base_params.textures.has_images()).then_some(index))
        .collect()
}

fn surface_item_id(region: Region, kind: SurfaceKind) -> ItemId {
    match kind {
        SurfaceKind::RegionWater => (0, u128::from(region.get()), 0),
        SurfaceKind::Terrain(patch) => {
            let (x, y) = patch.coordinates();
            (1, u128::from(region.get()), u16::from(y) * 16 + u16::from(x))
        }
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
