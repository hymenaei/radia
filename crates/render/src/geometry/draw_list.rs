use std::{
    array,
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
    mem,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use assets::TextureHandle;
use math::{DVec3, Frustum, Quat, Sphere, Vec3};
use wgpu::{Buffer, Queue};
use world::Region;

use super::{
    ShadingMode, TRANSFORM_BYTES,
    instance::{PARAMETER_BYTES, instance_base},
    items::{CachedFace, CachedItem, ItemId, ItemIndex, ItemSlab, draw_priority, surface_pipeline},
    mesh::GpuMesh,
    storage::Buffers,
    texture_bindings::{MaterialImages, TerrainDemand, TextureDemand},
    visibility::bounds_sphere,
};
use crate::{
    pipeline::{INSTANCE_BYTES, Pipeline},
    scene::{Camera, MeshId},
    texture::MaterialTextures,
};

pub(super) const INDIRECT_DRAW_BYTES: usize = 20;
pub(super) type FaceOrder = (u8, u128, u16, u8);
const CELL_SIZE: f64 = 16.0;
const MAX_PATCH_RANGES: usize = 32;
const MAX_LAYOUT_UPLOAD_RANGES: usize = 256;
const MAX_PATCH_GAP_SLOTS: u32 = 4;
const LAYOUT_CLOCK_STEPS: usize = 64;
const LAYOUT_BYTES_PER_FRAME: usize = 8 * 1024 * 1024;
pub(super) const LAYOUT_TIME_PER_FRAME: Duration = Duration::from_millis(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LayoutId(u64);

#[derive(Clone, Copy, Debug)]
pub(super) struct FaceSlot {
    layout: LayoutId,
    slot: u32,
    blended_index: Option<usize>,
}

#[derive(Default)]
pub(super) struct UploadStats {
    pub(super) bytes: usize,
    pub(super) calls: usize,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct DrawKey {
    pub(super) mesh: MeshId,
    pub(super) start: u32,
    pub(super) end: u32,
    pub(super) pipeline: Pipeline,
    pub(super) textures: MaterialTextures,
    pub(super) terrain: Option<Region>,
}

impl DrawKey {
    pub(super) fn blended(self) -> bool {
        matches!(self.pipeline, Pipeline::Blend | Pipeline::BlendDoubleSided | Pipeline::BlendFlipped)
    }

    pub(super) fn fullbright(self) -> bool {
        matches!(self.pipeline, Pipeline::Fullbright | Pipeline::FullbrightDoubleSided | Pipeline::FullbrightFlipped)
    }

    fn command(self, group: u32, first: u32, count: u32, cell: u32) -> DrawCommand {
        DrawCommand {
            group,
            mesh: self.mesh,
            pipeline: self.pipeline,
            first,
            count,
            indices: self.start..self.end,
            textures: self.textures,
            terrain: self.terrain,
            cell,
            sphere: None,
            ranged: true,
            live: true,
        }
    }
}

pub(super) struct DrawCommand {
    group: u32,
    pub(super) mesh: MeshId,
    pub(super) pipeline: Pipeline,
    pub(super) first: u32,
    pub(super) count: u32,
    pub(super) indices: Range<u32>,
    pub(super) textures: MaterialTextures,
    pub(super) terrain: Option<Region>,
    cell: u32,
    sphere: Option<Sphere>,
    ranged: bool,
    live: bool,
}

impl DrawCommand {
    fn include(&mut self, sphere: Sphere, shading: ShadingMode) {
        self.sphere = Some(self.sphere.map_or(sphere, |bounds| bounds.union(sphere)));
        self.ranged &= shading == ShadingMode::Surface;
    }

    fn visible(&self, cells: &[CellClass], eye: DVec3, draw_distance: f64, frustum: &Frustum) -> bool {
        if !self.live {
            return false;
        }
        let Some(sphere) = self.sphere else { return false };
        let Some(cell) = cells.get(self.cell as usize) else { return false };
        if !cell.visible {
            return false;
        }
        let center = sphere.center - eye;
        if self.ranged && !cell.near && center.length() - sphere.radius > draw_distance {
            return false;
        }
        cell.inside || frustum.intersects_sphere(center, sphere.radius)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct DrawBatchKey {
    pub(super) pipeline: Pipeline,
    pub(super) textures: MaterialTextures,
    pub(super) terrain: Option<Region>,
    page: usize,
}

pub(super) struct DrawBatch {
    id: usize,
    pub(super) key: DrawBatchKey,
    pub(super) buffers: Arc<Buffers>,
    pub(super) offset: u64,
    pub(super) count: u32,
}

struct BatchGroup {
    key: DrawBatchKey,
    buffers: Arc<Buffers>,
    refs: usize,
}

#[derive(Clone, Copy)]
struct BatchBinding {
    id: usize,
    index_offset: u32,
    base_vertex: i32,
}

#[derive(Clone, Copy)]
struct DrawIndexedArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[derive(Clone, Copy, Default)]
struct CellClass {
    visible: bool,
    inside: bool,
    near: bool,
}

impl CellClass {
    fn new(cell: Option<Sphere>, eye: DVec3, draw_distance: f64, frustum: &Frustum) -> Self {
        let Some(sphere) = cell else { return Self::default() };
        let center = sphere.center - eye;
        Self {
            visible: frustum.intersects_sphere(center, sphere.radius),
            inside: frustum.contains_sphere(center, sphere.radius),
            near: center.length() + sphere.radius <= draw_distance,
        }
    }
}

#[derive(Default)]
pub(super) struct ViewDraws {
    batches: Vec<DrawBatch>,
    indirect_bytes: Vec<u8>,
    first_instances: Vec<u32>,
    encoded: Vec<u8>,
    cells: Vec<CellClass>,
    opaque: Vec<u32>,
    blended: Vec<u32>,
    blended_distances: Vec<f64>,
    blended_marks: Vec<u64>,
    blended_keys: Vec<(f64, FaceOrder, u32)>,
    batch_commands: Vec<Vec<DrawIndexedArgs>>,
    batch_order: Vec<usize>,
    triangles: u64,
    mark: u64,
    layout: LayoutId,
}

impl ViewDraws {
    pub(super) fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    pub(super) fn indirect_bytes(&self) -> &[u8] {
        &self.indirect_bytes
    }

    pub(super) fn command_count(&self) -> usize {
        self.first_instances.len()
    }

    pub(super) fn triangles(&self) -> u64 {
        self.triangles
    }

    pub(super) fn first_instance(&self, offset: u64, command: u32) -> Option<u32> {
        let index = usize::try_from(offset).ok()?.checked_div(INDIRECT_DRAW_BYTES)?.checked_add(command as usize)?;
        self.first_instances.get(index).copied()
    }

    #[cfg(test)]
    pub(super) fn blended_order(&self) -> Vec<u32> {
        self.blended.clone()
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Default)]
pub(super) struct CullStats {
    pub(super) commands: usize,
    pub(super) changed: bool,
    pub(super) blended_sorted: bool,
}

struct DrawGroup {
    key: DrawKey,
    refs: usize,
    images: MaterialImages,
    terrain_textures: Option<[Option<TextureHandle>; 4]>,
    batch: Option<BatchBinding>,
}

#[derive(Default)]
struct DrawInterner {
    by_key: HashMap<DrawKey, u32>,
    groups: Vec<Option<DrawGroup>>,
    free: Vec<u32>,
    batches_by_key: HashMap<DrawBatchKey, usize>,
    batches: Vec<Option<BatchGroup>>,
    free_batches: Vec<usize>,
}

impl DrawInterner {
    pub(super) fn intern(
        &mut self,
        key: DrawKey,
        images: &MaterialImages,
        terrain_textures: Option<&[Option<TextureHandle>; 4]>,
    ) -> Option<u32> {
        if let Some(id) = self.by_key.get(&key).copied() {
            let group = self.groups.get_mut(id as usize).and_then(Option::as_mut)?;
            group.refs = group.refs.checked_add(1)?;
            if !images.is_empty() {
                group.images = images.clone();
            }
            if terrain_textures.is_some() {
                group.terrain_textures = terrain_textures.cloned();
            }
            return Some(id);
        }
        let id = match self.free.pop() {
            Some(id) => id,
            None => u32::try_from(self.groups.len()).ok()?,
        };
        if id as usize == self.groups.len() {
            self.groups.push(None);
        }
        let slot = self.groups.get_mut(id as usize)?;
        *slot = Some(DrawGroup { key, refs: 1, images: images.clone(), terrain_textures: terrain_textures.cloned(), batch: None });
        self.by_key.insert(key, id);
        Some(id)
    }

    pub(super) fn key(&self, id: u32) -> Option<DrawKey> {
        self.groups.get(id as usize).and_then(Option::as_ref).map(|group| group.key)
    }

    fn group(&self, id: u32) -> Option<&DrawGroup> {
        self.groups.get(id as usize).and_then(Option::as_ref)
    }

    fn retain(&mut self, id: u32) -> bool {
        let Some(group) = self.groups.get_mut(id as usize).and_then(Option::as_mut) else { return false };
        let Some(refs) = group.refs.checked_add(1) else { return false };
        group.refs = refs;
        true
    }

    fn batch(&self, id: u32) -> Option<BatchBinding> {
        self.group(id).and_then(|group| group.batch)
    }

    fn set_batch(&mut self, id: u32, batch: BatchBinding) {
        if let Some(group) = self.groups.get_mut(id as usize).and_then(Option::as_mut) {
            group.batch = Some(batch);
        }
    }

    fn batch_group(&self, id: usize) -> Option<&BatchGroup> {
        self.batches.get(id).and_then(Option::as_ref)
    }

    pub(super) fn release(&mut self, id: u32) {
        let batch = {
            let Some(group) = self.groups.get_mut(id as usize).and_then(Option::as_mut) else { return };
            if group.refs == 0 {
                return;
            }
            group.refs -= 1;
            if group.refs != 0 {
                return;
            }
            group.batch.map(|batch| batch.id)
        };
        let key = self.groups.get(id as usize).and_then(Option::as_ref).map(|group| group.key);
        let Some(key) = key else { return };
        self.by_key.remove(&key);
        if let Some(slot) = self.groups.get_mut(id as usize) {
            *slot = None;
            self.free.push(id);
        }
        if let Some(batch) = batch {
            self.release_batch(batch);
        }
    }

    fn intern_batch(&mut self, key: DrawBatchKey, buffers: &Arc<Buffers>) -> usize {
        if let Some(id) = self.batches_by_key.get(&key).copied() {
            if let Some(group) = self.batches.get_mut(id).and_then(Option::as_mut) {
                group.refs += 1;
            }
            return id;
        }
        let id = match self.free_batches.pop() {
            Some(id) => id,
            None => self.batches.len(),
        };
        if id == self.batches.len() {
            self.batches.push(None);
        }
        if let Some(slot) = self.batches.get_mut(id) {
            *slot = Some(BatchGroup { key, buffers: Arc::clone(buffers), refs: 1 });
        }
        self.batches_by_key.insert(key, id);
        id
    }

    fn release_batch(&mut self, id: usize) {
        let Some(group) = self.batches.get_mut(id).and_then(Option::as_mut) else { return };
        if group.refs == 0 {
            return;
        }
        group.refs -= 1;
        if group.refs != 0 {
            return;
        }
        self.batches_by_key.remove(&group.key);
        if let Some(slot) = self.batches.get_mut(id) {
            *slot = None;
            self.free_batches.push(id);
        }
    }

    fn invalidate_batches(&mut self) {
        self.batches_by_key.clear();
        self.batches.clear();
        self.free_batches.clear();
        for group in self.groups.iter_mut().flatten() {
            group.batch = None;
        }
    }

    pub(super) fn update_terrain(&mut self, region: Region, textures: &[Option<TextureHandle>; 4]) {
        for group in self.groups.iter_mut().flatten().filter(|group| group.key.terrain == Some(region)) {
            group.terrain_textures = Some(textures.clone());
        }
    }
}

#[derive(Clone, Copy)]
struct BlendedFace {
    order: FaceOrder,
    item: ItemIndex,
    face: usize,
    group: u32,
    cell: u32,
    center: Vec3,
}

#[derive(Clone, Copy, Default)]
struct PendingCommand {
    count: usize,
    cursor: usize,
    first: usize,
    draw: usize,
}

#[derive(Default)]
pub(super) struct LayoutStats {
    pub(super) omitted_faces: usize,
    pub(super) items_visited: usize,
    pub(super) faces_visited: usize,
}

pub(super) struct LayoutProgress {
    pub(super) stats: LayoutStats,
    pub(super) complete: bool,
    pub(super) stale: bool,
    pub(super) commit_ms: f64,
}

#[derive(Clone, Copy)]
struct PriorityItem {
    priority: f64,
    distance: f64,
    id: ItemId,
    index: ItemIndex,
    faces: usize,
}

impl Eq for PriorityItem {}

impl PartialEq for PriorityItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority.total_cmp(&other.priority) == Ordering::Equal
            && self.distance.total_cmp(&other.distance) == Ordering::Equal
            && self.id == other.id
            && self.index == other.index
            && self.faces == other.faces
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.total_cmp(&other.priority).then_with(|| other.distance.total_cmp(&self.distance)).then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct LayoutBuild {
    layout: LayoutId,
    state: LayoutState,
    uploads: HashSet<u32>,
    visible: Vec<ItemIndex>,
    removed: HashSet<ItemIndex>,
    evicted: HashSet<MeshId>,
    layout_items: Vec<ItemIndex>,
    selected_items: HashSet<ItemIndex>,
    phase: LayoutPhase,
    camera: Camera,
    item_cursor: usize,
    face_cursor: usize,
    item_has_slot: bool,
    item_sphere: Option<Sphere>,
    command_cursor: usize,
    commands: HashMap<(u32, u32), PendingCommand>,
    command_keys: Vec<(u32, u32)>,
    cells: HashMap<[i64; 3], u32>,
    item_cells: Vec<u32>,
    capacity: usize,
    required: usize,
    remaining: usize,
    omitted: usize,
    opaque_count: usize,
    instance_bytes: usize,
    instance_reserved: bool,
    blend_index: usize,
    visible_items: usize,
    priority: BinaryHeap<PriorityItem>,
    blended: Vec<BlendedFace>,
    items_visited: usize,
    faces_visited: usize,
    stale: bool,
}

#[derive(Clone, Copy)]
enum LayoutPhase {
    CountVisible,
    ClearInitialCounts,
    Priority,
    Select,
    CountSelected,
    HideOmitted,
    Commands,
    FullbrightCommands,
    Allocate,
    Fill,
    EmitBlended,
    Commit,
}

impl LayoutBuild {
    fn new(layout: LayoutId, visible: &[ItemIndex], camera: Camera, capacity: usize) -> Self {
        Self {
            layout,
            state: LayoutState::default(),
            uploads: HashSet::new(),
            visible: visible.to_vec(),
            removed: HashSet::new(),
            evicted: HashSet::new(),
            layout_items: Vec::new(),
            selected_items: HashSet::new(),
            phase: LayoutPhase::CountVisible,
            camera,
            item_cursor: 0,
            face_cursor: 0,
            item_has_slot: false,
            item_sphere: None,
            command_cursor: 0,
            commands: HashMap::new(),
            command_keys: Vec::new(),
            cells: HashMap::new(),
            item_cells: Vec::new(),
            capacity,
            required: 0,
            remaining: capacity,
            omitted: 0,
            opaque_count: 0,
            instance_bytes: 0,
            instance_reserved: false,
            blend_index: 0,
            visible_items: 0,
            priority: BinaryHeap::new(),
            blended: Vec::new(),
            items_visited: 0,
            faces_visited: 0,
            stale: false,
        }
    }

    fn upload(&mut self, slot: u32) {
        if self.uploads.insert(slot) {
            self.state.dirty_slots.push(slot);
        }
    }

    fn input_items(&self) -> &[ItemIndex] {
        if self.over_capacity() { &self.layout_items } else { &self.visible }
    }

    fn over_capacity(&self) -> bool {
        self.required > self.capacity
    }

    fn cell(&mut self, position: DVec3) -> u32 {
        let key = (position / CELL_SIZE).floor().to_array().map(|coordinate| coordinate as i64);
        let next = u32::try_from(self.state.cells.len()).unwrap_or(u32::MAX);
        let cell = *self.cells.entry(key).or_insert(next);
        if cell == next && cell != u32::MAX {
            self.state.cells.push(None);
        }
        cell
    }

    fn push_draw(&mut self, draw: DrawCommand) {
        let mesh = draw.mesh;
        let index = u32::try_from(self.state.draws.len()).unwrap_or(u32::MAX);
        let live = !self.evicted.contains(&mesh);
        self.state.draws.push(DrawCommand { live, ..draw });
        if live {
            self.state.mesh_references.entry(mesh).or_default().push(index);
        }
    }

    fn retired(&self) -> bool {
        let state = &self.state;
        state.draw_group_refs.is_empty()
            && state.active_textures.is_empty()
            && state.active_terrain.is_empty()
            && state.layout_items.is_empty()
    }
}

#[derive(Default)]
pub(super) struct LayoutState {
    draws: Vec<DrawCommand>,
    cells: Vec<Option<Sphere>>,
    mesh_references: HashMap<MeshId, Vec<u32>>,
    instance_bytes: Vec<u8>,
    active_textures: Vec<TextureDemand>,
    active_terrain: Vec<TerrainDemand>,
    texture_seen: HashSet<MaterialTextures>,
    terrain_seen: HashSet<Region>,
    layout_items: Vec<ItemIndex>,
    draw_group_refs: Vec<u32>,
    draw_group_seen: HashSet<u32>,
    blended: Vec<BlendedFace>,
    dirty_slots: Vec<u32>,
    opaque_draws: usize,
    visible_items: usize,
    omitted: usize,
    valid: bool,
}

impl LayoutState {
    fn opaque(&self) -> &[DrawCommand] {
        self.draws.get(..self.opaque_draws).unwrap_or_default()
    }

    fn disable(&mut self, mesh: MeshId) -> bool {
        let Some(draws) = self.mesh_references.remove(&mesh) else { return false };
        for index in draws {
            if let Some(draw) = self.draws.get_mut(index as usize) {
                draw.live = false;
            }
        }
        true
    }

    fn include(&mut self, slot: u32, blended_index: Option<usize>, sphere: Sphere, shading: ShadingMode) {
        let index = match blended_index {
            Some(index) => self.opaque_draws.saturating_add(index),
            None => self.opaque().partition_point(|draw| draw.first.saturating_add(draw.count) <= slot),
        };
        let Some(draw) = self.draws.get_mut(index) else { return };
        if slot < draw.first || slot - draw.first >= draw.count {
            return;
        }
        if draw.count == 1 {
            draw.sphere = None;
        }
        draw.include(sphere, shading);
        if let Some(cell) = self.cells.get_mut(draw.cell as usize) {
            *cell = Some(cell.map_or(sphere, |bounds| bounds.union(sphere)));
        }
    }
}

pub(super) struct DrawList {
    capacity: usize,
    groups: DrawInterner,
    committed: LayoutState,
    cull_dirty: bool,
    pending: Option<LayoutBuild>,
    retired: Option<LayoutBuild>,
    layout: LayoutId,
    last_layout: LayoutId,
}

impl DrawList {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            groups: DrawInterner::default(),
            committed: LayoutState::default(),
            cull_dirty: false,
            pending: None,
            retired: None,
            layout: LayoutId::default(),
            last_layout: LayoutId::default(),
        }
    }

    pub(super) fn request_layout(&mut self, visible: &[ItemIndex], camera: Camera) -> bool {
        if self.pending.is_some() {
            return false;
        }
        if self.retired.is_some() {
            return false;
        }
        let layout = self.next_layout();
        self.pending = Some(LayoutBuild::new(layout, visible, camera, self.capacity));
        true
    }

    fn next_layout(&mut self) -> LayoutId {
        self.last_layout = LayoutId(self.last_layout.0.checked_add(1).expect("layout id space exhausted"));
        self.last_layout
    }

    fn committed_address(&self, face: &CachedFace) -> Option<FaceSlot> {
        if face.layout == self.layout {
            return face.slot.map(|slot| FaceSlot { layout: face.layout, slot, blended_index: face.blended_index });
        }
        face.committed.filter(|committed| committed.layout == self.layout)
    }

    pub(super) fn evict(&mut self, mesh: MeshId) {
        if self.committed.disable(mesh) {
            self.cull_dirty = true;
        }
        if let Some(build) = self.pending.as_mut() {
            build.evicted.insert(mesh);
            build.state.disable(mesh);
        }
    }

    pub(super) fn shows_mesh(&self, mesh: MeshId, camera: Camera, frustum: &Frustum) -> bool {
        let Some(draws) = self.committed.mesh_references.get(&mesh) else { return false };
        let eye = camera.position();
        let draw_distance = camera.draw_distance();
        draws.iter().filter_map(|index| self.committed.draws.get(*index as usize)).any(|draw| {
            draw.sphere.is_some_and(|sphere| {
                let center = sphere.center - eye;
                (!draw.ranged || center.length() - sphere.radius <= draw_distance) && frustum.intersects_sphere(center, sphere.radius)
            })
        })
    }

    #[cfg(test)]
    pub(super) fn committed_pins_mesh(&self, mesh: MeshId) -> bool {
        self.committed.mesh_references.contains_key(&mesh)
    }

    #[cfg(test)]
    pub(super) fn staged_progress(&self) -> Option<(&'static str, usize, usize, usize, usize, usize, usize)> {
        let build = self.pending.as_ref()?;
        let phase = match build.phase {
            LayoutPhase::CountVisible => "count_visible",
            LayoutPhase::ClearInitialCounts => "clear_initial_counts",
            LayoutPhase::Priority => "priority",
            LayoutPhase::Select => "select",
            LayoutPhase::CountSelected => "count_selected",
            LayoutPhase::HideOmitted => "hide_omitted",
            LayoutPhase::Commands => "commands",
            LayoutPhase::FullbrightCommands => "fullbright_commands",
            LayoutPhase::Allocate => "allocate",
            LayoutPhase::Fill => "fill",
            LayoutPhase::EmitBlended => "emit_blended",
            LayoutPhase::Commit => "commit",
        };
        Some((
            phase,
            build.item_cursor,
            build.face_cursor,
            build.visible.len(),
            build.layout_items.len(),
            build.items_visited,
            build.faces_visited,
        ))
    }

    pub(super) fn cancel_layout(&mut self) {
        if self.retired.is_none()
            && let Some(build) = self.pending.take()
        {
            self.retired = Some(build);
        }
    }

    pub(super) fn layout_pending(&self) -> bool {
        self.pending.is_some() || self.retired.is_some()
    }

    #[cfg(test)]
    pub(super) fn layout_queues(&self) -> (bool, bool) {
        (self.pending.is_some(), self.retired.is_some())
    }

    #[cfg(test)]
    pub(super) fn retired_entries_remaining(&self) -> usize {
        let Some(build) = self.retired.as_ref() else { return 0 };
        let state = &build.state;
        state.draw_group_refs.len() + state.active_textures.len() + state.active_terrain.len() + state.layout_items.len()
    }

    pub(super) fn layout_building(&self) -> bool {
        self.pending.is_some()
    }

    pub(super) fn mark_layout_stale(&mut self) {
        if let Some(build) = self.pending.as_mut() {
            build.stale = true;
        }
    }

    pub(super) fn advance_layout(&mut self, items: &mut ItemSlab, budget: Duration) -> Option<LayoutProgress> {
        let started = Instant::now();
        self.retire_layout_step(items, started, budget);
        if self.retired.is_some() {
            return None;
        }
        let mut build = self.pending.take()?;
        if build.stale {
            self.retired = Some(build);
            return Some(LayoutProgress { stats: LayoutStats::default(), complete: false, stale: true, commit_ms: 0.0 });
        }
        let mut steps = 0usize;
        let mut byte_work = 0usize;
        let mut items_visited = 0usize;
        let mut faces_visited = 0usize;
        let mut commit_ms = 0.0;
        while byte_work < LAYOUT_BYTES_PER_FRAME {
            steps += 1;
            if steps.is_multiple_of(LAYOUT_CLOCK_STEPS) && started.elapsed() >= budget {
                break;
            }
            match build.phase {
                LayoutPhase::CountVisible | LayoutPhase::CountSelected => {
                    let count_visible = matches!(build.phase, LayoutPhase::CountVisible);
                    let input = if count_visible { &build.visible } else { &build.layout_items };
                    let Some(&index) = input.get(build.item_cursor) else {
                        build.item_cursor = 0;
                        build.face_cursor = 0;
                        if count_visible && build.over_capacity() {
                            build.phase = LayoutPhase::ClearInitialCounts;
                            build.state.visible_items = 0;
                        } else if count_visible {
                            build.phase = LayoutPhase::Commands;
                            build.command_cursor = 0;
                        } else {
                            build.phase = LayoutPhase::HideOmitted;
                        }
                        continue;
                    };
                    let Some(item) = items.get(index).filter(|_| !build.removed.contains(&index)) else {
                        if build.face_cursor == 0 {
                            build.item_cells.push(u32::MAX);
                        }
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        continue;
                    };
                    if build.face_cursor == 0 {
                        items_visited += 1;
                        build.items_visited += 1;
                        let cell = build.cell(item.position);
                        build.item_cells.push(cell);
                    }
                    let Some(face) = item.faces.get(build.face_cursor) else {
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        continue;
                    };
                    let group = face.group;
                    let cell = build.item_cells.last().copied().unwrap_or(u32::MAX);
                    let pending = build.commands.entry((group, cell)).or_insert_with(|| {
                        build.command_keys.push((group, cell));
                        PendingCommand::default()
                    });
                    pending.count = pending.count.saturating_add(1);
                    if count_visible {
                        build.required = build.required.saturating_add(1);
                    }
                    build.face_cursor += 1;
                    faces_visited += 1;
                    build.faces_visited += 1;
                    byte_work += size_of::<CachedFace>();
                }
                LayoutPhase::ClearInitialCounts => {
                    build.commands.clear();
                    build.command_keys.clear();
                    build.cells.clear();
                    build.item_cells.clear();
                    build.state.cells.clear();
                    build.phase = LayoutPhase::Priority;
                    build.item_cursor = 0;
                }
                LayoutPhase::Priority => {
                    let Some(&index) = build.visible.get(build.item_cursor) else {
                        build.item_cursor = 0;
                        build.phase = LayoutPhase::Select;
                        continue;
                    };
                    build.item_cursor += 1;
                    items_visited += 1;
                    build.items_visited += 1;
                    let Some(item) = items.get(index).filter(|_| !build.removed.contains(&index)) else { continue };
                    let (priority, distance) = draw_priority(item, build.camera);
                    build.priority.push(PriorityItem { priority, distance, id: item.id, index, faces: item.faces.len() });
                }
                LayoutPhase::Select => {
                    let Some(candidate) = build.priority.pop() else {
                        build.phase = LayoutPhase::CountSelected;
                        build.item_cursor = 0;
                        build.face_cursor = 0;
                        continue;
                    };
                    items_visited += 1;
                    build.items_visited += 1;
                    if candidate.faces <= build.remaining {
                        build.remaining -= candidate.faces;
                        build.layout_items.push(candidate.index);
                        build.selected_items.insert(candidate.index);
                    } else {
                        build.omitted = build.omitted.saturating_add(candidate.faces);
                    }
                }
                LayoutPhase::HideOmitted => {
                    let Some(&index) = build.over_capacity().then(|| build.visible.get(build.item_cursor)).flatten() else {
                        build.phase = LayoutPhase::Commands;
                        build.item_cursor = 0;
                        build.face_cursor = 0;
                        build.command_cursor = 0;
                        continue;
                    };
                    if build.selected_items.contains(&index) {
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        items_visited += 1;
                        build.items_visited += 1;
                        continue;
                    }
                    if build.face_cursor == 0 {
                        items_visited += 1;
                        build.items_visited += 1;
                    }
                    let Some(item) = items.get_mut(index).filter(|_| !build.removed.contains(&index)) else {
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        continue;
                    };
                    if build.face_cursor >= item.faces.len() {
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        continue;
                    }
                    if let Some(face) = item.faces.get_mut(build.face_cursor) {
                        self.hide_face(face);
                        faces_visited += 1;
                        build.faces_visited += 1;
                        byte_work += size_of::<CachedFace>();
                    }
                    build.face_cursor += 1;
                }
                LayoutPhase::Commands | LayoutPhase::FullbrightCommands => {
                    let fullbright = matches!(build.phase, LayoutPhase::FullbrightCommands);
                    let Some(key) = build.command_keys.get(build.command_cursor).copied() else {
                        build.command_cursor = 0;
                        if fullbright {
                            build.state.opaque_draws = build.state.draws.len();
                            build.phase = LayoutPhase::Allocate;
                        } else {
                            build.phase = LayoutPhase::FullbrightCommands;
                        }
                        continue;
                    };
                    let (group_id, cell) = key;
                    if let Some(group) = self.groups.group(group_id) {
                        if !fullbright
                            && group.key.textures != MaterialTextures::default()
                            && build.state.texture_seen.insert(group.key.textures)
                        {
                            build.state.active_textures.push(TextureDemand { key: group.key.textures, images: group.images.clone() });
                        }
                        if !fullbright
                            && let Some(region) = group.key.terrain
                            && build.state.terrain_seen.insert(region)
                        {
                            build.state.active_terrain.push(TerrainDemand {
                                region,
                                textures: group.terrain_textures.clone().unwrap_or_else(|| array::from_fn(|_| None)),
                            });
                        }
                        if !group.key.blended()
                            && group.key.fullbright() == fullbright
                            && let Some(pending) = build.commands.get_mut(&key)
                            && pending.count > 0
                            && let (Ok(first), Ok(count)) = (u32::try_from(build.opaque_count), u32::try_from(pending.count))
                        {
                            pending.first = build.opaque_count;
                            pending.draw = build.state.draws.len();
                            build.opaque_count = build.opaque_count.saturating_add(pending.count);
                            build.push_draw(group.key.command(group_id, first, count, cell));
                            self.retain_layout_group(&mut build.state, group_id);
                        }
                    }
                    build.command_cursor += 1;
                }
                LayoutPhase::Allocate => {
                    let target = build.opaque_count.saturating_mul(INSTANCE_BYTES);
                    build.instance_bytes = target;
                    if !build.instance_reserved {
                        build.state.instance_bytes.reserve_exact(target);
                        build.instance_reserved = true;
                    }
                    if build.state.instance_bytes.len() < target {
                        let allowance = LAYOUT_BYTES_PER_FRAME.saturating_sub(byte_work).min(target - build.state.instance_bytes.len());
                        if allowance == 0 {
                            break;
                        }
                        let end = build.state.instance_bytes.len() + allowance;
                        build.state.instance_bytes.resize(end, 0);
                        byte_work += allowance;
                        continue;
                    }
                    build.item_cursor = 0;
                    build.face_cursor = 0;
                    build.phase = LayoutPhase::Fill;
                }
                LayoutPhase::Fill => {
                    let Some(&index) = build.input_items().get(build.item_cursor) else {
                        build.phase = LayoutPhase::EmitBlended;
                        continue;
                    };
                    let cell = build.item_cells.get(build.item_cursor).copied().unwrap_or(u32::MAX);
                    let Some(item) = items.get_mut(index).filter(|_| !build.removed.contains(&index)) else {
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        build.item_has_slot = false;
                        continue;
                    };
                    if build.face_cursor == 0 {
                        items_visited += 1;
                        build.items_visited += 1;
                        build.item_has_slot = false;
                        build.item_sphere = bounds_sphere(item.position, item.rotation, item.scale, item.info.bounds);
                    }
                    if build.face_cursor >= item.faces.len() {
                        build.visible_items += usize::from(build.item_has_slot);
                        build.item_cursor += 1;
                        build.face_cursor = 0;
                        build.item_has_slot = false;
                        continue;
                    }
                    let face_index = build.face_cursor;
                    let Some(face) = item.faces.get_mut(face_index) else {
                        build.stale = true;
                        break;
                    };
                    if face.blended {
                        build.item_has_slot = true;
                        build.blended.push(BlendedFace {
                            order: face.order,
                            item: index,
                            face: face_index,
                            group: face.group,
                            cell,
                            center: face.center,
                        });
                    } else {
                        let placement = build.commands.get_mut(&(face.group, cell)).and_then(|pending| {
                            (pending.cursor < pending.count).then(|| {
                                pending.cursor += 1;
                                (pending.first.saturating_add(pending.cursor - 1), pending.draw)
                            })
                        });
                        let slot = placement.and_then(|(offset, _)| u32::try_from(offset).ok()).filter(|slot| {
                            write_layout_instance(
                                &self.committed.instance_bytes,
                                &mut build.state.instance_bytes,
                                &mut build.state.dirty_slots,
                                *slot,
                                &item.transform,
                                &face.parameter_bytes,
                            )
                        });
                        assign_slot(face, build.layout, self.layout, slot, None);
                        build.item_has_slot |= slot.is_some();
                        if slot.is_some()
                            && let Some((_, draw)) = placement
                            && let Some(sphere) = build.item_sphere
                            && let Some(command) = build.state.draws.get_mut(draw)
                        {
                            command.include(sphere, item.shading);
                            if let Some(bounds) = build.state.cells.get_mut(cell as usize) {
                                *bounds = Some(bounds.map_or(sphere, |bounds| bounds.union(sphere)));
                            }
                        }
                    }
                    build.face_cursor += 1;
                    faces_visited += 1;
                    build.faces_visited += 1;
                    byte_work += size_of::<CachedFace>();
                }
                LayoutPhase::EmitBlended => {
                    let Some(blended) = build.blended.pop() else {
                        build.phase = LayoutPhase::Commit;
                        continue;
                    };
                    if build.removed.contains(&blended.item) {
                        continue;
                    }
                    let Some(item) = items.get_mut(blended.item) else {
                        build.stale = true;
                        continue;
                    };
                    let sphere = bounds_sphere(item.position, item.rotation, item.scale, item.info.bounds);
                    let shading = item.shading;
                    let Some(face) = item.faces.get_mut(blended.face) else {
                        build.stale = true;
                        continue;
                    };
                    if face.group != blended.group || face.order != blended.order || !face.blended {
                        build.stale = true;
                        continue;
                    }
                    let Ok(slot) = u32::try_from(build.opaque_count.saturating_add(build.blend_index)) else {
                        build.stale = true;
                        continue;
                    };
                    let blend_index = build.blend_index;
                    build.blend_index += 1;
                    build.state.blended.push(blended);
                    let start = slot as usize * INSTANCE_BYTES;
                    let old = self.committed.instance_bytes.get(start..start + INSTANCE_BYTES);
                    build.state.instance_bytes.extend_from_slice(&item.transform);
                    build.state.instance_bytes.extend_from_slice(&face.parameter_bytes);
                    let new = build.state.instance_bytes.get(start..start + INSTANCE_BYTES);
                    if old.zip(new).is_none_or(|(old, new)| old != new) {
                        build.state.dirty_slots.push(slot);
                    }
                    let Some(key) = self.groups.key(blended.group) else {
                        build.stale = true;
                        break;
                    };
                    assign_slot(face, build.layout, self.layout, Some(slot), Some(blend_index));
                    let mut command = key.command(blended.group, slot, 1, blended.cell);
                    if let Some(sphere) = sphere {
                        command.include(sphere, shading);
                        if let Some(bounds) = build.state.cells.get_mut(blended.cell as usize) {
                            *bounds = Some(bounds.map_or(sphere, |bounds| bounds.union(sphere)));
                        }
                    }
                    build.push_draw(command);
                    self.retain_layout_group(&mut build.state, blended.group);
                    faces_visited += 1;
                    build.faces_visited += 1;
                    byte_work += INSTANCE_BYTES;
                }
                LayoutPhase::Commit => break,
            }
        }
        if build.stale {
            self.retired = Some(build);
            return Some(LayoutProgress {
                stats: LayoutStats { items_visited, faces_visited, ..LayoutStats::default() },
                complete: false,
                stale: true,
                commit_ms,
            });
        }
        let complete = matches!(build.phase, LayoutPhase::Commit);
        if complete {
            let commit_started = Instant::now();
            build.state.layout_items =
                if build.over_capacity() { mem::take(&mut build.layout_items) } else { mem::take(&mut build.visible) };
            build.state.visible_items = build.visible_items;
            build.state.omitted = build.omitted;
            build.state.valid = true;
            let omitted_faces = build.omitted;
            let stale = build.stale;
            mem::swap(&mut self.committed, &mut build.state);
            self.layout = build.layout;
            self.cull_dirty = true;
            self.retired = Some(build);
            commit_ms += commit_started.elapsed().as_secs_f64() * 1000.0;
            Some(LayoutProgress { stats: LayoutStats { omitted_faces, items_visited, faces_visited }, complete: true, stale, commit_ms })
        } else {
            let stale = build.stale;
            let omitted_faces = build.omitted;
            self.pending = Some(build);
            Some(LayoutProgress { stats: LayoutStats { omitted_faces, items_visited, faces_visited }, complete: false, stale, commit_ms })
        }
    }

    fn retain_layout_group(&mut self, state: &mut LayoutState, group: u32) {
        if state.draw_group_seen.insert(group) && self.groups.retain(group) {
            state.draw_group_refs.push(group);
        }
    }

    fn retire_layout_step(&mut self, items: &mut ItemSlab, started: Instant, budget: Duration) {
        let Some(build) = self.retired.as_mut() else { return };
        let mut steps = 0usize;
        while !build.retired() {
            steps += 1;
            if steps.is_multiple_of(LAYOUT_CLOCK_STEPS) && started.elapsed() >= budget {
                break;
            }
            if let Some(group) = build.state.draw_group_refs.pop() {
                self.groups.release(group);
            } else if build.state.active_textures.pop().is_some() || build.state.active_terrain.pop().is_some() {
                continue;
            } else if let Some(item) = build.state.layout_items.pop().and_then(|index| items.get_mut(index)) {
                for face in item.faces.iter_mut().filter(|face| face.layout != self.layout) {
                    face.slot = None;
                    face.blended_index = None;
                    face.committed = None;
                }
            }
        }
        if self.retired.as_ref().is_some_and(LayoutBuild::retired) {
            self.retired = None;
        }
    }

    pub(super) fn cull(
        &mut self,
        items: &ItemSlab,
        camera: Camera,
        frustum: &Frustum,
        meshes: &HashMap<MeshId, GpuMesh>,
        indirect_first_instance: bool,
        view: &mut ViewDraws,
    ) -> CullStats {
        self.cull_dirty = false;
        let blended_sorted = self.select(items, camera, frustum, view);
        let Self { groups, committed, .. } = self;
        for id in view.batch_order.drain(..) {
            if let Some(commands) = view.batch_commands.get_mut(id) {
                commands.clear();
            }
        }
        for index in &view.opaque {
            let Some(draw) = committed.draws.get(*index as usize) else { continue };
            let (binding, args) = ensure_batch(groups, draw, meshes);
            if binding.id >= view.batch_commands.len() {
                view.batch_commands.resize_with(binding.id + 1, Vec::new);
            }
            let Some(commands) = view.batch_commands.get_mut(binding.id) else { continue };
            if commands.is_empty() {
                view.batch_order.push(binding.id);
            }
            commands.push(args);
        }
        view.encoded.clear();
        view.first_instances.clear();
        view.batches.clear();
        view.triangles = 0;
        for id in &view.batch_order {
            let (Some(group), Some(commands)) = (groups.batch_group(*id), view.batch_commands.get(*id)) else { continue };
            let offset = view.encoded.len() as u64;
            for args in commands {
                view.first_instances.push(args.first_instance);
                view.triangles =
                    view.triangles.saturating_add(u64::from(args.index_count / 3).saturating_mul(u64::from(args.instance_count)));
                encode_draw(args, &mut view.encoded, indirect_first_instance);
            }
            view.batches.push(DrawBatch {
                id: *id,
                key: group.key,
                buffers: Arc::clone(&group.buffers),
                offset,
                count: commands.len() as u32,
            });
        }
        let blended_start = view.batches.len();
        for record in &view.blended {
            let Some(draw) = committed.draws.get(committed.opaque_draws.saturating_add(*record as usize)) else { continue };
            let (binding, args) = ensure_batch(groups, draw, meshes);
            let offset = view.encoded.len() as u64;
            view.first_instances.push(args.first_instance);
            view.triangles = view.triangles.saturating_add(u64::from(args.index_count / 3).saturating_mul(u64::from(args.instance_count)));
            encode_draw(&args, &mut view.encoded, indirect_first_instance);
            if view.batches.len() > blended_start
                && let Some(batch) = view.batches.last_mut()
                && batch.id == binding.id
            {
                batch.count += 1;
            } else if let Some(group) = groups.batch_group(binding.id) {
                view.batches.push(DrawBatch { id: binding.id, key: group.key, buffers: Arc::clone(&group.buffers), offset, count: 1 });
            }
        }
        let changed = view.encoded != view.indirect_bytes;
        if changed {
            mem::swap(&mut view.encoded, &mut view.indirect_bytes);
        }
        CullStats { commands: view.first_instances.len(), changed, blended_sorted }
    }

    fn select(&self, items: &ItemSlab, camera: Camera, frustum: &Frustum, view: &mut ViewDraws) -> bool {
        let state = &self.committed;
        let eye = camera.position();
        let draw_distance = camera.draw_distance();
        view.cells.clear();
        view.cells.extend(state.cells.iter().map(|cell| CellClass::new(*cell, eye, draw_distance, frustum)));
        view.opaque.clear();
        for (index, draw) in state.opaque().iter().enumerate() {
            if draw.visible(&view.cells, eye, draw_distance, frustum) {
                view.opaque.push(index as u32);
            }
        }
        if view.layout != self.layout {
            view.blended.clear();
            view.layout = self.layout;
        }
        view.mark += 2;
        let mark = view.mark;
        view.blended_marks.resize(state.blended.len(), 0);
        view.blended_distances.resize(state.blended.len(), 0.0);
        let blended_draws = state.draws.get(state.opaque_draws..).unwrap_or_default();
        for (((blended, draw), mark_slot), dist_slot) in
            state.blended.iter().zip(blended_draws).zip(&mut view.blended_marks).zip(&mut view.blended_distances)
        {
            if !draw.visible(&view.cells, eye, draw_distance, frustum) {
                continue;
            }
            let Some(item) = items.get(blended.item) else { continue };
            *mark_slot = mark;
            *dist_slot = blend_distance(item.position, item.rotation, item.scale, blended.center, camera);
        }
        view.blended_keys.clear();
        let push = |view: &mut ViewDraws, record: u32| {
            let index = record as usize;
            if let Some(mark_slot) = view.blended_marks.get_mut(index)
                && *mark_slot == mark
                && let Some(&distance) = view.blended_distances.get(index)
                && let Some(blended) = state.blended.get(index)
            {
                *mark_slot = mark + 1;
                view.blended_keys.push((distance, blended.order, record));
            }
        };
        let previous = mem::take(&mut view.blended);
        for record in previous.iter().copied() {
            push(view, record);
        }
        for record in 0..state.blended.len() {
            push(view, record as u32);
        }
        view.blended = previous;
        view.blended_keys.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        view.blended.clear();
        view.blended.extend(view.blended_keys.iter().map(|key| key.2));
        !view.blended.is_empty()
    }

    pub(super) fn slot_count(&self) -> usize {
        self.committed.instance_bytes.len() / INSTANCE_BYTES
    }

    pub(super) fn cull_dirty(&self) -> bool {
        self.cull_dirty
    }

    pub(super) fn patch(
        &mut self,
        items: &ItemSlab,
        faces: &[(ItemIndex, usize)],
        pose_items: &[ItemIndex],
        origin: DVec3,
        transform_origin: DVec3,
    ) -> usize {
        let mut patched = 0;
        for (item_id, face_index) in faces {
            let Some(item) = items.get(*item_id) else { continue };
            let Some(face) = item.faces.get(*face_index) else { continue };
            let sphere = if pose_items.binary_search(item_id).is_ok() {
                bounds_sphere(item.position, item.rotation, item.scale, item.info.bounds)
            } else {
                None
            };
            if let Some(build) = self.pending.as_mut()
                && !build.stale
                && face.layout == build.layout
                && let Some(slot) = face.slot
                && write_instance(&mut build.state.instance_bytes, slot, &item.transform, &face.parameter_bytes)
            {
                build.upload(slot);
                if let Some(sphere) = sphere {
                    build.state.include(slot, face.blended_index, sphere, item.shading);
                }
            }
            let Some(address) = self.committed_address(face) else { continue };
            let transform = if origin == transform_origin {
                item.transform
            } else {
                instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin)
            };
            if !write_instance(&mut self.committed.instance_bytes, address.slot, &transform, &face.parameter_bytes) {
                continue;
            }
            self.committed_written(address.slot);
            patched += 1;
            if let Some(sphere) = sphere {
                self.committed.include(address.slot, address.blended_index, sphere, item.shading);
                self.cull_dirty = true;
            }
        }
        patched
    }

    fn committed_written(&mut self, slot: u32) {
        self.committed.dirty_slots.push(slot);
        if let Some(build) = self.pending.as_mut()
            && (slot as usize) < build.state.instance_bytes.len() / INSTANCE_BYTES
        {
            build.upload(slot);
        }
    }

    pub(super) fn upload_layout(&mut self, queue: &Queue, buffer: &Buffer, full: bool) -> UploadStats {
        let mut stats = UploadStats::default();
        self.committed.dirty_slots.sort_unstable();
        self.committed.dirty_slots.dedup();
        let mut range_count = 0usize;
        let mut uploaded_bytes = 0usize;
        let mut first = None;
        let mut last = 0u32;
        for slot in self.committed.dirty_slots.iter().copied() {
            if first.is_none_or(|_| slot.saturating_sub(last) > MAX_PATCH_GAP_SLOTS) {
                if let Some(start) = first {
                    uploaded_bytes = uploaded_bytes.saturating_add((last as usize + 1 - start as usize) * INSTANCE_BYTES);
                }
                range_count += 1;
                first = Some(slot);
            }
            last = slot;
        }
        if let Some(start) = first {
            uploaded_bytes = uploaded_bytes.saturating_add((last as usize + 1 - start as usize) * INSTANCE_BYTES);
        }
        if full || range_count > MAX_LAYOUT_UPLOAD_RANGES || uploaded_bytes > self.committed.instance_bytes.len() / 2 {
            if !self.committed.instance_bytes.is_empty() {
                queue.write_buffer(buffer, 0, &self.committed.instance_bytes);
                stats.bytes = self.committed.instance_bytes.len();
                stats.calls = 1;
            }
        } else {
            first = None;
            for slot in self.committed.dirty_slots.iter().copied() {
                if let Some(start) = first
                    && slot.saturating_sub(last) > MAX_PATCH_GAP_SLOTS
                {
                    let uploaded = self.write_slot_range(queue, buffer, start, last);
                    stats.bytes += uploaded.bytes;
                    stats.calls += uploaded.calls;
                    first = None;
                }
                if first.is_none() {
                    first = Some(slot);
                }
                last = slot;
            }
            if let Some(start) = first {
                let uploaded = self.write_slot_range(queue, buffer, start, last);
                stats.bytes += uploaded.bytes;
                stats.calls += uploaded.calls;
            }
        }
        self.committed.dirty_slots.clear();
        stats
    }

    pub(super) fn upload_patch(&mut self, queue: &Queue, buffer: &Buffer) -> UploadStats {
        let mut stats = UploadStats::default();
        self.committed.dirty_slots.sort_unstable();
        self.committed.dirty_slots.dedup();
        let mut range_count = 0;
        let mut previous = None;
        for slot in self.committed.dirty_slots.iter().copied() {
            if previous.is_none_or(|last: u32| slot.saturating_sub(last) > MAX_PATCH_GAP_SLOTS) {
                range_count += 1;
            }
            previous = Some(slot);
        }
        if range_count > MAX_PATCH_RANGES {
            if !self.committed.instance_bytes.is_empty() {
                queue.write_buffer(buffer, 0, &self.committed.instance_bytes);
                stats.bytes = self.committed.instance_bytes.len();
                stats.calls = 1;
            }
            self.committed.dirty_slots.clear();
            return stats;
        }
        let mut first = None;
        let mut last = 0u32;
        for slot in self.committed.dirty_slots.iter().copied() {
            if let Some(start) = first
                && slot.saturating_sub(last) > MAX_PATCH_GAP_SLOTS
            {
                let uploaded = self.write_slot_range(queue, buffer, start, last);
                stats.bytes += uploaded.bytes;
                stats.calls += uploaded.calls;
                first = None;
            }
            if first.is_none() {
                first = Some(slot);
            }
            last = slot;
        }
        if let Some(start) = first {
            let uploaded = self.write_slot_range(queue, buffer, start, last);
            stats.bytes += uploaded.bytes;
            stats.calls += uploaded.calls;
        }
        self.committed.dirty_slots.clear();
        stats
    }

    fn write_slot_range(&self, queue: &Queue, buffer: &Buffer, first: u32, last: u32) -> UploadStats {
        let start = first as usize * INSTANCE_BYTES;
        let end = (last as usize + 1).saturating_mul(INSTANCE_BYTES);
        if let Some(bytes) = self.committed.instance_bytes.get(start..end) {
            queue.write_buffer(buffer, start as u64, bytes);
            return UploadStats { bytes: bytes.len(), calls: 1 };
        }
        UploadStats::default()
    }

    pub(super) fn intern_face(
        &mut self,
        key: DrawKey,
        images: &MaterialImages,
        terrain_textures: Option<&[Option<TextureHandle>; 4]>,
    ) -> Option<u32> {
        self.groups.intern(key, images, terrain_textures)
    }

    #[cfg(test)]
    pub(super) fn face_key(&self, group: u32) -> Option<DrawKey> {
        self.groups.key(group)
    }

    pub(super) fn refresh_winding(
        &mut self,
        faces: &mut [CachedFace],
        old_scale: Vec3,
        old_invert_normals: bool,
        scale: Vec3,
        invert_normals: bool,
    ) -> bool {
        if (math::flips_winding(old_scale) ^ old_invert_normals) == (math::flips_winding(scale) ^ invert_normals) {
            return false;
        }
        let mut changed = false;
        for face in faces {
            let Some(mut key) = self.groups.key(face.group) else { continue };
            let previous_pipeline = key.pipeline;
            key.pipeline = surface_pipeline(&face.base_params, scale, invert_normals);
            if key.pipeline == previous_pipeline {
                continue;
            }
            if let Some(group) = self.groups.intern(key, &face.images, None) {
                self.groups.release(face.group);
                face.group = group;
                changed = true;
            }
        }
        changed
    }

    pub(super) fn release_faces(&mut self, faces: &[CachedFace]) {
        for face in faces {
            self.groups.release(face.group);
        }
    }

    pub(super) fn forget_item(&mut self, index: ItemIndex) {
        if let Some(build) = self.pending.as_mut() {
            build.removed.insert(index);
        }
    }

    pub(super) fn hide_item(&mut self, index: ItemIndex, item: &mut CachedItem) {
        self.forget_item(index);
        let mut hidden = false;
        for face in &mut item.faces {
            hidden |= self.clear_face_slot(face);
        }
        if hidden {
            debug_assert!(self.committed.visible_items > 0);
            self.committed.visible_items -= 1;
        }
    }

    pub(super) fn hide_face(&mut self, face: &mut CachedFace) -> bool {
        if let Some(build) = self.pending.as_mut()
            && face.layout == build.layout
        {
            build.stale = true;
        }
        self.clear_face_slot(face)
    }

    fn clear_face_slot(&mut self, face: &mut CachedFace) -> bool {
        if let Some(build) = self.pending.as_mut()
            && !build.stale
            && face.layout == build.layout
            && let Some(slot) = face.slot
        {
            let start = slot as usize * INSTANCE_BYTES;
            if let Some(instance) = build.state.instance_bytes.get_mut(start..start + INSTANCE_BYTES) {
                instance.fill(0);
                build.upload(slot);
            }
            if let Some(blended_index) = face.blended_index
                && let Some(draw) = build.state.draws.get_mut(build.state.opaque_draws.saturating_add(blended_index))
            {
                draw.live = false;
            }
        }
        let committed = self.committed_address(face);
        face.slot = None;
        face.blended_index = None;
        face.committed = None;
        let Some(FaceSlot { slot, .. }) = committed else { return false };
        let start = slot as usize * INSTANCE_BYTES;
        if let Some(instance) = self.committed.instance_bytes.get_mut(start..start + INSTANCE_BYTES) {
            instance.fill(0);
            self.committed_written(slot);
        }
        true
    }

    fn release_draw_group_refs(&mut self) {
        for group in self.committed.draw_group_refs.drain(..) {
            self.groups.release(group);
        }
        self.committed.draw_group_seen.clear();
    }

    pub(super) fn release_group(&mut self, group: u32) {
        self.groups.release(group);
    }

    pub(super) fn update_terrain(&mut self, region: Region, textures: &[Option<TextureHandle>; 4]) {
        self.groups.update_terrain(region, textures);
        for demand in self.committed.active_terrain.iter_mut().filter(|demand| demand.region == region) {
            demand.textures.clone_from(textures);
        }
    }

    pub(super) fn invalidate_batches(&mut self) {
        self.groups.invalidate_batches();
        self.cull_dirty = true;
    }

    pub(super) fn texture_demands(&self) -> &[TextureDemand] {
        &self.committed.active_textures
    }

    pub(super) fn texture_keys(&self) -> &HashSet<MaterialTextures> {
        &self.committed.texture_seen
    }

    pub(super) fn terrain_demands(&self) -> &[TerrainDemand] {
        &self.committed.active_terrain
    }

    pub(super) fn terrain_regions(&self) -> &HashSet<Region> {
        &self.committed.terrain_seen
    }

    #[cfg(test)]
    pub(super) fn has_blended(&self) -> bool {
        !self.committed.blended.is_empty()
    }

    #[cfg(test)]
    pub(super) fn draws(&self) -> &[DrawCommand] {
        &self.committed.draws
    }

    #[cfg(test)]
    pub(super) fn instance_transform(&self, slot: u32) -> Option<&[u8]> {
        let start = usize::try_from(slot).ok()?.checked_mul(INSTANCE_BYTES)?;
        self.committed.instance_bytes.get(start..start.checked_add(TRANSFORM_BYTES)?)
    }

    pub(super) fn visible_items(&self) -> usize {
        self.committed.visible_items
    }

    pub(super) fn set_visible_items(&mut self, count: usize) {
        self.committed.visible_items = count;
    }

    pub(super) fn has_dirty_slots(&self) -> bool {
        !self.committed.dirty_slots.is_empty()
    }

    pub(super) fn valid(&self) -> bool {
        self.committed.valid
    }

    pub(super) fn clear(&mut self) {
        self.cancel_layout();
        self.release_draw_group_refs();
        self.committed = LayoutState::default();
        self.cull_dirty = true;
        self.layout = self.next_layout();
    }
}

fn assign_slot(face: &mut CachedFace, layout: LayoutId, committed: LayoutId, slot: Option<u32>, blended_index: Option<usize>) {
    if face.layout != layout {
        if face.layout == committed {
            face.committed = face.slot.map(|slot| FaceSlot { layout: committed, slot, blended_index: face.blended_index });
        }
        face.layout = layout;
    }
    face.slot = slot;
    face.blended_index = blended_index;
}

fn write_instance(bytes: &mut [u8], slot: u32, transform: &[u8; TRANSFORM_BYTES], parameters: &[u8; PARAMETER_BYTES]) -> bool {
    let start = slot as usize * INSTANCE_BYTES;
    let Some(destination) = bytes.get_mut(start..start + INSTANCE_BYTES) else { return false };
    let (transform_destination, parameter_destination) = destination.split_at_mut(TRANSFORM_BYTES);
    transform_destination.copy_from_slice(transform);
    parameter_destination.copy_from_slice(parameters);
    true
}

fn write_layout_instance(
    active: &[u8],
    staging: &mut [u8],
    dirty_slots: &mut Vec<u32>,
    slot: u32,
    transform: &[u8; TRANSFORM_BYTES],
    parameters: &[u8; PARAMETER_BYTES],
) -> bool {
    let start = slot as usize * INSTANCE_BYTES;
    let previous = active.get(start..start + INSTANCE_BYTES);
    if !write_instance(staging, slot, transform, parameters) {
        return false;
    }
    let current = staging.get(start..start + INSTANCE_BYTES);
    if previous.zip(current).is_none_or(|(previous, current)| previous != current) {
        dirty_slots.push(slot);
    }
    true
}

fn blend_distance(position: DVec3, rotation: Quat, scale: Vec3, center: Vec3, camera: Camera) -> f64 {
    let relative = position - camera.position() + (rotation * (center * scale)).as_dvec3();
    relative.length_squared()
}

fn ensure_batch(groups: &mut DrawInterner, draw: &DrawCommand, meshes: &HashMap<MeshId, GpuMesh>) -> (BatchBinding, DrawIndexedArgs) {
    let binding = if let Some(binding) = groups.batch(draw.group) {
        binding
    } else {
        let mesh = meshes.get(&draw.mesh).expect("draw command mesh is resident");
        let allocation = &mesh.allocation;
        let buffers = allocation.buffers();
        let key =
            DrawBatchKey { pipeline: draw.pipeline, textures: draw.textures, terrain: draw.terrain, page: Arc::as_ptr(buffers) as usize };
        let binding = BatchBinding {
            id: groups.intern_batch(key, buffers),
            index_offset: allocation.indices().start,
            base_vertex: i32::try_from(allocation.vertices().start).expect("mesh page fits the GPU budget"),
        };
        groups.set_batch(draw.group, binding);
        binding
    };
    let args = DrawIndexedArgs {
        index_count: draw.indices.end - draw.indices.start,
        instance_count: draw.count,
        first_index: binding.index_offset.checked_add(draw.indices.start).expect("mesh page index range fits the GPU budget"),
        base_vertex: binding.base_vertex,
        first_instance: draw.first,
    };
    (binding, args)
}

fn encode_draw(args: &DrawIndexedArgs, bytes: &mut Vec<u8>, indirect_first_instance: bool) {
    let start = bytes.len();
    bytes.extend_from_slice(&args.index_count.to_le_bytes());
    bytes.extend_from_slice(&args.instance_count.to_le_bytes());
    bytes.extend_from_slice(&args.first_index.to_le_bytes());
    bytes.extend_from_slice(&args.base_vertex.to_le_bytes());
    bytes.extend_from_slice(&(if indirect_first_instance { args.first_instance } else { 0 }).to_le_bytes());
    debug_assert_eq!(bytes.len() - start, INDIRECT_DRAW_BYTES);
}

#[cfg(test)]
#[path = "draw_list_tests.rs"]
mod tests;
