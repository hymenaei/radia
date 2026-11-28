use std::{
    assert_matches,
    collections::HashSet,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use math::{DVec3, Quat, Vec3};
use uuid::Uuid;
use wgpu::{CommandEncoderDescriptor, Device, DeviceDescriptor, TextureFormat};
use world::{
    Appearance, FaceAppearance, FieldUpdate, Kind, ObjectObservation, OrderEvidence, ParentUpdate, Pose, PrimShape, Region, RegionInfo,
    Shape, SurfaceAppearance, UpdateSource, World, WorldLimits,
};

use super::{CacheRebuild, CacheUpdate, FrameCache, ItemId, MeshInfoBuild};
use crate::{
    Limits,
    geometry::{
        GeometryRenderer, ShadingMode, fixtures,
        instance::instance_base,
        items::{CachedItem, CachedReplacement, mesh_info},
    },
    scene::{Budget, Camera, RANGE_MARGIN, Reconciler, SceneView, SharedMesh},
    test_support,
};

#[test]
fn item_order_tracks_lifecycle() {
    let (_region_world, region) = test_support::registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let mut cache = FrameCache::default();
    let high = (2, 9, 0);
    let middle = (1, 9, 0);
    let low = (0, 9, 0);
    let high_index = cache.insert_item(item(high, &mesh, region)).expect("high item index");
    let middle_index = cache.insert_item(item(middle, &mesh, region)).expect("middle item index");
    let low_index = cache.insert_item(item(low, &mesh, region)).expect("low item index");

    assert_eq!(ordered_ids(&mut cache), vec![low, middle, high]);

    assert_eq!(cache.insert_item(item(high, &mesh, region)), Some(high_index));
    assert_eq!(ordered_ids(&mut cache), vec![low, middle, high]);

    assert!(cache.remove_item(middle_index).is_some());
    let earlier = (0, 1, 0);
    assert_eq!(cache.insert_item(item(earlier, &mesh, region)), Some(middle_index));
    assert_eq!(ordered_ids(&mut cache), vec![earlier, low, high]);

    cache.items.get_mut(high_index).expect("high item").replacement = Some(Box::new(CachedReplacement { item: item(high, &mesh, region) }));
    cache.replacements.insert(high_index);
    assert_eq!(cache.promote_replacement(high_index).expect("promoted item").id, high);
    assert!(cache.items.get(high_index).is_some_and(|item| item.replacement.is_none()));
    assert!(!cache.replacements.contains(&high_index));
    assert_eq!(ordered_ids(&mut cache), vec![earlier, low, high]);

    cache.clear_items();
    assert!(ordered_ids(&mut cache).is_empty());
    cache.insert_item(item(high, &mesh, region)).expect("recovered high item index");
    cache.insert_item(item(low, &mesh, region)).expect("recovered low item index");
    assert_eq!(ordered_ids(&mut cache), vec![low, high]);

    assert!(cache.items.get(low_index).is_none());
}

#[test]
fn camera_recovery_progresses() {
    let (world, _scene, view) = fixtures::object_scene(1_100);
    let region = world.region_page(None, 1).expect("fixture region page").regions[0].region;
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let mut cache = FrameCache::default();
    for object in view.objects() {
        let summary = &object.detail().summary;
        let mut cached = item((2, summary.id.as_u128(), 0), &mesh, region);
        cached.entity = Some(summary.entity);
        let index = cache.insert_item(cached).expect("cached object index");
        cache.objects.insert(summary.entity, index);
    }
    let camera = |turn: u32| {
        Camera::new(DVec3::new(0.0, 0.0, f64::from(turn) * RANGE_MARGIN), Vec3::X, 1.0, [128, 128]).expect("moving fixture camera")
    };
    let mut rebuild = CacheRebuild::new(&view, cache.slot_count(), false);
    rebuild.object_change_page_complete = true;
    rebuild.added_object_page_complete = true;
    rebuild.note_camera(camera(0));
    rebuild.note_camera(camera(1));
    rebuild.finish_mesh_info();
    cache.rebuild = Some(rebuild);

    let mut visited = HashSet::new();
    let mut pass_cameras = Vec::new();
    let mut erased_cursors = 0;
    let mut completed = None;
    for turn in 2..1_000 {
        let rebuild = cache.rebuild.as_mut().expect("recovery is pending until it completes");
        let cursor = rebuild.visibility_object_after;
        pass_cameras.push(rebuild.note_camera(camera(turn)));
        erased_cursors += usize::from(cursor.is_some() && rebuild.visibility_object_after.is_none());
        let mut update = CacheUpdate::default();
        let mut items = 0;
        while cache.advance_visibility(&view, view.sequence(), Instant::now(), &mut items, &mut update) {}
        visited.extend(update.visibility_items);
        if update.rebuild_completed {
            completed = Some(turn);
            break;
        }
    }

    assert!(
        completed.is_some(),
        "visibility recovery never completed while the camera moved every turn: {:?}",
        cache.rebuild.as_ref().map(CacheRebuild::progress)
    );
    assert_eq!(erased_cursors, 0, "camera updates must not erase the in-progress pass cursor");
    assert_eq!(visited.len(), cache.objects.len(), "the pass revisits every cached object");
    assert!(pass_cameras.iter().all(|pass| *pass == camera(1)), "the whole pass evaluates its snapshot camera");
}

#[test]
fn info_resumes_within_limit() {
    let (_scene, view, _) = fixtures::scene(255);
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let indices = mesh.mesh().indices().len();
    let bytes_per_index = size_of::<u32>() + size_of::<Vec3>();
    let limit = 3 * bytes_per_index;
    let mut build = MeshInfoBuild::new(mesh.clone());
    let mut total = 0;
    let mut steps = 0;
    while !build.is_complete() {
        let (bytes, _) = build.advance(limit, usize::MAX, Duration::MAX);
        assert!(bytes <= limit, "one step stays within its byte limit: {bytes}");
        total += bytes;
        steps += 1;
        assert!(steps <= indices + mesh.mesh().faces().len(), "every step makes progress");
    }
    assert!(steps > mesh.mesh().faces().len(), "faces larger than the limit resume across steps");
    assert_eq!(total, indices * bytes_per_index, "each index is processed exactly once");
    assert_eq!(build.centers, mesh_info(&mesh).centers, "a resumed build matches the one-shot build");
}

#[test]
fn rebuild_finalizes_under_object_churn() {
    let (device, queue) = Device::noop(&DeviceDescriptor::default());
    let mut geometry = GeometryRenderer::new(&device, &queue, TextureFormat::Rgba8UnormSrgb, Limits::default());
    let (mut world, mut scene, mut view) = fixtures::object_scene(1_100);
    let region = world.region_page(None, 1).expect("fixture region page").regions[0].region;

    let initial = geometry.update_cache_for_frame(&view, false);
    assert!(initial.rebuild_started, "first frame starts a geometry rebuild");
    assert!(!initial.rebuild_completed, "1,100 objects require more than one rebuild frame");

    let mut completed = None;
    for turn in 1..1_000_u32 {
        let sequence_before = view.sequence();
        let order = 1_100 + u64::from(turn);
        let mut observation = ObjectObservation::new(
            UpdateSource::new(region, 1_100 + turn, OrderEvidence::Ordered(order), 0),
            Uuid::from_u128(1),
            NonZeroU32::new(1).expect("object local id"),
            Kind::Object,
            ParentUpdate::Root,
        );
        observation.pose = Some(Pose { position: Some(Vec3::new(10.0, turn as f32 * 0.01, 0.0)), rotation: Some(Quat::IDENTITY) });
        observation.scale = Some(Vec3::ONE);
        observation.shape = FieldUpdate::Set(Shape::Prim(PrimShape::default()));
        observation.appearance = FieldUpdate::Set(Appearance::Surface);
        observation.surface_appearance = Some(SurfaceAppearance::new(FaceAppearance::default(), Vec::new()).expect("fixture appearance"));
        let _ = world.observe_object(observation).expect("object update applies");
        scene.notify_world_change();
        fixtures::settle(&world, &mut scene, &mut view);
        assert!(view.sequence() > sequence_before, "each turn advances the scene sequence");

        let update = geometry.update_cache_for_frame(&view, false);
        if update.rebuild_completed {
            completed = Some(turn);
            break;
        }
    }

    let completed = completed.unwrap_or_else(|| {
        panic!(
            "geometry rebuild never finalized while objects kept changing: {:?}",
            geometry.cache.rebuild.as_ref().map(CacheRebuild::progress)
        )
    });
    assert!(geometry.cache.rebuild.is_none(), "completed rebuild clears active state");

    let target_entity = view
        .objects()
        .find(|object| object.detail().summary.id == Uuid::from_u128(1))
        .expect("updated object exists")
        .detail()
        .summary
        .entity;
    let cached_index = geometry.cache.objects.get(&target_entity).copied().expect("updated object is cached");
    assert_eq!(
        geometry.cache.items.get(cached_index).expect("cached item exists").position,
        DVec3::new(10.0, f64::from(completed as f32 * 0.01), 0.0),
        "the change from the finalizing turn is reflected in the cache"
    );

    let follow_up_turn = completed + 1;
    let follow_up_y = follow_up_turn as f32 * 0.01;
    let mut follow_up = ObjectObservation::new(
        UpdateSource::new(region, 1_100 + follow_up_turn, OrderEvidence::Ordered(1_100 + u64::from(follow_up_turn)), 0),
        Uuid::from_u128(1),
        NonZeroU32::new(1).expect("object local id"),
        Kind::Object,
        ParentUpdate::Root,
    );
    follow_up.pose = Some(Pose { position: Some(Vec3::new(10.0, follow_up_y, 0.0)), rotation: Some(Quat::IDENTITY) });
    follow_up.scale = Some(Vec3::ONE);
    follow_up.shape = FieldUpdate::Set(Shape::Prim(PrimShape::default()));
    follow_up.appearance = FieldUpdate::Set(Appearance::Surface);
    follow_up.surface_appearance = Some(SurfaceAppearance::new(FaceAppearance::default(), Vec::new()).expect("fixture appearance"));
    let _ = world.observe_object(follow_up).expect("post-finalize update applies");
    scene.notify_world_change();
    fixtures::settle(&world, &mut scene, &mut view);

    let incremental = geometry.update_cache_for_frame(&view, false);
    assert!(!incremental.rebuild_started && !incremental.rebuild_pending, "post-finalize update uses the incremental path");
    let cached_index = geometry.cache.objects.get(&target_entity).copied().expect("updated object remains cached");
    assert_eq!(
        geometry.cache.items.get(cached_index).expect("cached item exists").position,
        DVec3::new(10.0, f64::from(follow_up_y), 0.0),
        "post-finalize object change is reflected in the cache without losing updates"
    );
}

#[test]
fn rebuild_finalizes_under_visibility_churn() {
    let (device, queue) = Device::noop(&DeviceDescriptor::default());
    let mut geometry = GeometryRenderer::new(&device, &queue, TextureFormat::Rgba8UnormSrgb, Limits::default());
    let count = 1_100_u32;
    let (mut world, region, mut scene, mut view) = removable_object_scene(count, 1_024);

    let initial = geometry.update_cache_for_frame(&view, false);
    assert!(initial.rebuild_started, "first frame starts a geometry rebuild");
    assert!(!initial.rebuild_completed, "1,100 objects require more than one rebuild frame");

    let mut completed = None;
    for turn in 1..1_000_u32 {
        let visibility_before = view.visibility_version();
        let order = u64::from(count) + u64::from(turn);
        let _ = world
            .remove_object(
                UpdateSource::new(region, count + turn, OrderEvidence::Ordered(order), 0),
                NonZeroU32::new(turn).expect("removed local id"),
            )
            .expect("object removal applies");
        scene.notify_world_change();
        fixtures::settle(&world, &mut scene, &mut view);
        assert_ne!(view.visibility_version(), visibility_before, "each removal advances the visibility version");

        let update = geometry.update_cache_for_frame(&view, false);
        if update.rebuild_completed {
            completed = Some(turn);
            break;
        }
    }

    let removed = completed.unwrap_or_else(|| {
        panic!(
            "geometry rebuild never finalized while visibility kept changing: {:?}",
            geometry.cache.rebuild.as_ref().map(CacheRebuild::progress)
        )
    });
    let settled = geometry.update_cache_for_frame(&view, false);
    assert!(!settled.rebuild_pending, "a quiet frame after the rebuild starts no new work");
    let live = usize::try_from(count - removed).expect("live count fits usize");
    assert_eq!(geometry.cache.objects.len(), live, "every removed object leaves the cache once the scene settles");
}

#[test]
fn layout_commits_under_removal_churn() {
    let (device, queue) = Device::noop(&DeviceDescriptor::default());
    let mut geometry = GeometryRenderer::new(&device, &queue, TextureFormat::Rgba8UnormSrgb, Limits::default());
    let count = 1_500_u32;
    let (mut world, region, mut scene, mut view) = removable_object_scene(count, 1_024);
    let camera = Camera::new(DVec3::ZERO, Vec3::X, 1.0, [128, 128]).expect("fixture camera");
    let frame = |geometry: &mut GeometryRenderer, view: &SceneView| {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        geometry.prepare(&device, &mut encoder, view, None, camera, [128, 128], None);
    };

    let expected = usize::try_from(count).expect("count fits usize");
    for _ in 0..10_000 {
        frame(&mut geometry, &view);
        if geometry.cache.visible.len() == expected
            && geometry.draw_list.valid()
            && !geometry.stats.layout_pending
            && !geometry.stats.rebuild_active
            && !geometry.stats.rebuild_pending
            && geometry.stats.pending_meshes == 0
            && !geometry.layout_required
        {
            break;
        }
    }
    assert!(geometry.draw_list.valid(), "initial scene settles to a valid layout");
    assert!(!geometry.stats.layout_pending, "initial scene finishes its layout");
    assert_eq!(geometry.cache.visible.len(), expected, "every fixture object becomes visible");

    let committed_before = geometry.layout_counts.committed;
    let entering_id = count + 1;
    let mut observation = ObjectObservation::new(
        UpdateSource::new(region, entering_id, OrderEvidence::Ordered(u64::from(entering_id)), 0),
        Uuid::from_u128(u128::from(entering_id)),
        NonZeroU32::new(entering_id).expect("entering local id"),
        Kind::Object,
        ParentUpdate::Root,
    );
    observation.pose = Some(Pose { position: Some(Vec3::new(10.0, 0.0, 0.0)), rotation: Some(Quat::IDENTITY) });
    observation.scale = Some(Vec3::ONE);
    observation.shape = FieldUpdate::Set(Shape::Prim(PrimShape::default()));
    observation.appearance = FieldUpdate::Set(Appearance::Surface);
    observation.surface_appearance = Some(SurfaceAppearance::new(FaceAppearance::default(), Vec::new()).expect("fixture appearance"));
    let _ = world.observe_object(observation).expect("entering object applies");
    scene.notify_world_change();
    fixtures::settle(&world, &mut scene, &mut view);

    frame(&mut geometry, &view);
    assert!(geometry.stats.layout_pending, "entering item starts a multi-frame layout");

    let mut removed_meshes = Vec::new();
    for turn in 1..=1_000_u32 {
        if !geometry.stats.layout_pending {
            break;
        }
        let target_uuid = Uuid::from_u128(u128::from(turn));
        let target = view.objects().find(|object| object.detail().summary.id == target_uuid).expect("visible object exists");
        let entity = target.detail().summary.entity;
        let mesh_id = target.mesh().expect("object has mesh").1.id();
        let cached_index = geometry.cache.objects.get(&entity).copied().expect("object is cached");
        assert!(geometry.cache.visible.contains(&cached_index), "removed object is currently visible");
        removed_meshes.push(mesh_id);

        let order = u64::from(entering_id) + u64::from(turn);
        let _ = world
            .remove_object(
                UpdateSource::new(region, entering_id + turn, OrderEvidence::Ordered(order), 0),
                NonZeroU32::new(turn).expect("removed local id"),
            )
            .expect("object removal applies");
        scene.notify_world_change();
        fixtures::settle(&world, &mut scene, &mut view);

        frame(&mut geometry, &view);
        if geometry.layout_counts.committed > committed_before {
            break;
        }
    }

    assert!(!removed_meshes.is_empty(), "at least one visible object was removed while layout was pending");
    assert!(
        geometry.layout_counts.committed > committed_before,
        "layout must commit under removal churn (committed: {}, stale: {}, pending: {})",
        geometry.layout_counts.committed,
        geometry.layout_counts.stale,
        geometry.stats.layout_pending
    );
    assert_eq!(geometry.layout_counts.stale, 0, "removals must not stale a pending layout");
    for mesh_id in removed_meshes {
        for draw in geometry.draw_list.draws().iter().filter(|draw| draw.mesh == mesh_id) {
            for slot in draw.first..draw.first + draw.count {
                let transform = geometry.draw_list.instance_transform(slot).expect("committed draw slot has transform bytes");
                assert!(transform.iter().all(|byte| *byte == 0), "removed object mesh {mesh_id:?} still draws at slot {slot}");
            }
        }
    }
}

fn removable_object_scene(count: u32, tombstones: usize) -> (World, Region, Reconciler, SceneView) {
    let capacity = usize::try_from(count).expect("object count fits usize").next_power_of_two();
    let mut world = World::new(WorldLimits {
        max_regions: 1,
        max_entities: capacity,
        max_entities_per_region: capacity,
        max_tombstones_per_region: tombstones,
        max_tombstones: tombstones,
        max_tombstone_bytes_per_region: WorldLimits::tombstone_bytes_for_count(tombstones),
        max_tombstone_bytes: WorldLimits::tombstone_bytes_for_count(tombstones),
        max_journal_entries: 16,
        max_journal_changes: 64,
        max_region_name_bytes: 64,
        max_recovery_targets: capacity,
    });
    let region = world.register_region().expect("fixture region registers");
    let _ = world.patch_region(region, RegionInfo { origin: Some(DVec3::ZERO), ..Default::default() }).expect("fixture origin applies");
    for id in 1..=count {
        let mut observation = ObjectObservation::new(
            UpdateSource::new(region, id, OrderEvidence::Ordered(u64::from(id)), 0),
            Uuid::from_u128(u128::from(id)),
            NonZeroU32::new(id).expect("fixture local id"),
            Kind::Object,
            ParentUpdate::Root,
        );
        observation.pose = Some(Pose { position: Some(Vec3::new(10.0, 0.0, 0.0)), rotation: Some(Quat::IDENTITY) });
        observation.scale = Some(Vec3::ONE);
        observation.shape = FieldUpdate::Set(Shape::Prim(PrimShape {
            path_scale_x: u8::try_from(100 + (id % 100)).expect("scale x fits u8"),
            path_scale_y: u8::try_from(100 + ((id / 100) % 100)).expect("scale y fits u8"),
            ..PrimShape::default()
        }));
        observation.appearance = FieldUpdate::Set(Appearance::Surface);
        observation.surface_appearance = Some(SurfaceAppearance::new(FaceAppearance::default(), Vec::new()).expect("fixture appearance"));
        assert_matches!(world.observe_object(observation).expect("fixture object applies").outcome, world::ApplyOutcome::Introduced(_));
    }
    let (mut scene, mut view) = Reconciler::new(Budget::default());
    scene.set_camera(Camera::new(DVec3::ZERO, Vec3::X, 1.0, [128, 128]).expect("fixture camera"));
    fixtures::settle(&world, &mut scene, &mut view);
    assert_eq!(view.object_count(), usize::try_from(count).expect("object count fits usize"));
    (world, region, scene, view)
}

fn ordered_ids(cache: &mut FrameCache) -> Vec<ItemId> {
    cache.rebuild_item_order();
    cache.item_order.iter().map(|index| cache.items.get(*index).expect("ordered item exists").id).collect()
}

fn item(id: ItemId, mesh: &SharedMesh, region: Region) -> CachedItem {
    let position = DVec3::ZERO;
    let rotation = Quat::IDENTITY;
    let scale = Vec3::ONE;
    CachedItem {
        id,
        entity: None,
        region,
        mesh: mesh.clone(),
        info: mesh_info(mesh),
        position,
        rotation,
        scale,
        invert_normals: false,
        replacement: None,
        transform: instance_base(position, rotation, scale, false, DVec3::ZERO),
        pickable: true,
        shading: ShadingMode::Surface,
        terrain_textures: None,
        terrain_composition: None,
        surface_kind: None,
        source: None,
        faces: Vec::new(),
        animated_faces: Vec::new(),
        animation_deadline: None,
        parameters_dirty: false,
    }
}
