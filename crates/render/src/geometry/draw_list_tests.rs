use std::{
    assert_matches, cmp,
    collections::{HashMap, HashSet},
    f64::consts::TAU,
    iter,
    sync::Arc,
    time::{Duration, Instant},
};

use assets::{Budget, Content, LoadJob, Priority, Request, Resource, Sampler, Status, Store};
use math::{DVec3, Frustum, Mat4, Quat, Vec3};
use proptest::{collection, prelude::*, test_runner::RngSeed};
use test_support::texture::GRAY_ALPHA_J2C;
use utils::AssetId;
use uuid::Uuid;
use world::Region;

use super::{DrawKey, DrawList, LayoutStats, ViewDraws};
use crate::{
    geometry::{
        ShadingMode, TRANSFORM_BYTES,
        cache::FrameCache,
        fixtures,
        instance::{PARAMETER_BYTES, instance_base, parameter_part, scene_origin},
        items::{CachedFace, CachedItem, CachedReplacement, ItemId, ItemIndex, ItemSlab, MeshInfo},
        texture_bindings::MaterialImages,
        visibility::WorldSphere,
    },
    material::FaceParams,
    pipeline,
    pipeline::{INSTANCE_BYTES, Pipeline},
    scene,
    scene::{Camera, SceneView, SharedMesh},
    test_support::registered_region,
    texture::TextureKey,
};

type OpaqueOutput = HashMap<(DrawKey, Vec<u8>), usize>;
type BlendedOutput = Vec<(DrawKey, Vec<u8>)>;

#[test]
fn canceled_layout_retires_in_chunks() {
    let (_region_world, _view, camera, mut items, mut list, index) = active_layout();
    let group = items.items.get(index).expect("active item").faces[0].group;
    let active_refs = group_refs(&list, group);
    let active_commands = list.draws().len();
    let active_items = list.visible_items();
    let active_bytes = list.committed.instance_bytes.len();
    let staged = staged_layout(&mut list, index, camera, &items.items, STAGED_REFS);
    assert_eq!(group_refs(&list, group), active_refs + STAGED_REFS);
    list.pending = Some(staged);
    let initial_entries = list.pending.as_ref().map(staged_entry_count).expect("pending staged build");
    let initial_allocation = list.pending.as_ref().map(staged_allocation_bytes).expect("pending staged build");
    list.cancel_layout();

    assert!(list.layout_pending());
    assert!(list.pending.is_none());
    assert_eq!(staged_entry_count(list.retired.as_ref().expect("full staged build is retained")), initial_entries);
    for _ in 0..8 {
        assert!(!list.request_layout(&[index], camera));
        list.cancel_layout();
        assert!(list.pending.is_none());
        assert!(list.retired.is_some(), "repeated requests cannot add retired builds");
        assert_eq!(staged_allocation_bytes(list.retired.as_ref().expect("single retired build")), initial_allocation);
    }

    let mut steps = 0;
    while list.layout_pending() {
        let before = list.retired_entries_remaining();
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
        assert!(before.saturating_sub(list.retired_entries_remaining()) <= super::LAYOUT_CLOCK_STEPS);
        assert_eq!(list.draws().len(), active_commands, "retirement leaves active draw commands intact");
        assert_eq!(list.visible_items(), active_items, "retirement leaves active visibility intact");
        assert_eq!(list.committed.instance_bytes.len(), active_bytes, "retirement leaves active instance data intact");
        steps += 1;
        assert!(steps < 100, "staged group references are released in bounded turns");
    }
    assert!(steps > 1, "many staged group references require multiple retirement turns");
    assert!(list.retired.is_none(), "completed retirement releases the whole staged allocation");
    assert_eq!(group_refs(&list, group), active_refs, "retirement releases every staged group reference");
    assert!(list.request_layout(&[index], camera));
}

#[test]
fn stale_layout_retires_whole_build() {
    let (_region_world, _view, camera, mut items, mut list, index) = active_layout();
    let group = items.items.get(index).expect("active item").faces[0].group;
    let active_refs = group_refs(&list, group);
    let active_commands = list.draws().len();
    let active_items = list.visible_items();
    let staged = staged_layout(&mut list, index, camera, &items.items, STAGED_REFS);
    list.pending = Some(staged);
    list.mark_layout_stale();

    let progress = list.advance_layout(&mut items.items, Duration::ZERO).expect("stale build is observed");
    assert!(progress.stale);
    assert!(list.pending.is_none());
    assert!(list.retired.is_some(), "stale staged inputs remain owned by incremental retirement");
    assert_eq!(list.draws().len(), active_commands, "stale build cannot replace active commands");
    assert_eq!(list.visible_items(), active_items, "stale build cannot replace active visibility");
    assert!(list.retired_entries_remaining() > 0, "stale group references remain owned until retirement");
    let mut steps = 0;
    while list.layout_pending() {
        let before = list.retired_entries_remaining();
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
        assert!(before.saturating_sub(list.retired_entries_remaining()) <= super::LAYOUT_CLOCK_STEPS);
        assert_eq!(list.draws().len(), active_commands, "retirement leaves active draw commands intact");
        assert_eq!(list.visible_items(), active_items, "retirement leaves active visibility intact");
        steps += 1;
        assert!(steps < 100, "stale group references are released in bounded turns");
    }
    assert!(list.retired.is_none(), "stale allocations are eventually reclaimed");
    assert_eq!(group_refs(&list, group), active_refs, "retirement releases every staged group reference");
}

#[test]
fn canceled_build_keeps_committed_patches() {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let mut visible = (1..=4_096)
        .map(|id| items.insert_item(new_item((2, id, 0), 0, 0.0, origin, &mesh, region, &textures, &mut list)).expect("test item index"))
        .collect::<Vec<_>>();
    layout(&mut list, &visible, &mut items.items, camera);
    let target = visible[500];
    let target_slot = |items: &FrameCache| items.items.get(target).expect("target item").faces[0].slot.expect("target face slot");
    let committed = target_slot(&items);
    let rotate = |items: &mut FrameCache, list: &mut DrawList, angle: f32| {
        let item = items.items.get_mut(target).expect("target item");
        item.rotation = Quat::from_rotation_z(angle);
        item.transform = instance_base(item.position, item.rotation, item.scale, false, origin);
        list.patch(&items.items, &[(target, 0)], &[target], origin, origin);
        items.items.get(target).expect("target item").transform
    };

    let entering = items.insert_item(new_item((2, 0, 0), 0, 0.0, origin, &mesh, region, &textures, &mut list)).expect("entering item");
    visible.insert(0, entering);
    assert!(list.request_layout(&visible, camera));
    for _ in 0..10_000 {
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
        if list.staged_progress().is_some_and(|(phase, cursor, ..)| phase == "fill" && cursor > 501) {
            break;
        }
    }
    assert_matches!(list.staged_progress(), Some(("fill", cursor, ..)) if cursor > 501, "the build staged the target face");
    assert_ne!(target_slot(&items), committed, "the staged build moved the target face");

    let transform = rotate(&mut items, &mut list, 0.3);
    assert_eq!(list.instance_transform(committed), Some(&transform[..]), "a pending build keeps committed poses patched");
    let rebased = origin + DVec3::new(256.0, 0.0, 0.0);
    list.patch(&items.items, &[(target, 0)], &[target], rebased, origin);
    let item = items.items.get(target).expect("target item");
    let transform = instance_base(item.position, item.rotation, item.scale, false, rebased);
    assert_eq!(list.instance_transform(committed), Some(&transform[..]), "a pending rebase keeps committed poses in the active origin");
    list.cancel_layout();
    let transform = rotate(&mut items, &mut list, 0.6);
    assert_eq!(list.instance_transform(committed), Some(&transform[..]), "a canceled build keeps committed poses patched");
    for _ in 0..100 {
        if !list.layout_pending() {
            break;
        }
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
    }
    assert!(!list.layout_pending(), "the canceled build retires");
    let transform = rotate(&mut items, &mut list, 0.9);
    assert_eq!(list.instance_transform(committed), Some(&transform[..]), "retirement keeps committed poses patched");

    layout(&mut list, &visible, &mut items.items, camera);
    let next = target_slot(&items);
    assert_ne!(next, committed, "the next generation moves the target face");
    let transform = rotate(&mut items, &mut list, 1.2);
    assert_eq!(list.instance_transform(next), Some(&transform[..]), "the next generation patches its own address");

    visible.remove(0);
    assert!(list.request_layout(&visible, camera));
    for _ in 0..10_000 {
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
        if list.staged_progress().is_some_and(|(phase, cursor, ..)| phase == "fill" && cursor > 500) {
            break;
        }
    }
    assert_matches!(list.staged_progress(), Some(("fill", cursor, ..)) if cursor > 500, "the build staged the target face");
    let staged = target_slot(&items);
    list.hide_item(target, items.items.get_mut(target).expect("target item"));
    assert!(instance_bytes(&list, next).iter().all(|byte| *byte == 0), "removing a staged item clears its committed instance");
    let mut completed = false;
    for _ in 0..10_000 {
        if let Some(progress) = list.advance_layout(&mut items.items, Duration::ZERO) {
            assert!(!progress.stale, "hiding a staged item must not stale the pending build");
            if progress.complete {
                completed = true;
                break;
            }
        }
    }
    assert!(completed, "the pending build commits after hiding a staged item");
    assert!(instance_bytes(&list, staged).iter().all(|byte| *byte == 0), "hiding a staged item keeps its staged slot zeroed after commit");
}

#[test]
fn plain_staging_retires_at_once() {
    let (_region_world, _view, camera, mut items, mut list, index) = active_layout();
    let staged = staged_layout(&mut list, index, camera, &items.items, 0);
    assert!(staged_entry_count(&staged) > 4 * super::LAYOUT_CLOCK_STEPS);
    list.pending = Some(staged);
    list.cancel_layout();

    let _ = list.advance_layout(&mut items.items, Duration::ZERO);
    assert!(!list.layout_pending(), "staged data without resources does not hold the next layout back");
    assert!(list.request_layout(&[index], camera));
}

#[test]
fn draw_changes_match_fresh_layout() {
    let operations = vec![
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 2, 5, 0, 0),
        (0, 3, 4, 0, 0),
        (2, 0, 1, 0, 0),
        (1, 1, 0, 0, 0),
        (0, 4, 0, 0, 0),
        (3, 2, 0, 1, -1),
        (4, 0, 0, 1, 0),
        (5, 0, 0, 0, 0),
    ];
    verify_operations(&operations);
}

#[test]
fn hiding_slot_preserves_others() {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let object = view.objects().next().expect("fixture object");
    let (_, mesh) = object.mesh().expect("fixture mesh");
    let mesh = mesh.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let first = items.insert_item(new_item(item_id(0), 0, 0.0, origin, &mesh, region, &textures, &mut list)).expect("first item index");
    let second = items.insert_item(new_item(item_id(1), 1, 0.0, origin, &mesh, region, &textures, &mut list)).expect("second item index");
    layout(&mut list, &[first, second], &mut items.items, camera);
    let commands = list.draws().len();
    let hidden_slot = items.items.get(first).and_then(|item| item.faces.first()).and_then(|face| face.slot).expect("first draw slot");
    let survivor_slot = items.items.get(second).and_then(|item| item.faces.first()).and_then(|face| face.slot).expect("second draw slot");
    let survivor_bytes = instance_bytes(&list, survivor_slot);

    let hidden = items.items.get_mut(first).and_then(|item| item.faces.first_mut()).expect("first face");
    assert!(list.hide_face(hidden));

    assert!(instance_bytes(&list, hidden_slot).iter().all(|byte| *byte == 0), "a hidden face draws a zero instance");
    assert_eq!(instance_bytes(&list, survivor_slot), survivor_bytes);
    assert_eq!(list.draws().len(), commands, "hiding keeps committed commands until the next layout");
    assert!(list.committed.dirty_slots.contains(&hidden_slot));
}

#[test]
fn overflow_omits_until_in_budget() {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let object = view.objects().next().expect("fixture object");
    let (_, mesh) = object.mesh().expect("fixture mesh");
    let mesh = mesh.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let capacity = 64;
    let faces_per_item = capacity / 2 + 1;
    let mut items = FrameCache::default();
    let mut list = DrawList::new(capacity);
    let mut indices = Vec::new();

    for (slot, style, x) in [(0, 0, 1.0), (1, 3, 100.0)] {
        let id = item_id(slot);
        let mut item = new_item(id, style, 0.0, origin, &mesh, region, &textures, &mut list);
        item.position = DVec3::new(x, 0.0, 0.0);
        let first_face = item.faces.pop().expect("test item face");
        let center = item.info.centers[0];
        let scale = item.scale;
        let mut faces = Vec::with_capacity(faces_per_item);
        faces.push(first_face);
        for _ in 1..faces_per_item {
            faces.push(new_face(id, style, &mesh, center, scale, &textures, &mut list));
        }
        item.faces = faces;
        indices.push(items.insert_item(item).expect("test item index"));
    }

    let visible = indices.clone();
    let stats = layout(&mut list, &visible, &mut items.items, camera);
    let omitted = items.items.get(indices[1]).expect("omitted item");
    let emitted = list.draws().iter().map(|draw| draw.count as usize).sum::<usize>();
    let retained_slots: HashSet<_> =
        items.items.get(indices[0]).expect("retained item").faces.iter().filter_map(|face| face.slot).collect();

    assert_eq!(stats.omitted_faces, faces_per_item);
    assert!(list.slot_count() <= capacity);
    assert_eq!(emitted, list.slot_count());
    assert_eq!(retained_slots.len(), faces_per_item);
    assert!(retained_slots.iter().all(|slot| usize::try_from(*slot).is_ok_and(|slot| slot < capacity)));
    assert!(omitted.faces.iter().all(|face| face.slot.is_none()), "omitted items cannot own instance slots");
    assert!(list.draws().iter().all(|draw| draw.pipeline != Pipeline::Blend), "omitted items cannot appear in draw commands");

    let retained_faces = capacity - faces_per_item - 1;
    let removed = items.items.get_mut(indices[1]).expect("second item").faces.split_off(retained_faces);
    list.release_faces(&removed);
    let stats = layout(&mut list, &visible, &mut items.items, camera);
    let emitted = list.draws().iter().map(|draw| draw.count as usize).sum::<usize>();
    let slots: HashSet<_> = indices
        .iter()
        .filter_map(|index| items.items.get(*index))
        .flat_map(|item| item.faces.iter().filter_map(|face| face.slot))
        .collect();

    assert_eq!(stats.omitted_faces, 0);
    assert_eq!(list.slot_count(), faces_per_item + retained_faces);
    assert_eq!(emitted, list.slot_count());
    assert_eq!(slots.len(), list.slot_count());
    assert!(slots.iter().all(|slot| usize::try_from(*slot).is_ok_and(|slot| slot < capacity)));
    assert!(list.slot_count() < capacity);
    assert!(indices.iter().all(|index| items.items.get(*index).is_some_and(|item| item.faces.iter().all(|face| face.slot.is_some()))));
    assert!(list.draws().iter().any(|draw| draw.pipeline == Pipeline::Blend), "restored items return to draw commands");
}

#[test]
fn moved_faces_cull_by_pose() {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let visible = (1..=4_096)
        .map(|id| items.insert_item(new_item((2, id, 0), 0, 0.0, origin, &mesh, region, &textures, &mut list)).expect("test item index"))
        .collect::<Vec<_>>();
    let target = visible[500];
    let move_target = |items: &mut FrameCache, list: &mut DrawList, position: DVec3| {
        let item = items.items.get_mut(target).expect("target item");
        item.position = position;
        item.transform = instance_base(item.position, item.rotation, item.scale, false, origin);
        list.patch(&items.items, &[(target, 0)], &[target], origin, origin);
    };
    let target_emitted = |items: &FrameCache, list: &DrawList, camera: Camera| {
        let mut view = ViewDraws::default();
        list.select(&items.items, camera, &Frustum::from_clip_matrix(pipeline::projection(camera, [128, 128])), &mut view);
        let face = &items.items.get(target).expect("target item").faces[0];
        face_command(list, &view, face).1
    };

    assert!(list.request_layout(&visible, camera));
    for _ in 0..10_000 {
        let _ = list.advance_layout(&mut items.items, Duration::ZERO);
        if list.staged_progress().is_some_and(|(phase, cursor, ..)| phase == "fill" && cursor > 501) {
            break;
        }
    }
    assert_matches!(list.staged_progress(), Some(("fill", cursor, ..)) if cursor > 501, "the build staged the target face");
    move_target(&mut items, &mut list, DVec3::new(-150.0, 0.0, 0.0));
    settle(&mut list, &mut items.items);
    let behind = Camera::new(DVec3::new(-100.0, 0.0, 0.0), Vec3::NEG_X, 1.0, [128, 128]).expect("test camera");
    assert!(target_emitted(&items, &list, behind), "a pose patched into a staged build moves its command bounds");

    move_target(&mut items, &mut list, DVec3::new(0.0, 0.0, 150.0));
    assert!(list.cull_dirty(), "a committed pose patch requests a cull");
    let above = Camera::new(DVec3::new(0.0, 0.0, 100.0), Vec3::Z, 1.0, [128, 128]).expect("test camera");
    assert!(target_emitted(&items, &list, above), "a pose patched into the committed list moves its command bounds");
}

#[test]
fn promoted_blended_keeps_drawing() {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let object = view.objects().next().expect("fixture object");
    let (_, mesh) = object.mesh().expect("fixture mesh");
    let mesh = mesh.clone();
    let textures = [loaded_texture(9_300_001), loaded_texture(9_300_002)];
    let camera = camera(-5.0);
    let origin = scene_origin(camera.position());
    let frustum = Frustum::from_clip_matrix(pipeline::projection(camera, [128, 128]));
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let index = items.insert_item(new_item(item_id(2), 3, 0.0, origin, &mesh, region, &textures, &mut list)).expect("item index");
    layout(&mut list, &[index], &mut items.items, camera);
    let mut view = ViewDraws::default();
    list.select(&items.items, camera, &frustum, &mut view);
    assert_eq!(view.blended.len(), 1, "the blended face draws after its layout commits");

    let replacement = new_item(item_id(2), 4, 0.0, origin, &mesh, region, &textures, &mut list);
    items.items.get_mut(index).expect("item").replacement = Some(Box::new(CachedReplacement { item: replacement }));
    items.replacements.insert(index);
    let previous = items.promote_replacement(index).expect("promoted item");
    list.release_faces(&previous.faces);

    list.select(&items.items, camera, &frustum, &mut view);
    assert_eq!(view.blended.len(), 1, "the committed blended face keeps drawing until the next layout commits");
}

#[test]
fn evicted_mesh_never_draws() {
    let (_region_world, region) = registered_region();
    let meshes = scene::distinct_meshes_for_test(2);
    let (evicted, kept) = (meshes[0].id(), meshes[1].id());
    let textures = [loaded_texture(9_200_001), loaded_texture(9_200_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let frustum = Frustum::from_clip_matrix(pipeline::projection(camera, [128, 128]));
    let drawn = |list: &DrawList, items: &FrameCache| {
        let mut view = ViewDraws::default();
        list.select(&items.items, camera, &frustum, &mut view);
        let state = &list.committed;
        let opaque = view.opaque.iter().map(|index| state.draws[*index as usize].mesh);
        let blended = view.blended.iter().map(|record| state.draws[state.opaque_draws + *record as usize].mesh);
        opaque.chain(blended).collect::<HashSet<_>>()
    };
    for step in 0.. {
        let mut items = FrameCache::default();
        let mut list = DrawList::new(usize::MAX);
        let visible = (1..=36u8)
            .map(|id| {
                let item = new_item(item_id(id), id, 0.0, origin, &meshes[usize::from(id % 2)], region, &textures, &mut list);
                items.insert_item(item).expect("test item index")
            })
            .collect::<Vec<_>>();
        layout(&mut list, &visible, &mut items.items, camera);
        assert_eq!(drawn(&list, &items), HashSet::from([evicted, kept]), "both meshes draw before eviction");

        assert!(list.request_layout(&visible, camera));
        let mut building = true;
        for _ in 0..step {
            let _ = list.advance_layout(&mut items.items, Duration::ZERO);
            building = list.layout_building();
            if !building {
                break;
            }
        }
        list.evict(evicted);
        assert!(!list.shows_mesh(evicted, camera, &frustum));
        assert!(!list.committed_pins_mesh(evicted));
        assert_eq!(drawn(&list, &items), HashSet::from([kept]), "eviction disables committed commands at once (step {step})");
        settle(&mut list, &mut items.items);
        assert!(!list.committed_pins_mesh(evicted));
        assert_eq!(drawn(&list, &items), HashSet::from([kept]), "a build that saw the eviction never draws the mesh (step {step})");
        if !building {
            break;
        }
    }
}

#[test]
fn forgotten_index_reuse_skipped() {
    let (_region_world, region) = registered_region();
    let meshes = scene::distinct_meshes_for_test(2);
    let (kept_mesh, reused_mesh) = (&meshes[0], &meshes[1]);
    let textures = [loaded_texture(9_200_001), loaded_texture(9_200_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let multi_face_item = |id: ItemId, mesh: &SharedMesh, list: &mut DrawList| {
        let mut item = new_item(id, 0, 0.0, origin, mesh, region, &textures, list);
        let center = item.info.centers[0];
        let scale = item.scale;
        for _ in 1..24 {
            item.faces.push(new_face(id, 0, mesh, center, scale, &textures, list));
        }
        item
    };

    for (distinct, replacement_mesh) in [(false, kept_mesh), (true, reused_mesh)] {
        let mut saw_fill = false;
        for step in 0.. {
            let mut items = FrameCache::default();
            let mut list = DrawList::new(usize::MAX);
            let visible = (1..=3u8)
                .map(|id| items.insert_item(multi_face_item(item_id(id), kept_mesh, &mut list)).expect("test item index"))
                .collect::<Vec<_>>();
            let target = visible[1];

            assert!(list.request_layout(&visible, camera));
            let mut building = true;
            for _ in 0..step {
                let _ = list.advance_layout(&mut items.items, Duration::ZERO);
                building = list.layout_building();
                if !building {
                    break;
                }
            }
            if list.staged_progress().is_some_and(|(phase, ..)| phase == "fill") {
                saw_fill = true;
            }
            list.forget_item(target);
            let (removed, _) = items.remove_item(target).expect("removed target item");
            list.release_faces(&removed.faces);
            let reused = items.insert_item(multi_face_item(item_id(9), replacement_mesh, &mut list)).expect("reused item index");
            assert_eq!(reused, target, "slab reuses the removed index immediately");

            settle(&mut list, &mut items.items);
            let replacement = items.items.get(reused).expect("reused item exists");
            assert!(
                replacement.faces.iter().all(|face| face.slot.is_none()),
                "reused item must not receive a slot from the pending build (distinct={distinct}, step={step})"
            );
            if distinct {
                assert!(!list.committed_pins_mesh(reused_mesh.id()), "no committed mesh reference for reused item (step={step})");
                assert!(
                    list.draws().iter().all(|draw| draw.mesh != reused_mesh.id()),
                    "no committed draw references reused item's mesh (step={step})"
                );
            }
            for survivor in [visible[0], visible[2]] {
                let item = items.items.get(survivor).expect("surviving item exists");
                assert!(item.faces.iter().all(|face| face.slot.is_some()), "surviving items still receive slots (step={step})");
            }
            if !building {
                break;
            }
        }
        assert!(saw_fill, "test exercised index reuse during the fill phase (distinct={distinct})");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        rng_seed: RngSeed::Fixed(0x51c3_77e0_9b2d_4f18),
        ..ProptestConfig::default()
    })]

    #[test]
    fn culling_matches_face_visibility(
        placements in collection::vec((-100.0f64..100.0, -100.0f64..100.0, -12.0f64..12.0, 0.25f32..12.0, 0u8..6), 1..160),
        eye in (-50.0f64..50.0, -50.0f64..50.0, -8.0f64..8.0),
        yaw in 0.0f64..TAU,
        pitch in -0.6f64..0.6,
        draw_distance in 32.0f64..160.0,
    ) {
        assert_cull_matches_faces(&placements, DVec3::new(eye.0, eye.1, eye.2), yaw, pitch, draw_distance);
    }
}

fn assert_cull_matches_faces(placements: &[(f64, f64, f64, f32, u8)], eye: DVec3, yaw: f64, pitch: f64, draw_distance: f64) {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let direction = DVec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()).as_vec3();
    let viewport = [128, 96];
    let camera =
        Camera::new(eye, direction, 1.0, viewport).and_then(|camera| camera.with_quality(draw_distance, 1.0)).expect("test camera");
    let origin = scene_origin(camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let mut visible = Vec::new();
    for (id, (x, y, z, scale, style)) in placements.iter().enumerate() {
        let mut item = new_item((2, id as u128, 0), *style, 0.0, origin, &mesh, region, &textures, &mut list);
        item.position = DVec3::new(*x, *y, *z);
        item.scale = Vec3::splat(*scale);
        item.transform = instance_base(item.position, item.rotation, item.scale, false, origin);
        visible.push(items.insert_item(item).expect("test item index"));
    }
    visible.sort_unstable_by_key(|index| items.items.get(*index).map(|item| item.id));
    layout(&mut list, &visible, &mut items.items, camera);

    let frustum = Frustum::from_clip_matrix(pipeline::projection(camera, viewport));
    let mut view = ViewDraws::default();
    list.select(&items.items, camera, &frustum, &mut view);
    for index in &visible {
        let item = items.items.get(*index).expect("visible item");
        let expected = WorldSphere::new(item.position, item.rotation, item.scale, item.info.bounds, camera)
            .is_some_and(|sphere| sphere.visible(item.shading, camera, &frustum));
        for face in &item.faces {
            let (count, emitted) = face_command(&list, &view, face);
            if count == 1 {
                assert_eq!(emitted, expected, "a single-instance command is emitted exactly when its face is visible");
            } else if expected {
                assert!(emitted, "an instanced command is emitted when any of its faces is visible");
            }
        }
    }
}

fn face_command(list: &DrawList, view: &ViewDraws, face: &CachedFace) -> (u32, bool) {
    let state = &list.committed;
    let address = list.committed_address(face).expect("face has a committed slot");
    match address.blended_index {
        Some(record) => {
            let draw = &state.draws[state.opaque_draws + record];
            (draw.count, view.blended.contains(&(record as u32)))
        }
        None => {
            let index = state.opaque().partition_point(|draw| draw.first + draw.count <= address.slot);
            (state.draws[index].count, view.opaque.contains(&(index as u32)))
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        max_shrink_iters: 20_000,
        rng_seed: RngSeed::Fixed(0x8d12_2d3a_a5c7_49e1),
        ..ProptestConfig::default()
    })]

    #[test]
    fn randomized_updates_match_fresh_layouts(
        operations in collection::vec((0u8..6, 0u8..8, 0u8..9, -3i8..4, -3i8..4), 1..80)
    ) {
        let generated = operations;
        let mut operations = vec![
            (0, 0, 0, 0, 0),
            (0, 1, 3, 0, 0),
            (0, 2, 5, 0, 0),
            (2, 0, 1, 0, 0),
            (3, 1, 0, 1, -1),
            (4, 0, 0, 1, 0),
            (5, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
        ];
        operations.extend(generated);
        verify_operations(&operations);
    }

    #[test]
    fn fullbright_draws_follow_lit(styles in collection::vec(0u8..9, 1..24)) {
        let (_region_world, region) = registered_region();
        let (_scene, view, _) = fixtures::scene(255);
        let mesh = view.objects().next().expect("fixture object").mesh().expect("fixture mesh").1.clone();
        let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
        let camera = camera(0.0);
        let origin = scene_origin(camera.position());
        let mut items = FrameCache::default();
        let mut list = DrawList::new(usize::MAX);
        let visible = styles
            .iter()
            .copied()
            .chain(iter::repeat_n(0, 1_100))
            .zip(1u128..)
            .map(|(style, id)| {
                items.insert_item(new_item((2, id, 0), style, 0.0, origin, &mesh, region, &textures, &mut list)).expect("test item index")
            })
            .collect::<Vec<_>>();
        let has_fullbright = styles.iter().any(|style| style % 9 >= 6);
        let rank = |list: &DrawList, group| {
            let key = list.face_key(group).expect("drawn group key");
            if key.blended() { 2 } else { u8::from(key.fullbright()) }
        };

        layout(&mut list, &visible, &mut items.items, camera);
        let ranks = list.draws().iter().map(|draw| rank(&list, draw.group)).collect::<Vec<_>>();
        prop_assert!(ranks.is_sorted(), "draw classes out of order: {ranks:?}");
        prop_assert_eq!(ranks.iter().filter(|rank| **rank < 2).count(), list.committed.opaque_draws);
        prop_assert_eq!(ranks.contains(&1), has_fullbright);

        let mut view = ViewDraws::default();
        list.select(&items.items, camera, &Frustum::from_clip_matrix(pipeline::projection(camera, [128, 128])), &mut view);
        let culled = view.opaque.iter().map(|index| ranks[*index as usize]).collect::<Vec<_>>();
        prop_assert!(!culled.is_empty() && culled.is_sorted(), "culled opaque classes out of order: {culled:?}");

        prop_assert!(list.request_layout(&visible, camera));
        let mut steps = 0;
        while !list.staged_progress().is_some_and(|(phase, ..)| matches!(phase, "allocate" | "fill")) {
            let _ = list.advance_layout(&mut items.items, Duration::ZERO);
            steps += 1;
            prop_assert!(steps < 100, "staged layout reaches allocation");
        }
        let state = &list.pending.as_ref().expect("staged layout").state;
        let staged = state.draws.iter().map(|draw| rank(&list, draw.group)).collect::<Vec<_>>();
        prop_assert_eq!(staged.len(), state.opaque_draws);
        prop_assert!(staged.is_sorted() && !staged.contains(&2), "staged opaque classes out of order: {staged:?}");
        prop_assert_eq!(staged.contains(&1), has_fullbright);
    }
}

fn verify_operations(operations: &[(u8, u8, u8, i8, i8)]) {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let first_object = view.objects().next().expect("fixture object");
    let (_, mesh) = first_object.mesh().expect("fixture mesh");
    let mesh = mesh.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera_x = 0.0;
    let current_camera = camera(camera_x);
    let origin = scene_origin(current_camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let mut visible = visible_items(&items.items, camera_x);
    layout(&mut list, &visible, &mut items.items, current_camera);
    assert_matches_fresh(&list, &items.items, &visible, current_camera);
    let mut camera_x = camera_x;
    let mut origin = origin;

    for (kind, item_index, style, delta_x, delta_y) in operations {
        let id = item_id(*item_index);
        let previous_visible = visible.clone();
        let mut relayout = false;
        match kind % 6 {
            0 => {
                if items.index(id).is_none() {
                    items
                        .insert_item(new_item(id, *style, camera_x, origin, &mesh, region, &textures, &mut list))
                        .expect("test item index");
                    relayout = true;
                }
            }
            1 => {
                if let Some(index) = items.index(id)
                    && let Some((item, _)) = items.remove_item(index)
                {
                    list.release_faces(&item.faces);
                    relayout = true;
                }
            }
            2 => {
                if let Some(item) = items.get_by_id_mut(id) {
                    list.release_faces(&item.faces);
                    item.faces = vec![new_face(id, *style, &item.mesh, item.info.centers[0], item.scale, &textures, &mut list)];
                    relayout = true;
                }
            }
            3 => {
                let changed = if let Some(item) = items.get_by_id_mut(id) {
                    item.position.x += f64::from(*delta_x);
                    item.position.y += f64::from(*delta_y);
                    item.rotation = Quat::from_rotation_y(f32::from(*delta_x) * 0.025);
                    item.transform = instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin);
                    true
                } else {
                    false
                };
                if changed {
                    let index = items.index(id).expect("updated test item");
                    list.patch(&items.items, &[(index, 0)], &[index], origin, origin);
                }
            }
            4 => {
                camera_x += f64::from(*delta_x);
                let next_origin = scene_origin(camera(camera_x).position());
                if next_origin != origin {
                    origin = next_origin;
                    for (_, item) in items.items.iter_mut() {
                        item.transform = instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin);
                    }
                    relayout = true;
                }
            }
            _ => {
                camera_x = (camera_x / 256.0).floor() * 256.0 + 256.0;
                origin = scene_origin(camera(camera_x).position());
                for (_, item) in items.items.iter_mut() {
                    item.transform = instance_base(item.position, item.rotation, item.scale, item.invert_normals, origin);
                }
                relayout = true;
            }
        }

        let camera = camera(camera_x);
        let next_visible = visible_items(&items.items, camera_x);
        relayout |= previous_visible != next_visible;
        if relayout {
            layout(&mut list, &next_visible, &mut items.items, camera);
        }
        visible = next_visible;
        assert_matches_fresh(&list, &items.items, &visible, camera);
    }
}

fn active_layout() -> (world::World, SceneView, Camera, FrameCache, DrawList, ItemIndex) {
    let (_region_world, region) = registered_region();
    let (_scene, view, _) = fixtures::scene(255);
    let object = view.objects().next().expect("fixture object");
    let (_, mesh) = object.mesh().expect("fixture mesh");
    let mesh = mesh.clone();
    let textures = [loaded_texture(9_100_001), loaded_texture(9_100_002)];
    let camera = camera(0.0);
    let origin = scene_origin(camera.position());
    let mut items = FrameCache::default();
    let mut list = DrawList::new(usize::MAX);
    let index = items.insert_item(new_item(item_id(0), 0, 0.0, origin, &mesh, region, &textures, &mut list)).expect("test item index");
    layout(&mut list, &[index], &mut items.items, camera);
    (_region_world, view, camera, items, list, index)
}

const STAGED_REFS: usize = 2_047;

fn staged_layout(list: &mut DrawList, index: ItemIndex, camera: Camera, items: &ItemSlab, refs: usize) -> super::LayoutBuild {
    let input = vec![index; 2_047];
    let mut build = super::LayoutBuild::new(Default::default(), &input, camera, usize::MAX);
    let group = items.get(index).expect("staged item").faces[0].group;
    for _ in 0..refs {
        assert!(list.groups.retain(group));
        build.state.draw_group_refs.push(group);
    }
    build.state.instance_bytes.resize(3 * 1024 * 1024, 0);
    build
}

fn group_refs(list: &DrawList, group: u32) -> usize {
    list.groups.group(group).expect("interned group").refs
}

fn staged_entry_count(build: &super::LayoutBuild) -> usize {
    let state = &build.state;
    build.uploads.len()
        + build.visible.len()
        + build.removed.len()
        + build.evicted.len()
        + build.layout_items.len()
        + build.selected_items.len()
        + build.commands.len()
        + build.command_keys.len()
        + build.cells.len()
        + build.item_cells.len()
        + build.priority.len()
        + build.blended.len()
        + state.draws.len()
        + state.cells.len()
        + state.mesh_references.len()
        + state.active_textures.len()
        + state.active_terrain.len()
        + state.texture_seen.len()
        + state.terrain_seen.len()
        + state.layout_items.len()
        + state.draw_group_refs.len()
        + state.draw_group_seen.len()
        + state.blended.len()
        + state.dirty_slots.len()
}

fn staged_allocation_bytes(build: &super::LayoutBuild) -> usize {
    fn vec_bytes<T>(values: &Vec<T>) -> usize {
        values.capacity().saturating_mul(size_of::<T>())
    }

    fn set_bytes<T>(values: &HashSet<T>) -> usize {
        values.capacity().saturating_mul(size_of::<T>())
    }

    fn map_bytes<K, V>(values: &HashMap<K, V>) -> usize {
        values.capacity().saturating_mul(size_of::<(K, V)>())
    }

    let state = &build.state;
    [
        set_bytes(&build.uploads),
        vec_bytes(&build.visible),
        set_bytes(&build.removed),
        set_bytes(&build.evicted),
        vec_bytes(&build.layout_items),
        set_bytes(&build.selected_items),
        map_bytes(&build.commands),
        vec_bytes(&build.command_keys),
        map_bytes(&build.cells),
        vec_bytes(&build.item_cells),
        build.priority.capacity().saturating_mul(size_of::<super::PriorityItem>()),
        vec_bytes(&build.blended),
        vec_bytes(&state.draws),
        vec_bytes(&state.cells),
        map_bytes(&state.mesh_references) + state.mesh_references.values().map(vec_bytes).sum::<usize>(),
        vec_bytes(&state.instance_bytes),
        vec_bytes(&state.active_textures),
        vec_bytes(&state.active_terrain),
        set_bytes(&state.texture_seen),
        set_bytes(&state.terrain_seen),
        vec_bytes(&state.layout_items),
        vec_bytes(&state.draw_group_refs),
        set_bytes(&state.draw_group_seen),
        vec_bytes(&state.blended),
        vec_bytes(&state.dirty_slots),
    ]
    .into_iter()
    .fold(0usize, usize::saturating_add)
}

fn new_item(
    id: ItemId,
    style: u8,
    camera_x: f64,
    origin: DVec3,
    mesh: &SharedMesh,
    region: Region,
    textures: &[assets::TextureHandle; 2],
    list: &mut DrawList,
) -> CachedItem {
    let scale = Vec3::ONE;
    let position = DVec3::new(camera_x + (id.1 % 3) as f64 - 1.0, (id.1 % 2) as f64, 0.0);
    let rotation = Quat::IDENTITY;
    let bounds = mesh.mesh().bounds();
    let center = bounds.min * 0.5 + bounds.max * 0.5;
    let info = Arc::new(MeshInfo { bounds, centers: vec![center] });
    let transform = instance_base(position, rotation, scale, false, origin);
    let face = new_face(id, style, mesh, center, scale, textures, list);
    CachedItem {
        id,
        entity: None,
        region,
        mesh: mesh.clone(),
        info,
        position,
        rotation,
        scale,
        invert_normals: false,
        replacement: None,
        transform,
        pickable: true,
        shading: ShadingMode::Surface,
        terrain_textures: None,
        terrain_composition: None,
        surface_kind: None,
        source: None,
        faces: vec![face],
        animated_faces: Vec::new(),
        animation_deadline: None,
        parameters_dirty: false,
    }
}

fn new_face(
    id: ItemId,
    style: u8,
    mesh: &SharedMesh,
    center: Vec3,
    scale: Vec3,
    textures: &[assets::TextureHandle; 2],
    list: &mut DrawList,
) -> CachedFace {
    let mesh_face = mesh.mesh().faces().first().expect("fixture mesh face");
    let (pipeline, texture_index, blend, double_sided, fullbright): (Pipeline, Option<usize>, bool, bool, bool) = match style % 9 {
        0 => (Pipeline::Surface, None, false, false, false),
        1 => (Pipeline::Surface, Some(0), false, false, false),
        2 => (Pipeline::DoubleSided, Some(1), false, true, false),
        3 => (Pipeline::Blend, None, true, false, false),
        4 => (Pipeline::BlendDoubleSided, Some(0), true, true, false),
        5 => (Pipeline::BlendFlipped, Some(1), true, false, false),
        6 => (Pipeline::Fullbright, None, false, false, true),
        7 => (Pipeline::FullbrightDoubleSided, Some(0), false, true, true),
        _ => (Pipeline::FullbrightFlipped, Some(1), false, false, true),
    };
    let texture = texture_index.map(|index| textures[index].clone());
    let mut params = FaceParams::new(None);
    params.blend = blend;
    params.double_sided = double_sided;
    params.fullbright = fullbright;
    params.textures.base = texture.as_ref().map(|handle| TextureKey { id: handle.id(), sampler: Sampler::default() });
    let images = MaterialImages { base: texture, emissive: None };
    params.color = [0.3 + f32::from(style) * 0.05, 0.4, 0.5, 1.0];
    let key = DrawKey {
        mesh: mesh.id(),
        start: mesh_face.indices.start,
        end: mesh_face.indices.end,
        pipeline,
        textures: params.textures,
        terrain: None,
    };
    let group = list.intern_face(key, &images, None).expect("test draw group ID");
    CachedFace {
        order: (id.0, id.1, id.2, mesh_face.slot),
        indices: mesh_face.indices.clone(),
        center,
        base_params: params,
        params,
        images,
        material_index: None,
        texture_entry_transform: Default::default(),
        group,
        blended: key.blended(),
        parameter_bytes: parameter_part(&params, scale),
        slot: None,
        blended_index: None,
        layout: Default::default(),
        committed: None,
    }
}

fn assert_matches_fresh(list: &DrawList, items: &ItemSlab, visible: &[ItemIndex], camera: Camera) {
    let expected = fresh_output(list, items, visible, camera);
    let state = &list.committed;
    let mut opaque = OpaqueOutput::new();
    let mut blended = HashMap::<(DrawKey, Vec<u8>), usize>::new();
    let mut slots = HashSet::new();
    for (index, draw) in state.draws.iter().enumerate() {
        let key = draw_key(draw);
        for slot in draw.first..draw.first.saturating_add(draw.count) {
            assert!(slots.insert(slot), "each draw command covers a unique instance slot");
            let bytes = instance_bytes(list, slot);
            if index < state.opaque_draws {
                *opaque.entry((key, bytes)).or_default() += 1;
            } else {
                *blended.entry((key, bytes)).or_default() += 1;
            }
        }
    }
    let expected_slots =
        items.iter().filter(|(index, _)| contains_item(visible, items, *index)).map(|(_, item)| item.faces.len()).sum::<usize>();
    assert_eq!(slots.len(), expected_slots, "every visible face has one draw slot");
    assert_eq!(list.slot_count(), expected_slots, "instance output contains only visible face bytes");
    assert!(opaque == expected.0, "opaque draw key and packed byte multiset differs from a fresh layout");
    let mut expected_blended = HashMap::<(DrawKey, Vec<u8>), usize>::new();
    for entry in &expected.1 {
        *expected_blended.entry(entry.clone()).or_default() += 1;
    }
    assert!(blended == expected_blended, "blended draw key and packed byte multiset differs from a fresh layout");

    let mut view = ViewDraws::default();
    list.select(items, camera, &Frustum::from_clip_matrix(Mat4::ZERO), &mut view);
    assert_eq!(view.opaque.len(), state.opaque_draws, "every opaque command has bounds");
    let order = view
        .blended
        .iter()
        .map(|record| {
            let draw = &state.draws[state.opaque_draws + *record as usize];
            (draw_key(draw), instance_bytes(list, draw.first))
        })
        .collect::<BlendedOutput>();
    assert!(order == expected.1, "blended draws are not emitted back to front");
    for (index, item) in items.iter() {
        for face in &item.faces {
            assert_eq!(face.slot.is_some(), contains_item(visible, items, index), "hidden items cannot retain instance slots");
        }
    }
}

fn draw_key(draw: &super::DrawCommand) -> DrawKey {
    DrawKey {
        mesh: draw.mesh,
        start: draw.indices.start,
        end: draw.indices.end,
        pipeline: draw.pipeline,
        textures: draw.textures,
        terrain: draw.terrain,
    }
}

fn fresh_output(list: &DrawList, items: &ItemSlab, visible: &[ItemIndex], camera: Camera) -> (OpaqueOutput, BlendedOutput) {
    let mut opaque = OpaqueOutput::new();
    let mut blended = Vec::new();
    for id in visible {
        let Some(item) = items.get(*id) else { continue };
        for face in &item.faces {
            let Some(key) = list.face_key(face.group) else { continue };
            let bytes = instance_from_parts(&item.transform, &face.parameter_bytes);
            if key.blended() {
                let relative = item.position - camera.position() + (item.rotation * (face.center * item.scale)).as_dvec3();
                blended.push((relative.length_squared(), face.order, key, bytes));
            } else {
                *opaque.entry((key, bytes)).or_default() += 1;
            }
        }
    }
    blended.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    (opaque, blended.into_iter().map(|(_, _, key, bytes)| (key, bytes)).collect())
}

fn instance_bytes(list: &DrawList, slot: u32) -> Vec<u8> {
    let start = slot as usize * INSTANCE_BYTES;
    list.committed.instance_bytes.get(start..start + INSTANCE_BYTES).expect("draw slot has packed instance bytes").to_vec()
}

fn layout(list: &mut DrawList, visible: &[ItemIndex], items: &mut ItemSlab, camera: Camera) -> LayoutStats {
    settle(list, items);
    assert!(list.request_layout(visible, camera), "an idle list accepts a layout");
    let mut stats = None;
    for _ in 0..100_000 {
        if let Some(progress) = list.advance_layout(items, Duration::ZERO) {
            assert!(!progress.stale, "test layouts do not go stale");
            if progress.complete {
                stats = Some(progress.stats);
            }
        }
        if stats.is_some() && !list.layout_pending() {
            break;
        }
    }
    assert!(!list.layout_pending(), "the layout commits and retires");
    stats.expect("the layout commits")
}

fn settle(list: &mut DrawList, items: &mut ItemSlab) {
    for _ in 0..100_000 {
        if !list.layout_pending() {
            return;
        }
        let _ = list.advance_layout(items, Duration::ZERO);
    }
    panic!("pending layout work did not settle");
}

fn instance_from_parts(transform: &[u8; TRANSFORM_BYTES], parameters: &[u8; PARAMETER_BYTES]) -> Vec<u8> {
    transform.iter().chain(parameters).copied().collect()
}

fn visible_items(items: &ItemSlab, camera_x: f64) -> Vec<ItemIndex> {
    let mut visible =
        items.iter().filter(|(_, item)| (item.position.x - camera_x).abs() <= 6.0).map(|(index, _)| index).collect::<Vec<_>>();
    visible.sort_unstable_by_key(|index| items.get(*index).map(|item| item.id));
    visible
}

fn contains_item(visible: &[ItemIndex], items: &ItemSlab, index: ItemIndex) -> bool {
    let Some(id) = items.get(index).map(|item| item.id) else { return false };
    visible.binary_search_by(|visible| items.get(*visible).map_or(cmp::Ordering::Less, |item| item.id.cmp(&id))).is_ok()
}

fn camera(camera_x: f64) -> Camera {
    Camera::new(DVec3::new(camera_x, 0.0, 0.0), Vec3::X, 1.0, [128, 128]).expect("test camera")
}

fn item_id(index: u8) -> ItemId {
    (2, u128::from(index), 0)
}

fn loaded_texture(id: u128) -> assets::TextureHandle {
    let mut store = Store::new(Budget::default());
    let route = store.register_route().expect("test asset route");
    let now = Instant::now();
    let resource = Resource { id: AssetId::new(Uuid::from_u128(id)), kind: assets::Kind::Texture };
    let demand = store.request(route, Request::new(resource), Priority::Visible, now).expect("test texture request");
    let LoadJob::Fetch(fetch) = store.next_job(now).expect("fresh test texture fetch") else {
        panic!("fresh test texture used a cache entry")
    };
    let input = fetch.complete(Ok(GRAY_ALPHA_J2C.to_vec())).expect("test texture response");
    let completion = store.start_decode(input).expect("test decode admission").run();
    store.complete(completion, now);
    let Status::Ready(Content::Texture(texture)) = demand.status() else { panic!("test texture did not decode") };
    texture
}
