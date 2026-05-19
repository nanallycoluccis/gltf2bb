from __future__ import annotations

import json
import math
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (
    CleanupConfig,
    ComplexSplitConfig,
    FaceFeatureProtectionConfig,
    HybridDetailSplitConfig,
    OrientedCubesConfig,
    ProcessingConfig,
    UnskinnedMeshesConfig,
    resolve_processing_config,
)
from .constants import MODE_TRIANGLES, MODE_TRIANGLE_FAN, MODE_TRIANGLE_STRIP
from .errors import ConvertError, InspectError
from .inspect import SUPPORTED_MODEL_SUFFIXES, build_parent_map, is_valid_index, load_gltf, read_accessor
from .partition import (
    BonePartition,
    PartitionReport,
    build_filtered_bone_partitions,
    choose_face_owner,
    fallback_joint_for_skin,
    read_faces,
    read_optional_accessor,
    resolve_bone_node,
)
from .conversion.bbmodel import build_bbmodel
from .conversion.constants import AUTO_SPATIAL_SPLIT_OWNER_BUDGET_MULTIPLIER, DEFAULT_COMPLEX_SPLIT_BONE, EPSILON, MIN_CUBE_SIZE
from .conversion.geometry import (
    bbox_corners,
    bbox_dimensions,
    bbox_volume,
    box_volume,
    box_volume_from_dimensions,
    clamp_point_to_bbox,
    combined_bbox,
    compute_scale_and_offset,
    compute_world_matrices,
    count_small_cubes,
    ensure_min_cube_size,
    has_nonzero_rotation,
    identity_matrix,
    matrix_to_euler_xyz_degrees,
    matrix_translation,
    normalized_rotation_matrix,
    oriented_accumulator_bounds,
    planar_principal_rotation_matrices,
    points_bbox,
    points_centroid,
    principal_axis_3d,
    rotation_matrix_from_y_axis,
    row_to_vec3,
    to_blockbench_space,
    transform_point,
    zero_thickness_axes,
)
from .conversion.reporting import convert_result_to_dict, format_convert_summary, write_convert_report
from .conversion.types import (
    AssignedUnskinnedMesh,
    AutoSpatialPart,
    BBoxAccumulator,
    CleanupPartReport,
    CleanupReport,
    ComplexSplitBoneReport,
    ComplexSplitSubpartReport,
    ConvertResult,
    Cuboid,
    FaceFeatureProtectionAction,
    HairSplitPart,
    HairSplitResult,
    HybridModeReport,
    OrientedCubeReport,
    OrientationDecisionReport,
    PartKey,
    SkippedUnskinnedMesh,
    SplitFace,
)


SUPPORTED_CONVERT_MODES = {"cuboid", "hybrid"}
HYBRID_SPECIAL_CUBE_BONES = ("head", "hair", "skirt", "coat", "accessory")
AUTO_SPATIAL_SPLIT_MIN_FACES = 48
AUTO_SPATIAL_SPLIT_VOLUME_RATIO = 0.004
AUTO_SPATIAL_SPLIT_LONG_DIM_RATIO = 0.30
AUTO_SPATIAL_SPLIT_SECOND_DIM_RATIO = 0.14
AUTO_SPATIAL_SPLIT_VOLUME_LONG_DIM_RATIO = 0.18
AUTO_SPATIAL_SPLIT_VOLUME_SECOND_DIM_RATIO = 0.08
AUTO_SPATIAL_SPLIT_AXIS_DIM_RATIO = 0.16
AUTO_SPATIAL_SPLIT_TARGET_FACES = 80
AUTO_SPATIAL_SPLIT_MAX_AXES = 3
AUTO_ORIENT_MIN_LONG_DIM = 0.25
AUTO_ORIENT_MIN_VOLUME = 0.005
AUTO_ORIENT_MIN_FACES = 16
AUTO_ORIENT_MIN_VOLUME_REDUCTION = 0.12
HEAD_HAIR_COMPONENT_PRESERVE_LIMIT = 8
DEFAULT_HEAD_FRONT_SIGN = -1
HEAD_DETAIL_AUTO_ORIENT_MIN_VOLUME_REDUCTION = 0.03
HEAD_ACCESSORY_SPLIT_MIN_FACES = 24
HEAD_CORE_SPLIT_MIN_FACES = 800
HEAD_CORE_SPLIT_MIN_BUCKET_FACES = 64
HEAD_HAIR_REFINEMENT_MIN_FACES = 1200
HEAD_HAIR_REFINEMENT_MIN_BUCKET_FACES = 400
HEAD_HAIR_REFINEMENT_TARGET_FACES = 900
HEAD_HAIR_REFINEMENT_MIN_AXIS_RATIO = 0.32
HEAD_HAIR_REFINEMENT_MAX_AXES = 3
REGULAR_SPATIAL_DETAIL_MIN_FACES = 200
REGULAR_SPATIAL_DETAIL_AXIS_RATIO = 0.40
REGULAR_SPATIAL_DETAIL_MAX_AXES = 2
HAIR_BUCKET_MIN_FACES = 4
HAIR_BUCKET_MIN_FACE_RATIO = 0.08
HAIR_BUCKET_OVERLAP_RATIO = 0.015
HAIR_BUCKET_OVERLAP_MIN = MIN_CUBE_SIZE * 0.25
FACE_FEATURE_PROTECTION_MARGIN_RATIO = 0.002
FACE_FEATURE_PROTECTION_MIN_FACES = 32
COMPLEX_BONE_ALIASES = {
    "head": ("head", "頭", "頭部"),
    "hair": ("hair", "髪", "hair_", "hair-"),
    "skirt": ("skirt", "スカート"),
    "coat": ("coat", "jacket", "cloak", "cape", "上着"),
    "accessory": ("accessory", "ribbon", "hat", "飾", "リボン"),
    "neck": ("neck", "首"),
}
HEAD_MATERIAL_PATTERNS = {
    "hair": ("hair", "髪", "髮", "前髪", "後髪", "头发", "頭髮", "頭发"),
    "brow": ("brow", "eyebrow", "眉"),
    "eyelash": ("eyelash", "lash", "まつげ", "睫"),
    "eye": ("eye", "iris", "pupil", "目", "眼", "瞳"),
    "mouth": ("mouth", "teeth", "tongue", "口", "嘴", "唇", "舌", "牙"),
    "nose": ("nose", "鼻"),
    "ear": ("ear", "耳"),
    "head_accessory": (
        "ribbon",
        "accessory",
        "hat",
        "glasses",
        "goggle",
        "eyewear",
        "リボン",
        "メガネ",
        "ゴーグル",
        "眼鏡",
        "眼镜",
        "护目镜",
        "護目鏡",
        "头饰",
        "頭飾",
        "头带",
        "頭帶",
        "饰",
        "飾",
    ),
    "head_core": ("face", "skin", "head", "顔", "肌", "脸", "臉", "皮肤", "皮膚"),
}
FACE_FEATURE_PART_PREFIXES = ("eye", "mouth", "brow", "eyelash", "nose")
SIDE_FEATURE_PART_PREFIXES = ("ear",)


def convert_model(
    input_path: Path,
    output_path: Path,
    *,
    mode: str = "cuboid",
    target_height: float = 32.0,
    preset: str | None = None,
    config_path: Path | None = None,
    processing_config: ProcessingConfig | None = None,
    complex_split: tuple[str, ...] | list[str] | None = None,
    report_path: Path | None = None,
) -> ConvertResult:
    if mode not in SUPPORTED_CONVERT_MODES:
        supported = ", ".join(sorted(SUPPORTED_CONVERT_MODES))
        raise ConvertError(f"unsupported convert mode {mode!r}; supported modes: {supported}")
    if target_height <= 0:
        raise ConvertError("target height must be greater than 0")
    if not input_path.exists():
        raise ConvertError(f"input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ConvertError(f"input path is not a file: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise ConvertError(f"expected a .gltf, .glb, or .vrm file, got: {input_path.name}")

    try:
        gltf, binary_chunk = load_gltf(input_path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error, InspectError) as exc:
        raise ConvertError(f"failed to read glTF data from {input_path}: {exc}") from exc

    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    skins = gltf.get("skins", [])
    accessors = gltf.get("accessors", [])
    parent_map = build_parent_map(nodes)
    config = processing_config or resolve_processing_config(preset, config_path)
    if complex_split is not None:
        config = apply_complex_split_override(config, complex_split)
    config, hybrid_report = apply_mode_defaults(config, mode)
    bones, bone_resolution = build_filtered_bone_partitions(nodes, skins, parent_map, config)
    vrm_humanoid_nodes = extract_vrm_humanoid_nodes(gltf, bone_resolution.resolved_bones)
    world_matrices = compute_world_matrices(nodes, parent_map)
    buffer_cache: dict[int, bytes] = {}
    warnings: list[str] = list(bone_resolution.warnings)
    accumulators: dict[PartKey, BBoxAccumulator] = {}
    regular_faces_by_part: dict[PartKey, list[SplitFace]] = {}
    split_faces: list[SplitFace] = []
    assigned_unskinned_meshes: list[AssignedUnskinnedMesh] = []
    skipped_unskinned_meshes: list[SkippedUnskinnedMesh] = []

    scratch_report = PartitionReport(
        path=input_path,
        scenes=len(gltf.get("scenes", [])),
        nodes=len(nodes),
        meshes=len(meshes),
        skins=len(skins),
        animations=len(gltf.get("animations", [])),
        materials=len(gltf.get("materials", [])),
        bones=bones,
        bone_resolution=bone_resolution,
        primitives=[],
        warnings=warnings,
    )

    for node_index, node in enumerate(nodes):
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if mesh_index is None:
            continue
        if not is_valid_index(meshes, mesh_index):
            warnings.append(f"Node {node_index} references missing mesh {mesh_index}; skipped convert.")
            continue
        if skin_index is None:
            if not config.unskinned_meshes.enabled:
                skipped_unskinned_meshes.append(
                    SkippedUnskinnedMesh(node_index=node_index, mesh_index=mesh_index, reason="disabled")
                )
                warnings.append(f"Node {node_index} mesh {mesh_index} has no skin; skipped convert.")
                continue

            node_world = world_matrices.get(node_index, identity_matrix())
            before_count = len(assigned_unskinned_meshes)
            before_skipped_count = len(skipped_unskinned_meshes)
            node_name = node.get("name") if isinstance(node.get("name"), str) else None
            mesh_name = meshes[mesh_index].get("name") if isinstance(meshes[mesh_index].get("name"), str) else None
            for primitive_index, primitive in enumerate(meshes[mesh_index].get("primitives", [])):
                material_name = primitive_material_name(gltf.get("materials", []), primitive.get("material"))
                ignore_reason = ignored_unskinned_primitive_reason(
                    config.unskinned_meshes,
                    node_name,
                    mesh_name,
                    material_name,
                )
                if ignore_reason is not None:
                    skipped_unskinned_meshes.append(
                        SkippedUnskinnedMesh(
                            node_index=node_index,
                            mesh_index=mesh_index,
                            primitive_index=primitive_index,
                            reason=ignore_reason,
                        )
                    )
                    continue
                assignment = collect_unskinned_primitive_cuboids(
                    gltf,
                    input_path,
                    binary_chunk,
                    buffer_cache,
                    accessors,
                    primitive,
                    node_world,
                    node_index,
                    mesh_index,
                    primitive_index,
                    gltf.get("materials", []),
                    config.unskinned_meshes,
                    parent_map,
                    bone_resolution.resolved_bones,
                    bones,
                    world_matrices,
                    scratch_report,
                    accumulators,
                    regular_faces_by_part,
                )
                if assignment is not None:
                    assigned_unskinned_meshes.append(assignment)
            if len(assigned_unskinned_meshes) == before_count and len(skipped_unskinned_meshes) == before_skipped_count:
                skipped_unskinned_meshes.append(
                    SkippedUnskinnedMesh(
                        node_index=node_index,
                        mesh_index=mesh_index,
                        reason="no_convertible_or_assignable_primitives",
                    )
                )
            continue
        if not is_valid_index(skins, skin_index):
            warnings.append(f"Node {node_index} references missing skin {skin_index}; skipped convert.")
            continue

        skin = skins[skin_index]
        skin_joints = skin.get("joints", [])
        fallback_bone = resolve_bone_node(
            fallback_joint_for_skin(skin, skin_joints, parent_map), bone_resolution.resolved_bones
        )
        node_world = world_matrices.get(node_index, identity_matrix())

        for primitive_index, primitive in enumerate(meshes[mesh_index].get("primitives", [])):
            collect_primitive_cuboids(
                gltf,
                input_path,
                binary_chunk,
                buffer_cache,
                accessors,
                primitive,
                node_world,
                node_index,
                mesh_index,
                primitive_index,
                skin_joints,
                fallback_bone,
                bone_resolution.resolved_bones,
                vrm_humanoid_nodes,
                gltf.get("materials", []),
                config.complex_split,
                scratch_report,
                bones,
                accumulators,
                regular_faces_by_part,
                split_faces,
            )

    complex_split_report, face_feature_protection_actions = apply_complex_split(
        split_faces,
        accumulators,
        config.complex_split,
        config.face_feature_protection,
        bones,
        vrm_humanoid_nodes,
    )
    cleanup_protected_part_keys: set[PartKey] = set()
    complex_split_report.extend(
        apply_regular_detail_split(
            accumulators,
            regular_faces_by_part,
            bones,
            mode,
            config.hybrid_detail_split,
            cleanup_protected_part_keys,
        )
    )
    face_feature_protection_actions.extend(
        apply_explicit_face_feature_visibility_protection(
            accumulators,
            bones,
            config.face_feature_protection,
        )
    )
    complex_split_report.extend(apply_auto_spatial_split(accumulators, regular_faces_by_part, bones, mode))
    cleanup_report = apply_cleanup(
        accumulators,
        bones,
        config.cleanup,
        warnings,
        regular_faces_by_part,
        cleanup_protected_part_keys,
    )
    populated = {
        part_key: accumulator
        for part_key, accumulator in accumulators.items()
        if accumulator.min_xyz is not None and accumulator.max_xyz is not None and accumulator.faces > 0
    }
    if not populated:
        raise ConvertError("no triangle faces could be converted into cuboids")

    model_min, model_max = combined_bbox(populated.values())
    scale, offset = compute_scale_and_offset(model_min, model_max, target_height, warnings)
    cuboids, oriented_cube_report, orientation_decisions = build_cuboids(
        populated,
        bones,
        world_matrices,
        scale,
        offset,
        config.oriented_cubes,
        vrm_humanoid_nodes,
        auto_orient=mode == "hybrid",
    )
    model = build_bbmodel(input_path.stem, bones, world_matrices, cuboids, scale, offset)
    populated_bones = {part_key.owner_bone for part_key in populated}
    empty_bones = sum(1 for bone_index in bones if bone_index not in populated_bones)
    small_cubes = count_small_cubes(cuboids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = ConvertResult(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        preset=config.preset,
        scale=scale,
        cubes=cuboids,
        bone_resolution=bone_resolution,
        empty_bones=empty_bones,
        small_cubes=small_cubes,
        complex_split=complex_split_report,
        cleanup=cleanup_report,
        oriented_cubes=oriented_cube_report,
        hybrid=hybrid_report,
        hybrid_detail_split=config.hybrid_detail_split,
        unskinned_meshes=config.unskinned_meshes,
        face_feature_protection=config.face_feature_protection,
        face_feature_protection_actions=face_feature_protection_actions,
        assigned_unskinned_meshes=assigned_unskinned_meshes,
        skipped_unskinned_meshes=skipped_unskinned_meshes,
        warnings=warnings,
        orientation_decisions=orientation_decisions,
    )
    if report_path is not None:
        write_convert_report(result, report_path)
    return result


def collect_primitive_cuboids(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessors: list[dict[str, Any]],
    primitive: dict[str, Any],
    node_world: list[list[float]],
    node_index: int,
    mesh_index: int,
    primitive_index: int,
    skin_joints: list[int],
    fallback_bone: int | None,
    resolved_bones: dict[int, int],
    vrm_humanoid_nodes: dict[str, set[int]],
    materials: list[dict[str, Any]],
    complex_split: ComplexSplitConfig,
    report: PartitionReport,
    bones: dict[int, BonePartition],
    accumulators: dict[PartKey, BBoxAccumulator],
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
    split_faces: list[SplitFace],
) -> None:
    attributes = primitive.get("attributes", {})
    position_accessor = attributes.get("POSITION")
    if not is_valid_index(accessors, position_accessor):
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} has no valid POSITION accessor; skipped convert."
        )
        return

    vertex_count = int(accessors[position_accessor].get("count", 0))
    mode = int(primitive.get("mode", MODE_TRIANGLES))
    if mode not in {MODE_TRIANGLES, MODE_TRIANGLE_STRIP, MODE_TRIANGLE_FAN}:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} uses non-triangle mode {mode}; skipped convert."
        )
        return

    try:
        raw_positions = read_accessor(gltf, path, binary_chunk, buffer_cache, position_accessor)
    except InspectError as exc:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} could not read POSITION accessor "
            f"{position_accessor}: {exc}; skipped convert."
        )
        return

    if raw_positions is None:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} POSITION accessor has no buffer data; skipped convert."
        )
        return


    positions = [transform_point(node_world, row_to_vec3(row)) for row in raw_positions[:vertex_count]]
    faces = read_faces(
        gltf,
        path,
        binary_chunk,
        buffer_cache,
        accessors,
        primitive,
        vertex_count,
        report,
        mesh_index,
        primitive_index,
    )
    if not faces:
        return

    joints = read_optional_accessor(
        gltf,
        path,
        binary_chunk,
        buffer_cache,
        attributes.get("JOINTS_0"),
        "JOINTS_0",
        report,
        mesh_index,
        primitive_index,
    )
    weights = read_optional_accessor(
        gltf,
        path,
        binary_chunk,
        buffer_cache,
        attributes.get("WEIGHTS_0"),
        "WEIGHTS_0",
        report,
        mesh_index,
        primitive_index,
    )

    primitive_key = f"node={node_index}/mesh={mesh_index}/primitive={primitive_index}"
    material_name = primitive_material_name(materials, primitive.get("material"))
    for face in faces:
        owner, _fallback_used, _invalid_refs, _invalid_face_indices = choose_face_owner(
            face, joints, weights, skin_joints, fallback_bone, vertex_count, resolved_bones
        )
        if owner is None:
            continue
        if owner not in bones:
            report.warnings.append(
                f"Mesh {mesh_index} primitive {primitive_index} assigned face to joint node {owner}, "
                "but that node is not present in the skeleton."
            )
            continue

        bone = bones[owner]
        points: list[list[float]] = []
        vertex_keys: list[tuple[str, int]] = []
        for vertex_index in face:
            if 0 <= vertex_index < len(positions):
                points.append(positions[vertex_index])
                vertex_keys.append((primitive_key, vertex_index))
        split_face = SplitFace(
            owner_bone=owner,
            bone_name=bone.name,
            points=points,
            vertex_keys=vertex_keys,
            material_name=material_name,
        )
        if should_complex_split_bone(owner, bone.name, complex_split, vrm_humanoid_nodes):
            split_faces.append(split_face)
            continue

        part_key = PartKey(owner, bone.name)
        regular_faces_by_part.setdefault(part_key, []).append(split_face)
        accumulators.setdefault(part_key, BBoxAccumulator()).add_face(points, vertex_keys)


def collect_unskinned_primitive_cuboids(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessors: list[dict[str, Any]],
    primitive: dict[str, Any],
    node_world: list[list[float]],
    node_index: int,
    mesh_index: int,
    primitive_index: int,
    materials: list[dict[str, Any]],
    unskinned_meshes: UnskinnedMeshesConfig,
    parent_map: dict[int, int],
    resolved_bones: dict[int, int],
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
    report: PartitionReport,
    accumulators: dict[PartKey, BBoxAccumulator],
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
) -> AssignedUnskinnedMesh | None:
    attributes = primitive.get("attributes", {})
    position_accessor = attributes.get("POSITION")
    if not is_valid_index(accessors, position_accessor):
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} has no valid POSITION accessor; skipped convert."
        )
        return None

    vertex_count = int(accessors[position_accessor].get("count", 0))
    mode = int(primitive.get("mode", MODE_TRIANGLES))
    if mode not in {MODE_TRIANGLES, MODE_TRIANGLE_STRIP, MODE_TRIANGLE_FAN}:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} uses non-triangle mode {mode}; skipped convert."
        )
        return None

    try:
        raw_positions = read_accessor(gltf, path, binary_chunk, buffer_cache, position_accessor)
    except InspectError as exc:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} could not read POSITION accessor "
            f"{position_accessor}: {exc}; skipped convert."
        )
        return None

    if raw_positions is None:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} POSITION accessor has no buffer data; skipped convert."
        )
        return None

    positions = [transform_point(node_world, row_to_vec3(row)) for row in raw_positions[:vertex_count]]
    faces = read_faces(
        gltf,
        path,
        binary_chunk,
        buffer_cache,
        accessors,
        primitive,
        vertex_count,
        report,
        mesh_index,
        primitive_index,
    )
    if not faces:
        return None

    owner, reason = choose_unskinned_mesh_owner(
        unskinned_meshes.strategy,
        node_index,
        positions,
        parent_map,
        resolved_bones,
        bones,
        world_matrices,
    )
    if owner is None or reason is None:
        report.warnings.append(
            f"Node {node_index} mesh {mesh_index} primitive {primitive_index} has no skin and could not be "
            f"assigned by unskinned_meshes.strategy={unskinned_meshes.strategy!r}; skipped convert."
        )
        return None

    bone = bones[owner]
    part_name = f"{bone.name}_unskinned_{node_index}_{mesh_index}_{primitive_index}"
    part_key = PartKey(owner, part_name)
    accumulator = accumulators.setdefault(part_key, BBoxAccumulator())
    primitive_key = f"unskinned-node={node_index}/mesh={mesh_index}/primitive={primitive_index}"
    material_name = primitive_material_name(materials, primitive.get("material"))
    assigned_faces = 0
    assigned_vertices: set[tuple[str, int]] = set()
    for face in faces:
        points: list[list[float]] = []
        vertex_keys: list[tuple[str, int]] = []
        for vertex_index in face:
            if 0 <= vertex_index < len(positions):
                points.append(positions[vertex_index])
                vertex_key = (primitive_key, vertex_index)
                vertex_keys.append(vertex_key)
                assigned_vertices.add(vertex_key)
        if not points:
            continue
        split_face = SplitFace(
            owner_bone=owner,
            bone_name=bone.name,
            points=points,
            vertex_keys=vertex_keys,
            material_name=material_name,
        )
        regular_faces_by_part.setdefault(part_key, []).append(split_face)
        accumulator.add_face(points, vertex_keys)
        assigned_faces += 1

    if assigned_faces == 0:
        return None
    return AssignedUnskinnedMesh(
        node_index=node_index,
        mesh_index=mesh_index,
        primitive_index=primitive_index,
        owner_bone=owner,
        owner_bone_name=bone.name,
        part_name=part_name,
        strategy=unskinned_meshes.strategy,
        reason=reason,
        faces=assigned_faces,
        vertices=len(assigned_vertices),
    )


def choose_unskinned_mesh_owner(
    strategy: str,
    node_index: int,
    positions: list[list[float]],
    parent_map: dict[int, int],
    resolved_bones: dict[int, int],
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
) -> tuple[int | None, str | None]:
    if strategy in {"node_parent", "node_parent_then_nearest"}:
        owner = nearest_resolved_ancestor_bone(node_index, parent_map, resolved_bones, bones)
        if owner is not None:
            return owner, "node_parent"
    if strategy in {"nearest_bone", "node_parent_then_nearest"}:
        owner = nearest_bone_to_points(positions, bones, world_matrices)
        if owner is not None:
            return owner, "nearest_bone"
    return None, None


def nearest_resolved_ancestor_bone(
    node_index: int,
    parent_map: dict[int, int],
    resolved_bones: dict[int, int],
    bones: dict[int, BonePartition],
) -> int | None:
    current: int | None = node_index
    seen: set[int] = set()
    while current is not None and current not in seen:
        seen.add(current)
        owner = resolved_bones.get(current)
        if owner in bones:
            return owner
        current = parent_map.get(current)
    return None


def nearest_bone_to_points(
    positions: list[list[float]],
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
) -> int | None:
    if not positions or not bones:
        return None
    min_xyz, max_xyz = points_bbox(positions)
    center = [(min_xyz[index] + max_xyz[index]) * 0.5 for index in range(3)]
    return min(
        bones,
        key=lambda bone_index: (
            squared_distance(center, matrix_translation(world_matrices.get(bone_index, identity_matrix()))),
            bone_index,
        ),
    )


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((left[index] - right[index]) ** 2 for index in range(3))


def apply_complex_split_override(
    config: ProcessingConfig, requested_bones: tuple[str, ...] | list[str]
) -> ProcessingConfig:
    bones = tuple(item for item in requested_bones if item)
    if not bones:
        return replace(config, complex_split=ComplexSplitConfig(enabled=False))
    return replace(config, complex_split=ComplexSplitConfig(enabled=True, bones=bones))


def apply_mode_defaults(config: ProcessingConfig, mode: str) -> tuple[ProcessingConfig, HybridModeReport]:
    if mode != "hybrid":
        return config, HybridModeReport()

    configured_bones = config.complex_split.bones if config.complex_split.enabled else ()
    special_cube_bones = dedupe_names((*HYBRID_SPECIAL_CUBE_BONES, *configured_bones))
    complex_split = replace(config.complex_split, enabled=True, bones=special_cube_bones)
    return replace(config, complex_split=complex_split), HybridModeReport(
        enabled=True,
        special_cube_bones=special_cube_bones,
        mesh_strategy="special_cubes",
        cuboid_strategy="one_bbox_for_non_complex_resolved_bones",
    )


def dedupe_names(names: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        result.append(name)
    return tuple(result)


def primitive_material_name(materials: list[dict[str, Any]], material_index: Any) -> str | None:
    if not is_valid_index(materials, material_index):
        return None
    name = materials[material_index].get("name")
    return name if isinstance(name, str) and name else None


def ignored_unskinned_primitive_reason(
    config: UnskinnedMeshesConfig,
    node_name: str | None,
    mesh_name: str | None,
    material_name: str | None,
) -> str | None:
    reason = unskinned_name_match_reason(
        material_name,
        config.ignore_material_name_contains,
        config.case_sensitive,
        "material_name_contains",
    )
    if reason is not None:
        return reason
    reason = unskinned_name_match_reason(
        node_name,
        config.ignore_node_name_contains,
        config.case_sensitive,
        "node_name_contains",
    )
    if reason is not None:
        return reason
    return unskinned_name_match_reason(
        mesh_name,
        config.ignore_mesh_name_contains,
        config.case_sensitive,
        "mesh_name_contains",
    )


def unskinned_name_match_reason(
    name: str | None,
    patterns: tuple[str, ...],
    case_sensitive: bool,
    reason_prefix: str,
) -> str | None:
    if name is None:
        return None
    haystack = name if case_sensitive else name.casefold()
    for pattern in patterns:
        needle = pattern if case_sensitive else pattern.casefold()
        if needle and needle in haystack:
            return f"ignored_{reason_prefix}:{pattern}"
    return None


def extract_vrm_humanoid_nodes(gltf: dict[str, Any], resolved_bones: dict[int, int]) -> dict[str, set[int]]:
    vrm = gltf.get("extensions", {}).get("VRMC_vrm")
    if not isinstance(vrm, dict):
        return {}
    humanoid = vrm.get("humanoid")
    if not isinstance(humanoid, dict):
        return {}
    human_bones = humanoid.get("humanBones")
    if not isinstance(human_bones, dict):
        return {}

    nodes: dict[str, set[int]] = {}
    for humanoid_name, bone_info in human_bones.items():
        if not isinstance(humanoid_name, str) or not isinstance(bone_info, dict):
            continue
        node_index = bone_info.get("node")
        if not isinstance(node_index, int):
            continue
        targets = {node_index}
        resolved = resolved_bones.get(node_index)
        if resolved is not None:
            targets.add(resolved)
        nodes.setdefault(humanoid_name, set()).update(targets)
    return nodes


def should_complex_split_bone(
    node_index: int,
    name: str,
    config: ComplexSplitConfig,
    vrm_humanoid_nodes: dict[str, set[int]],
) -> bool:
    if not config.enabled:
        return False
    requested = config.bones or (DEFAULT_COMPLEX_SPLIT_BONE,)
    haystack = name if config.case_sensitive else name.casefold()
    for bone_name in requested:
        if node_index in vrm_humanoid_nodes.get(bone_name, set()):
            return True
        aliases = COMPLEX_BONE_ALIASES.get(bone_name, (bone_name,))
        for alias in aliases:
            needle = alias if config.case_sensitive else alias.casefold()
            if needle and needle in haystack:
                return True
    return False


def apply_complex_split(
    split_faces: list[SplitFace],
    accumulators: dict[PartKey, BBoxAccumulator],
    config: ComplexSplitConfig,
    face_feature_protection: FaceFeatureProtectionConfig,
    bones: dict[int, BonePartition],
    vrm_humanoid_nodes: dict[str, set[int]],
) -> tuple[list[ComplexSplitBoneReport], list[FaceFeatureProtectionAction]]:
    faces_by_bone: dict[int, list[SplitFace]] = {}
    for face in split_faces:
        if face.points:
            faces_by_bone.setdefault(face.owner_bone, []).append(face)

    reports: list[ComplexSplitBoneReport] = []
    protection_actions: list[FaceFeatureProtectionAction] = []
    for owner_bone, faces in sorted(faces_by_bone.items()):
        if is_head_complex_split_bone(owner_bone, faces[0].bone_name, config, vrm_humanoid_nodes):
            report, actions = apply_head_complex_split(
                owner_bone,
                faces,
                accumulators,
                config,
                face_feature_protection,
                bones,
            )
            reports.append(report)
            protection_actions.extend(actions)
        else:
            reports.append(apply_generic_complex_split(owner_bone, faces, accumulators, config))
    return reports, protection_actions


def is_head_complex_split_bone(
    node_index: int,
    name: str,
    config: ComplexSplitConfig,
    vrm_humanoid_nodes: dict[str, set[int]],
) -> bool:
    if node_index in vrm_humanoid_nodes.get("head", set()):
        return True
    haystack = name if config.case_sensitive else name.casefold()
    for alias in COMPLEX_BONE_ALIASES["head"]:
        needle = alias if config.case_sensitive else alias.casefold()
        if needle and needle in haystack:
            return True
    return False


def apply_head_complex_split(
    owner_bone: int,
    faces: list[SplitFace],
    accumulators: dict[PartKey, BBoxAccumulator],
    config: ComplexSplitConfig,
    face_feature_protection: FaceFeatureProtectionConfig,
    bones: dict[int, BonePartition],
) -> tuple[ComplexSplitBoneReport, list[FaceFeatureProtectionAction]]:
    owner_bbox = faces_bbox(faces)
    front_sign = infer_head_front_sign(faces, owner_bbox)
    if config.connected_components.enabled:
        component_indices = connected_face_components(faces)
    else:
        component_indices = [list(range(len(faces)))]
    component_indices.sort(key=lambda component: (-len(component), component[0] if component else 0))
    component_indices, tiny_faces_by_target, merged_tiny_components, deleted_tiny_components = apply_tiny_component_rules(
        component_indices,
        faces,
        config,
    )
    explicit_face_feature_bbox = None
    explicit_face_feature_protection_bbox = None
    if face_feature_protection.enabled:
        explicit_face_feature_bbox = explicit_face_feature_accumulators_bbox(
            accumulators,
            owner_bone,
            owner_bbox,
            face_feature_protection,
        )
        explicit_face_feature_protection_bbox = explicit_face_feature_visibility_bbox(
            accumulators,
            owner_bone,
            owner_bbox,
            front_sign,
            face_feature_protection,
        ) or explicit_face_feature_bbox
    suppress_material_face_features = explicit_face_feature_bbox is not None
    largest_component = tuple(component_indices[0]) if component_indices else ()
    report_accumulators: dict[str, BBoxAccumulator] = {}
    report_methods: dict[str, str] = {}
    pending_hair_parts: dict[str, list[tuple[str, list[SplitFace]]]] = {}
    pending_accessory_parts: dict[str, list[tuple[str, list[SplitFace]]]] = {}
    head_core_faces: list[SplitFace] = []
    merged_tiny_hair_buckets = 0
    expanded_hair_bucket_overlap = 0

    def add_to_part(part_name: str, method: str, part_faces: list[SplitFace]) -> BBoxAccumulator:
        accumulator = accumulators.setdefault(PartKey(owner_bone, part_name), BBoxAccumulator())
        accumulator.is_complex_split = True
        report_accumulator = report_accumulators.setdefault(part_name, BBoxAccumulator())
        for face in part_faces:
            accumulator.add_face(face.points, face.vertex_keys)
            report_accumulator.add_face(face.points, face.vertex_keys)
        existing_method = report_methods.get(part_name)
        if existing_method is None:
            report_methods[part_name] = method
        elif existing_method != method:
            report_methods[part_name] = "mixed"
        if part_name == "head_core":
            head_core_faces.extend(part_faces)
        return accumulator

    def add_hair_to_parts(part_name: str, method: str, part_faces: list[SplitFace]) -> None:
        if part_faces:
            pending_hair_parts.setdefault(part_name, []).append((method, part_faces))

    def add_accessory_to_parts(part_name: str, method: str, part_faces: list[SplitFace]) -> None:
        if part_faces:
            pending_accessory_parts.setdefault(part_name, []).append((method, part_faces))

    def flush_hair_parts() -> None:
        nonlocal merged_tiny_hair_buckets, expanded_hair_bucket_overlap
        for part_name, chunks in sorted(pending_hair_parts.items()):
            if len(chunks) <= HEAD_HAIR_COMPONENT_PRESERVE_LIMIT:
                hair_chunks = chunks
            else:
                methods = sorted({method for method, _part_faces in chunks})
                merged_method = methods[0] if len(methods) == 1 else "mixed"
                merged_faces = [face for _method, part_faces in chunks for face in part_faces]
                hair_chunks = [(merged_method, merged_faces)]

            for method, part_faces in hair_chunks:
                split_result = split_hair_part_faces(part_name, part_faces, owner_bbox, front_sign)
                merged_tiny_hair_buckets += split_result.merged_tiny_buckets
                for split_part in split_result.parts:
                    unique_name = unique_part_name(split_part.name, report_accumulators)
                    part_accumulator = add_to_part(unique_name, method, split_part.faces)
                    if split_part.split_axes and expand_hair_bucket_accumulator(part_accumulator, split_part.split_axes):
                        expanded_hair_bucket_overlap += 1

    def flush_accessory_parts() -> None:
        for part_name, chunks in sorted(pending_accessory_parts.items()):
            methods = sorted({method for method, _part_faces in chunks})
            merged_method = methods[0] if len(methods) == 1 else "mixed"
            merged_faces = [face for _method, part_faces in chunks for face in part_faces]
            for split_part_name, split_faces in split_head_accessory_part_faces(part_name, merged_faces, owner_bbox):
                add_to_part(unique_part_name(split_part_name, report_accumulators), merged_method, split_faces)

    for component in component_indices:
        component_key = tuple(component)
        component_faces = [faces[index] for index in component]
        merged_faces = tiny_faces_by_target.get(component_key, [])
        merge_part = (
            "head_core"
            if component_key == largest_component
            else classify_head_component_spatial(component_faces, owner_bbox, front_sign)
        )
        material_faces: dict[str, list[SplitFace]] = {}
        unclassified_faces: list[SplitFace] = []
        for face in component_faces:
            material_part = classify_head_material([face])
            if material_part is None:
                unclassified_faces.append(face)
            else:
                material_faces.setdefault(material_part, []).append(face)

        if material_faces:
            for material_part, part_faces in sorted(material_faces.items()):
                if suppress_material_face_features and material_part in FACE_FEATURE_PART_PREFIXES:
                    continue
                if material_part == "hair":
                    hair_faces_by_part: dict[str, list[SplitFace]] = {}
                    for face in part_faces:
                        hair_faces_by_part.setdefault(classify_hair_face(face, owner_bbox, front_sign), []).append(face)
                    for part_name, hair_faces in sorted(hair_faces_by_part.items()):
                        add_hair_to_parts(part_name, "spatial_region", hair_faces)
                elif material_part in {"eye", "brow", "eyelash", "ear"}:
                    feature_faces_by_part: dict[str, list[SplitFace]] = {}
                    for face in part_faces:
                        feature_faces_by_part.setdefault(
                            classify_lateral_feature_face(face, owner_bbox, material_part),
                            [],
                        ).append(face)
                    for part_name, feature_faces in sorted(feature_faces_by_part.items()):
                        add_to_part(part_name, "material_name", feature_faces)
                elif material_part == "head_accessory":
                    add_accessory_to_parts("head_accessory", "material_name", part_faces)
                else:
                    add_to_part(material_part, "material_name", part_faces)
            if unclassified_faces:
                fallback_part = (
                    "head_core"
                    if component_key == largest_component
                    else classify_head_component_spatial(unclassified_faces, owner_bbox, front_sign)
                )
                if is_hair_part(fallback_part):
                    add_hair_to_parts(fallback_part, "connected_component", unclassified_faces)
                elif fallback_part == "head_accessory":
                    add_accessory_to_parts(fallback_part, "connected_component", unclassified_faces)
                else:
                    add_to_part(fallback_part, "connected_component", unclassified_faces)
            if merged_faces:
                if is_hair_part(merge_part):
                    add_hair_to_parts(merge_part, "connected_component_merge", merged_faces)
                elif merge_part == "head_accessory":
                    add_accessory_to_parts(merge_part, "connected_component_merge", merged_faces)
                else:
                    add_to_part(merge_part, "connected_component_merge", merged_faces)
            continue

        if component_key == largest_component:
            add_to_part("head_core", "connected_component", component_faces)
        else:
            fallback_part = classify_head_component_spatial(component_faces, owner_bbox, front_sign)
            if is_hair_part(fallback_part):
                add_hair_to_parts(fallback_part, "connected_component", component_faces)
            elif fallback_part == "head_accessory":
                add_accessory_to_parts(fallback_part, "connected_component", component_faces)
            else:
                add_to_part(fallback_part, "connected_component", component_faces)
        if merged_faces:
            if is_hair_part(merge_part):
                add_hair_to_parts(merge_part, "connected_component_merge", merged_faces)
            elif merge_part == "head_accessory":
                add_accessory_to_parts(merge_part, "connected_component_merge", merged_faces)
            else:
                add_to_part(merge_part, "connected_component_merge", merged_faces)

    flush_hair_parts()
    flush_accessory_parts()
    head_core_front_sign = (
        infer_face_feature_front_sign(owner_bbox, explicit_face_feature_bbox)
        if explicit_face_feature_bbox is not None
        else front_sign
    )
    split_head_core_parts(
        owner_bone,
        head_core_faces,
        accumulators,
        report_accumulators,
        report_methods,
        owner_bbox,
        head_core_front_sign,
    )
    if explicit_face_feature_protection_bbox is not None:
        protected_feature_names = explicit_face_feature_names(
            accumulators,
            owner_bone,
            owner_bbox,
            face_feature_protection,
        )
        protection_actions = protect_explicit_face_feature_visibility(
            owner_bone,
            accumulators,
            report_accumulators,
            explicit_face_feature_protection_bbox,
            owner_bbox,
            head_core_front_sign,
            face_feature_protection,
            bones,
            protected_feature_names,
        )
    else:
        protection_actions = []

    subparts = [
        ComplexSplitSubpartReport(
            name=name,
            method=report_methods[name],
            faces=accumulator.faces,
            vertices=len(accumulator.vertices),
        )
        for name, accumulator in sorted(report_accumulators.items())
    ]
    return ComplexSplitBoneReport(
        bone=owner_bone,
        bone_name=faces[0].bone_name if faces else f"bone_{owner_bone}",
        source_faces=len(faces),
        subparts=subparts,
        merged_tiny_components=merged_tiny_components,
        deleted_tiny_components=deleted_tiny_components,
        merged_tiny_hair_buckets=merged_tiny_hair_buckets,
        expanded_hair_bucket_overlap=expanded_hair_bucket_overlap,
    ), protection_actions


def apply_generic_complex_split(
    owner_bone: int,
    faces: list[SplitFace],
    accumulators: dict[PartKey, BBoxAccumulator],
    config: ComplexSplitConfig,
) -> ComplexSplitBoneReport:
    owner_bbox = faces_bbox(faces)
    if config.connected_components.enabled:
        component_indices = connected_face_components(faces)
    else:
        component_indices = [list(range(len(faces)))]
    component_indices.sort(key=lambda component: (-len(component), component[0] if component else 0))
    component_indices, tiny_faces_by_target, merged_tiny_components, deleted_tiny_components = apply_tiny_component_rules(
        component_indices,
        faces,
        config,
    )

    report_accumulators: dict[str, BBoxAccumulator] = {}
    report_methods: dict[str, str] = {}
    bone_name = faces[0].bone_name if faces else f"bone_{owner_bone}"
    owner_min, owner_max = owner_bbox
    owner_height = max(owner_max[1] - owner_min[1], EPSILON)
    owner_volume = bbox_volume(owner_min, owner_max)

    def add_to_part(part_name: str, method: str, part_faces: list[SplitFace]) -> None:
        accumulator = accumulators.setdefault(PartKey(owner_bone, part_name), BBoxAccumulator())
        accumulator.is_complex_split = True
        report_accumulator = report_accumulators.setdefault(part_name, BBoxAccumulator())
        for face in part_faces:
            accumulator.add_face(face.points, face.vertex_keys)
            report_accumulator.add_face(face.points, face.vertex_keys)
        existing_method = report_methods.get(part_name)
        if existing_method is None:
            report_methods[part_name] = method
        elif existing_method != method:
            report_methods[part_name] = "mixed"

    for component_index, component in enumerate(component_indices, start=1):
        component_faces = [faces[index] for index in component]
        merged_faces = tiny_faces_by_target.get(tuple(component), [])
        part_name = unique_part_name(
            generic_component_part_name(bone_name, component_faces, owner_bbox, component_index),
            report_accumulators,
        )
        component_accumulator = accumulator_from_faces(component_faces)
        split_parts = []
        requested_split_parts: list[AutoSpatialPart] = []
        capped_to_budget = False
        budget_limit = AUTO_SPATIAL_SPLIT_OWNER_BUDGET_MULTIPLIER
        if (
            should_generic_complex_auto_spatial_split(bone_name)
            and len(component_faces) >= AUTO_SPATIAL_SPLIT_MIN_FACES
            and should_auto_spatial_split(component_accumulator, owner_height, owner_volume)
        ):
            requested_split_parts = generic_complex_auto_spatial_split_faces(
                part_name,
                component_faces,
                component_accumulator,
                owner_height,
                generic_complex_auto_spatial_axis_limit(bone_name),
            )
            split_parts, capped_to_budget = cap_auto_spatial_parts(requested_split_parts, budget_limit)
        if len(split_parts) >= 2:
            for split_part in split_parts:
                split_part_name = unique_part_name(split_part.name, report_accumulators)
                add_to_part(split_part_name, "auto_spatial_grid", split_part.faces)
        else:
            add_to_part(part_name, "connected_component", component_faces)
        if merged_faces:
            add_to_part(part_name, "connected_component_merge", merged_faces)

    subparts = [
        ComplexSplitSubpartReport(
            name=name,
            method=report_methods[name],
            faces=accumulator.faces,
            vertices=len(accumulator.vertices),
        )
        for name, accumulator in sorted(report_accumulators.items())
    ]
    return ComplexSplitBoneReport(
        bone=owner_bone,
        bone_name=bone_name,
        source_faces=len(faces),
        subparts=subparts,
        merged_tiny_components=merged_tiny_components,
        deleted_tiny_components=deleted_tiny_components,
    )


def should_generic_complex_auto_spatial_split(bone_name: str) -> bool:
    normalized = bone_name.casefold()
    if "hair" in normalized or "髪" in bone_name or "髮" in bone_name:
        return False
    return any(
        token in normalized
        for token in (
            "hood",
            "string",
            "spine",
            "chest",
            "upperarm",
            "lowerarm",
            "shoulder",
            "sleeve",
            "arm",
        )
    )


def generic_complex_auto_spatial_axis_limit(bone_name: str) -> int:
    normalized = bone_name.casefold()
    if any(token in normalized for token in ("spine", "chest", "shoulder", "hood")):
        return 2
    return 1


def apply_regular_detail_split(
    accumulators: dict[PartKey, BBoxAccumulator],
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
    bones: dict[int, BonePartition],
    mode: str,
    detail_split: HybridDetailSplitConfig,
    cleanup_protected_part_keys: set[PartKey] | None = None,
) -> list[ComplexSplitBoneReport]:
    if mode != "hybrid" or not detail_split.enabled:
        return []

    populated = [
        accumulator
        for accumulator in accumulators.values()
        if accumulator.min_xyz is not None and accumulator.max_xyz is not None and accumulator.faces > 0
    ]
    if not populated:
        return []
    model_min, model_max = combined_bbox(populated)
    model_height = max(model_max[1] - model_min[1], EPSILON)

    reports: list[ComplexSplitBoneReport] = []
    for part_key, faces in sorted(list(regular_faces_by_part.items()), key=lambda item: (item[0].owner_bone, item[0].name)):
        accumulator = accumulators.get(part_key)
        if accumulator is None or accumulator.is_complex_split or len(faces) < detail_split.min_faces:
            continue
        if accumulator.min_xyz is None or accumulator.max_xyz is None:
            continue
        if not should_regular_detail_split_accumulator(accumulator, model_height, detail_split):
            continue

        split_specs = regular_detail_split_specs(part_key.name, faces, detail_split)
        if len(split_specs) < 2:
            continue

        del accumulators[part_key]
        regular_faces_by_part.pop(part_key, None)
        used_names: dict[str, BBoxAccumulator] = {}
        subparts: list[ComplexSplitSubpartReport] = []
        for base_name, method, part_faces in split_specs:
            if not part_faces:
                continue
            part_name = unique_part_name(base_name, used_names)
            part_accumulator = accumulator_from_split_faces(part_faces)
            part_accumulator.is_complex_split = True
            new_key = PartKey(part_key.owner_bone, part_name)
            accumulators[new_key] = part_accumulator
            regular_faces_by_part[new_key] = part_faces
            if cleanup_protected_part_keys is not None and is_significant_regular_detail_method(method, part_name):
                cleanup_protected_part_keys.add(new_key)
            used_names[part_name] = part_accumulator
            subparts.append(
                ComplexSplitSubpartReport(
                    name=part_name,
                    method=method,
                    faces=part_accumulator.faces,
                    vertices=len(part_accumulator.vertices),
                )
            )

        if not subparts:
            continue
        bone = bones.get(part_key.owner_bone)
        budget_limit = AUTO_SPATIAL_SPLIT_OWNER_BUDGET_MULTIPLIER
        budget_status = "within_budget" if len(split_specs) <= budget_limit else "over_budget_warning"
        reports.append(
            ComplexSplitBoneReport(
                bone=part_key.owner_bone,
                bone_name=bone.name if bone is not None else part_key.name,
                source_faces=accumulator.faces,
                subparts=sorted(subparts, key=lambda item: item.name),
                original_cube_dimensions=bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz),
                split_method="regular_detail_split",
                requested_subpart_count=len(split_specs),
                budget_limit=budget_limit,
                budget_status=budget_status,
                budget_reason="regular_detail_split_exceeds_owner_budget" if budget_status == "over_budget_warning" else None,
            )
        )
    return reports


def is_significant_regular_detail_method(method: str, part_name: str) -> bool:
    if part_name.endswith("_misc"):
        return False
    return method in {"regular_material_component", "regular_connected_component", "regular_spatial_detail"}


def should_regular_detail_split_accumulator(
    accumulator: BBoxAccumulator,
    model_height: float,
    detail_split: HybridDetailSplitConfig,
) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    dimensions = bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz)
    return max(dimensions) / model_height <= detail_split.max_long_dim_ratio


def regular_detail_split_specs(
    part_name: str,
    faces: list[SplitFace],
    detail_split: HybridDetailSplitConfig,
) -> list[tuple[str, str, list[SplitFace]]]:
    sanitized_name = sanitize_part_name(part_name)
    material_specs = regular_material_split_specs(part_name, faces, detail_split) if detail_split.by_material else []
    if len(material_specs) >= 2:
        result: list[tuple[str, str, list[SplitFace]]] = []
        for material_name, material_faces in material_specs:
            if detail_split.by_connected_component:
                result.extend(
                    split_regular_faces_by_components(
                        material_name,
                        material_faces,
                        "regular_material_component",
                        detail_split,
                    )
                )
            else:
                result.append((material_name, "regular_material", material_faces))
        return result
    if not detail_split.by_connected_component:
        return [(sanitized_name, "regular_detail", faces)]

    component_specs: list[tuple[str, str, list[SplitFace]]] = []
    if should_split_single_material_components(part_name, faces):
        component_specs = split_regular_faces_by_components(
            sanitized_name,
            faces,
            "regular_connected_component",
            detail_split,
        )
        if len(component_specs) >= 2:
            return refine_regular_component_specs_by_spatial_detail(component_specs, detail_split)

    spatial_specs = split_regular_faces_by_spatial_detail(sanitized_name, faces, detail_split)
    if len(spatial_specs) >= 2:
        return spatial_specs
    return component_specs or [(sanitized_name, "regular_detail", faces)]


def should_split_single_material_components(part_name: str, faces: list[SplitFace]) -> bool:
    if any(matches_complex_alias(part_name, alias) for alias in ("hair", "skirt", "coat", "accessory")):
        return True
    material_part = classify_generic_material(faces)
    return material_part in {"hair", "head_accessory", "ear"}


def regular_material_split_specs(
    part_name: str,
    faces: list[SplitFace],
    detail_split: HybridDetailSplitConfig,
) -> list[tuple[str, list[SplitFace]]]:
    faces_by_material: dict[str, list[SplitFace]] = {}
    unnamed_faces: list[SplitFace] = []
    for face in faces:
        if face.material_name:
            faces_by_material.setdefault(face.material_name, []).append(face)
        else:
            unnamed_faces.append(face)

    if len(faces_by_material) < 2:
        return []

    min_faces = max(
        detail_split.min_material_faces,
        math.ceil(len(faces) * detail_split.min_material_ratio),
    )
    significant: list[tuple[str, list[SplitFace]]] = []
    remainder: list[SplitFace] = list(unnamed_faces)
    for material, material_faces in sorted(faces_by_material.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(material_faces) >= min_faces:
            significant.append((f"{sanitize_part_name(part_name)}_{sanitize_part_name(material)}", material_faces))
        else:
            remainder.extend(material_faces)
    if len(significant) < 2:
        return []
    if remainder:
        significant.append((f"{sanitize_part_name(part_name)}_misc", remainder))
    return significant


def split_regular_faces_by_components(
    base_name: str,
    faces: list[SplitFace],
    method: str,
    detail_split: HybridDetailSplitConfig,
) -> list[tuple[str, str, list[SplitFace]]]:
    if len(faces) < 2:
        return [(base_name, method, faces)]

    component_indices = connected_cleanup_face_components(faces)
    if len(component_indices) < 2:
        return [(base_name, method, faces)]

    min_faces = max(
        detail_split.min_component_faces,
        math.ceil(len(faces) * detail_split.min_component_ratio),
    )
    large_components = [component for component in component_indices if len(component) >= min_faces]
    if len(large_components) < 2:
        return [(base_name, method, faces)]

    specs: list[tuple[str, str, list[SplitFace]]] = []
    remainder: list[SplitFace] = []
    for component_index, component in enumerate(component_indices, start=1):
        component_faces = [faces[index] for index in component]
        if len(component) >= min_faces:
            specs.append((f"{base_name}_{component_index}", method, component_faces))
        else:
            remainder.extend(component_faces)
    if remainder:
        specs.append((f"{base_name}_misc", method, remainder))
    return specs


def refine_regular_component_specs_by_spatial_detail(
    specs: list[tuple[str, str, list[SplitFace]]],
    detail_split: HybridDetailSplitConfig,
) -> list[tuple[str, str, list[SplitFace]]]:
    refined: list[tuple[str, str, list[SplitFace]]] = []
    for base_name, method, faces in specs:
        spatial_specs = split_regular_faces_by_spatial_detail(base_name, faces, detail_split)
        if len(spatial_specs) >= 2:
            refined.extend(spatial_specs)
        else:
            refined.append((base_name, method, faces))
    return refined


def split_regular_faces_by_spatial_detail(
    base_name: str,
    faces: list[SplitFace],
    detail_split: HybridDetailSplitConfig,
) -> list[tuple[str, str, list[SplitFace]]]:
    if len(faces) < max(REGULAR_SPATIAL_DETAIL_MIN_FACES, detail_split.min_faces):
        return [(base_name, "regular_detail", faces)]

    part_min, part_max = faces_bbox(faces)
    dimensions = bbox_dimensions(part_min, part_max)
    longest = max(dimensions) if dimensions else 0.0
    if longest <= EPSILON:
        return [(base_name, "regular_detail", faces)]

    axes = [
        axis
        for axis in sorted(range(3), key=lambda item: dimensions[item], reverse=True)
        if dimensions[axis] >= longest * REGULAR_SPATIAL_DETAIL_AXIS_RATIO
    ]
    if len(axes) < 2:
        return [(base_name, "regular_detail", faces)]

    min_bucket_faces = max(
        detail_split.min_component_faces,
        math.ceil(len(faces) * detail_split.min_component_ratio),
    )
    parts: list[tuple[str, list[SplitFace]]] = [(base_name, faces)]
    for axis in axes[:REGULAR_SPATIAL_DETAIL_MAX_AXES]:
        split_parts = split_faces_by_axis(parts, axis, 2, axis_suffixes(axis, 2))
        split_parts, _merged = merge_tiny_named_parts(split_parts, min_bucket_faces)
        if len(split_parts) > len(parts):
            parts = split_parts

    if len(parts) < 2:
        return [(base_name, "regular_detail", faces)]
    return [(name, "regular_spatial_detail", part_faces) for name, part_faces in parts]


def apply_auto_spatial_split(
    accumulators: dict[PartKey, BBoxAccumulator],
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
    bones: dict[int, BonePartition],
    mode: str,
) -> list[ComplexSplitBoneReport]:
    if mode != "hybrid":
        return []

    populated = [
        accumulator
        for accumulator in accumulators.values()
        if accumulator.min_xyz is not None and accumulator.max_xyz is not None and accumulator.faces > 0
    ]
    if not populated:
        return []
    model_min, model_max = combined_bbox(populated)
    model_height = max(model_max[1] - model_min[1], EPSILON)
    model_volume = bbox_volume(model_min, model_max)

    reports: list[ComplexSplitBoneReport] = []
    for part_key, faces in sorted(list(regular_faces_by_part.items()), key=lambda item: (item[0].owner_bone, item[0].name)):
        accumulator = accumulators.get(part_key)
        if accumulator is None or accumulator.min_xyz is None or accumulator.max_xyz is None:
            continue
        if accumulator.is_complex_split or accumulator.faces < AUTO_SPATIAL_SPLIT_MIN_FACES or len(faces) < 2:
            continue
        if not should_auto_spatial_split(accumulator, model_height, model_volume):
            continue

        requested_split_parts = auto_spatial_split_faces(part_key.name, faces, accumulator, model_height)
        if len(requested_split_parts) < 2:
            continue
        budget_limit = AUTO_SPATIAL_SPLIT_OWNER_BUDGET_MULTIPLIER
        split_parts, capped_to_budget = cap_auto_spatial_parts(requested_split_parts, budget_limit)
        if len(split_parts) < 2:
            continue

        del accumulators[part_key]
        regular_faces_by_part.pop(part_key, None)
        report_accumulators: dict[str, BBoxAccumulator] = {}
        used_names: dict[str, BBoxAccumulator] = {}
        for fallback_index, split_part in enumerate(split_parts, start=1):
            if not split_part.faces:
                continue
            part_name = unique_part_name(split_part.name or f"{sanitize_part_name(part_key.name)}_{fallback_index}", used_names)
            part_accumulator = accumulator_from_auto_spatial_part(split_part)
            part_accumulator.is_complex_split = True
            new_key = PartKey(part_key.owner_bone, part_name)
            accumulators[new_key] = part_accumulator
            regular_faces_by_part[new_key] = split_part.faces
            report_accumulators[part_name] = part_accumulator
            used_names[part_name] = part_accumulator

        if not report_accumulators:
            continue
        bone = bones.get(part_key.owner_bone)
        reports.append(
            ComplexSplitBoneReport(
                bone=part_key.owner_bone,
                bone_name=bone.name if bone is not None else part_key.name,
                source_faces=accumulator.faces,
                subparts=[
                    ComplexSplitSubpartReport(
                        name=name,
                        method="auto_spatial_grid",
                        faces=part_accumulator.faces,
                        vertices=len(part_accumulator.vertices),
                    )
                    for name, part_accumulator in sorted(report_accumulators.items())
                ],
                original_cube_dimensions=bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz),
                split_method="auto_spatial_grid",
                requested_subpart_count=len(requested_split_parts),
                budget_limit=budget_limit,
                budget_status="capped" if capped_to_budget else "within_budget",
                budget_reason="auto_spatial_split_capped_to_budget" if capped_to_budget else None,
            )
        )
    return reports


def should_auto_spatial_split(accumulator: BBoxAccumulator, model_height: float, model_volume: float) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    dimensions = bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz)
    sorted_dimensions = sorted(dimensions, reverse=True)
    if sorted_dimensions[0] <= EPSILON or sorted_dimensions[1] <= EPSILON:
        return False
    volume = bbox_volume(accumulator.min_xyz, accumulator.max_xyz)
    if (
        model_volume > EPSILON
        and volume / model_volume >= AUTO_SPATIAL_SPLIT_VOLUME_RATIO
        and sorted_dimensions[0] / model_height >= AUTO_SPATIAL_SPLIT_VOLUME_LONG_DIM_RATIO
        and sorted_dimensions[1] / model_height >= AUTO_SPATIAL_SPLIT_VOLUME_SECOND_DIM_RATIO
    ):
        return True
    return (
        sorted_dimensions[0] / model_height >= AUTO_SPATIAL_SPLIT_LONG_DIM_RATIO
        and sorted_dimensions[1] / model_height >= AUTO_SPATIAL_SPLIT_SECOND_DIM_RATIO
    )


def auto_spatial_split_faces(
    part_name: str,
    faces: list[SplitFace],
    accumulator: BBoxAccumulator,
    model_height: float,
) -> list[AutoSpatialPart]:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return []
    dimensions = bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz)
    axes = auto_spatial_split_axes(dimensions, model_height)
    if not axes:
        return []

    parts = [AutoSpatialPart(sanitize_part_name(part_name), faces, accumulator.min_xyz.copy(), accumulator.max_xyz.copy())]
    for axis in axes:
        segment_count = auto_spatial_segment_count(dimensions[axis], model_height, len(faces))
        parts = split_auto_spatial_parts(parts, axis, segment_count)
    return [part for part in parts if part.faces]


def generic_complex_auto_spatial_split_faces(
    part_name: str,
    faces: list[SplitFace],
    accumulator: BBoxAccumulator,
    model_height: float,
    max_axes: int,
) -> list[AutoSpatialPart]:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return []
    dimensions = bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz)
    axes = auto_spatial_split_axes(dimensions, model_height)[:max(1, max_axes)]
    if not axes:
        return []

    parts = [AutoSpatialPart(sanitize_part_name(part_name), faces, accumulator.min_xyz.copy(), accumulator.max_xyz.copy())]
    for axis in axes:
        segment_count = auto_spatial_segment_count(dimensions[axis], model_height, len(faces))
        parts = split_auto_spatial_parts(parts, axis, segment_count)
    return [part for part in parts if part.faces]


def split_auto_spatial_parts(
    parts: list[AutoSpatialPart],
    axis: int,
    segment_count: int,
) -> list[AutoSpatialPart]:
    result: list[AutoSpatialPart] = []
    for part in parts:
        axis_min = part.min_xyz[axis]
        axis_max = part.max_xyz[axis]
        axis_size = axis_max - axis_min
        if len(part.faces) < 2 or axis_size <= EPSILON:
            result.append(part)
            continue

        buckets: list[list[SplitFace]] = [[] for _ in range(segment_count)]
        for face in part.faces:
            centroid_value = points_centroid(face.points)[axis]
            ratio = (centroid_value - axis_min) / axis_size
            bucket_index = min(max(int(ratio * segment_count), 0), segment_count - 1)
            buckets[bucket_index].append(face)

        split_parts: list[AutoSpatialPart] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            child_min = part.min_xyz.copy()
            child_max = part.max_xyz.copy()
            child_min[axis] = axis_min + axis_size * index / segment_count
            child_max[axis] = axis_min + axis_size * (index + 1) / segment_count
            split_parts.append(AutoSpatialPart(f"{part.name}_{index + 1}", bucket, child_min, child_max))
        result.extend(split_parts or [part])
    return result



def cap_auto_spatial_parts(
    parts: list[AutoSpatialPart],
    budget_limit: int,
) -> tuple[list[AutoSpatialPart], bool]:
    if budget_limit <= 0 or len(parts) <= budget_limit:
        return parts, False

    all_faces = [face for part in parts for face in part.faces]
    if len(all_faces) < 2:
        return parts[:budget_limit], True

    min_xyz, max_xyz = faces_bbox(all_faces)
    dimensions = bbox_dimensions(min_xyz, max_xyz)
    axis = max(range(3), key=lambda index: dimensions[index])
    base_name = parts[0].name.rsplit("_", 1)[0] if parts else "part"
    capped = split_auto_spatial_parts(
        [AutoSpatialPart(f"{base_name}_budget", all_faces, min_xyz, max_xyz)],
        axis,
        budget_limit,
    )
    return [part for part in capped if part.faces] or parts[:budget_limit], True

def accumulator_from_auto_spatial_part(part: AutoSpatialPart) -> BBoxAccumulator:
    accumulator = BBoxAccumulator(min_xyz=part.min_xyz.copy(), max_xyz=part.max_xyz.copy(), faces=len(part.faces))
    for face in part.faces:
        accumulator.vertices.update(face.vertex_keys)
        for point, vertex_key in zip(face.points, face.vertex_keys, strict=False):
            accumulator.points_by_vertex.setdefault(vertex_key, clamp_point_to_bbox(point, part.min_xyz, part.max_xyz))
    return accumulator


def accumulator_from_faces(faces: list[SplitFace]) -> BBoxAccumulator:
    accumulator = BBoxAccumulator()
    for face in faces:
        accumulator.add_face(face.points, face.vertex_keys)
    return accumulator


def auto_spatial_split_axes(dimensions: list[float], model_height: float) -> list[int]:
    ranked_axes = sorted(range(3), key=lambda axis: dimensions[axis], reverse=True)
    axes: list[int] = []
    for axis in ranked_axes:
        if len(axes) >= AUTO_SPATIAL_SPLIT_MAX_AXES:
            break
        if dimensions[axis] / model_height >= AUTO_SPATIAL_SPLIT_AXIS_DIM_RATIO:
            axes.append(axis)
    return axes or ranked_axes[:1]


def auto_spatial_segment_count(dimension: float, model_height: float, face_count: int) -> int:
    ratio = dimension / model_height
    by_dimension = 2
    if ratio >= 0.50:
        by_dimension = 4
    elif ratio >= 0.33:
        by_dimension = 3
    by_faces = max(2, math.ceil(face_count / AUTO_SPATIAL_SPLIT_TARGET_FACES))
    return max(2, min(max(by_dimension, by_faces), 4))


def generic_component_part_name(
    bone_name: str,
    faces: list[SplitFace],
    owner_bbox: tuple[list[float], list[float]],
    component_index: int,
) -> str:
    base_name = sanitize_part_name(bone_name)
    if matches_complex_alias(bone_name, "hair"):
        return f"{classify_generic_hair_component(bone_name, faces, owner_bbox)}_{component_index}"
    material_part = classify_generic_material(faces)
    if material_part is not None:
        return f"{base_name}_{material_part}_{component_index}"
    if matches_complex_alias(bone_name, "skirt"):
        return f"{base_name}_{classify_ring_component_region(faces, owner_bbox)}_{component_index}"
    return f"{base_name}_part_{component_index}"


def classify_generic_hair_component(
    bone_name: str,
    faces: list[SplitFace],
    owner_bbox: tuple[list[float], list[float]],
) -> str:
    material_names = [face.material_name for face in faces if face.material_name]
    if material_names:
        material_region = classify_hair_material_region(max(set(material_names), key=material_names.count))
        if material_region is not None:
            return material_region
    bone_region = classify_hair_material_region(bone_name)
    if bone_region is not None:
        return bone_region
    spatial_region = classify_head_component_spatial(faces, owner_bbox)
    return spatial_region if spatial_region.startswith("hair") else "hair"


def classify_generic_material(faces: list[SplitFace]) -> str | None:
    names = [face.material_name for face in faces if face.material_name]
    if not names:
        return None
    material_name = max(set(names), key=names.count)
    haystack = material_name.casefold()
    for part_name, patterns in HEAD_MATERIAL_PATTERNS.items():
        if any(pattern.casefold() in haystack for pattern in patterns):
            return part_name
    return sanitize_part_name(material_name)


def classify_ring_component_region(faces: list[SplitFace], owner_bbox: tuple[list[float], list[float]]) -> str:
    centroid = points_centroid([point for face in faces for point in face.points])
    min_xyz, max_xyz = owner_bbox
    center_x = (min_xyz[0] + max_xyz[0]) * 0.5
    center_z = (min_xyz[2] + max_xyz[2]) * 0.5
    dx = centroid[0] - center_x
    dz = centroid[2] - center_z
    if abs(dx) > abs(dz):
        return "side_l" if dx < 0 else "side_r"
    return "front" if dz < 0 else "back"


def matches_complex_alias(name: str, alias_name: str) -> bool:
    haystack = name.casefold()
    for alias in COMPLEX_BONE_ALIASES.get(alias_name, (alias_name,)):
        if alias.casefold() in haystack:
            return True
    return False


def sanitize_part_name(name: str) -> str:
    cleaned = []
    for char in name.strip():
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    return result or "part"


def apply_tiny_component_rules(
    component_indices: list[list[int]],
    faces: list[SplitFace],
    config: ComplexSplitConfig,
) -> tuple[list[list[int]], dict[tuple[int, ...], list[SplitFace]], int, int]:
    connected = config.connected_components
    if not connected.enabled or connected.min_faces <= 0 or not component_indices:
        return component_indices, {}, 0, 0

    largest_component = tuple(component_indices[0])
    tiny_keys = {
        tuple(component)
        for component in component_indices
        if len(component) < connected.min_faces and tuple(component) != largest_component
    }
    if not tiny_keys:
        return component_indices, {}, 0, 0

    kept_components = [component for component in component_indices if tuple(component) not in tiny_keys]
    if connected.merge_tiny_components_to_nearest:
        tiny_faces_by_target: dict[tuple[int, ...], list[SplitFace]] = {}
        for component in component_indices:
            component_key = tuple(component)
            if component_key not in tiny_keys:
                continue
            target = nearest_component(component, kept_components, faces)
            tiny_faces_by_target.setdefault(tuple(target), []).extend(faces[index] for index in component)
        return kept_components, tiny_faces_by_target, len(tiny_keys), 0

    if connected.delete_tiny_components:
        return kept_components, {}, 0, len(tiny_keys)

    return component_indices, {}, 0, 0


def nearest_component(
    component: list[int],
    targets: list[list[int]],
    faces: list[SplitFace],
) -> list[int]:
    if not targets:
        return component
    centroid = component_centroid(component, faces)
    return min(targets, key=lambda target: squared_distance(centroid, component_centroid(target, faces)))


def component_centroid(component: list[int], faces: list[SplitFace]) -> list[float]:
    return points_centroid([point for face_index in component for point in faces[face_index].points])


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((left[index] - right[index]) ** 2 for index in range(3))


def connected_face_components(faces: list[SplitFace]) -> list[list[int]]:
    parent = list(range(len(faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_face_by_vertex: dict[tuple[str, int], int] = {}
    for face_index, face in enumerate(faces):
        for vertex_key in face.vertex_keys:
            other = first_face_by_vertex.get(vertex_key)
            if other is None:
                first_face_by_vertex[vertex_key] = face_index
            else:
                union(other, face_index)

    components: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        components.setdefault(find(face_index), []).append(face_index)
    return list(components.values())


def classify_head_material(faces: list[SplitFace]) -> str | None:
    material_names = [face.material_name for face in faces if face.material_name]
    for part_name, patterns in HEAD_MATERIAL_PATTERNS.items():
        for material_name in material_names:
            haystack = material_name.casefold()
            if any(pattern.casefold() in haystack for pattern in patterns):
                return part_name
    return None


def classify_hair_face(
    face: SplitFace,
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int = DEFAULT_HEAD_FRONT_SIGN,
) -> str:
    material_part = classify_hair_material_region(face.material_name)
    if material_part is not None:
        return material_part

    centroid = points_centroid(face.points)
    min_xyz, max_xyz = owner_bbox
    width = max_xyz[0] - min_xyz[0]
    depth = max_xyz[2] - min_xyz[2]
    height = max_xyz[1] - min_xyz[1]

    if height > EPSILON and centroid[1] >= max_xyz[1] - height * 0.12:
        return "hair_top"
    if width > EPSILON:
        if centroid[0] <= min_xyz[0] + width * 0.25:
            return "hair_side_l"
        if centroid[0] >= max_xyz[0] - width * 0.25:
            return "hair_side_r"
    if depth > EPSILON:
        if front_sign >= 0:
            if centroid[2] >= max_xyz[2] - depth * 0.35:
                return "hair_front"
            if centroid[2] <= min_xyz[2] + depth * 0.35:
                return "hair_back"
        else:
            if centroid[2] <= min_xyz[2] + depth * 0.35:
                return "hair_front"
            if centroid[2] >= max_xyz[2] - depth * 0.35:
                return "hair_back"
    return "hair"


def classify_hair_material_region(material_name: str | None) -> str | None:
    if not material_name:
        return None
    haystack = material_name.casefold()
    if any(pattern in haystack for pattern in ("front", "bang", "前髪".casefold(), "前发", "前髮")):
        return "hair_front"
    if any(pattern in haystack for pattern in ("back", "rear", "後髪".casefold(), "后发", "後髮")):
        return "hair_back"
    if any(pattern in haystack for pattern in ("top", "ahoge", "アホ毛".casefold(), "呆毛")):
        return "hair_top"
    if "left" in haystack:
        return "hair_side_l"
    if "right" in haystack:
        return "hair_side_r"
    return None


def infer_head_front_sign(faces: list[SplitFace], owner_bbox: tuple[list[float], list[float]]) -> int:
    feature_points: list[list[float]] = []
    head_core_points: list[list[float]] = []
    for face in faces:
        material_part = classify_head_material([face])
        if material_part in FACE_FEATURE_PART_PREFIXES:
            feature_points.extend(face.points)
        elif material_part == "head_core":
            head_core_points.extend(face.points)

    if not feature_points:
        return DEFAULT_HEAD_FRONT_SIGN

    min_xyz, max_xyz = owner_bbox
    depth = max_xyz[2] - min_xyz[2]
    reference_points = head_core_points or [point for face in faces for point in face.points]
    feature_z = points_centroid(feature_points)[2]
    reference_z = points_centroid(reference_points)[2]
    if depth <= EPSILON or abs(feature_z - reference_z) <= depth * 0.03:
        return DEFAULT_HEAD_FRONT_SIGN
    return 1 if feature_z > reference_z else -1


def is_hair_part(name: str) -> bool:
    return name == "hair" or name.startswith("hair_")


def is_head_core_part(name: str) -> bool:
    return name == "head_core" or name.startswith("head_core_")


def is_face_feature_name(name: str) -> bool:
    haystack = name.casefold()
    return any(
        pattern.casefold() in haystack
        for pattern in (
            "eye",
            "目",
            "瞳",
            "brow",
            "眉",
            "eyelash",
            "まつげ",
            "睫",
            "mouth",
            "口",
            "nose",
            "鼻",
        )
    )


def classify_eye_face(face: SplitFace, owner_bbox: tuple[list[float], list[float]]) -> str:
    return classify_lateral_feature_face(face, owner_bbox, "eye")


def classify_lateral_feature_face(
    face: SplitFace,
    owner_bbox: tuple[list[float], list[float]],
    prefix: str,
) -> str:
    centroid = points_centroid(face.points)
    min_xyz, max_xyz = owner_bbox
    center_x = (min_xyz[0] + max_xyz[0]) * 0.5
    return f"{prefix}_l" if centroid[0] < center_x else f"{prefix}_r"


def split_hair_part_faces(
    part_name: str,
    faces: list[SplitFace],
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int = DEFAULT_HEAD_FRONT_SIGN,
) -> HairSplitResult:
    if len(faces) < 2:
        return HairSplitResult(parts=[HairSplitPart(part_name, faces)])

    part_min, part_max = faces_bbox(faces)
    owner_min, owner_max = owner_bbox
    part_width = part_max[0] - part_min[0]
    part_depth = part_max[2] - part_min[2]
    owner_width = owner_max[0] - owner_min[0]
    owner_depth = owner_max[2] - owner_min[2]
    part_height = part_max[1] - part_min[1]
    owner_height = owner_max[1] - owner_min[1]

    parts = [HairSplitPart(part_name, faces)]
    merged_tiny_buckets = 0
    if len(faces) >= 6 and part_name in {"hair", "hair_front", "hair_back", "hair_top"}:
        if part_width > EPSILON and owner_width > EPSILON and part_width >= owner_width * 0.45:
            segment_count = 3 if len(faces) < 1500 else 4
            parts = split_hair_parts_by_axis(parts, axis=0, segment_count=segment_count, suffixes=horizontal_suffixes(segment_count))
            parts, merged = merge_tiny_hair_face_parts(parts, owner_bbox)
            merged_tiny_buckets += merged
        min_depth_for_split = max(owner_depth * 0.45, max(part_width, part_height) * 0.25)
        if part_depth > EPSILON and owner_depth > EPSILON and part_depth >= min_depth_for_split:
            parts = split_hair_parts_by_axis(parts, axis=2, segment_count=2, suffixes=head_core_depth_suffixes(front_sign))
            parts, merged = merge_tiny_hair_face_parts(parts, owner_bbox)
            merged_tiny_buckets += merged

    if len(faces) < 250 or part_height <= EPSILON or owner_height <= EPSILON or part_height < owner_height * 0.3:
        parts = refine_large_hair_split_parts(parts, owner_bbox)
        return HairSplitResult(parts=parts, merged_tiny_buckets=merged_tiny_buckets)

    segment_count = max(math.ceil(part_height / (owner_height * 0.3)), math.ceil(len(faces) / 2500))
    segment_count = max(2, min(segment_count, 4))
    parts = split_hair_parts_by_axis(parts, axis=1, segment_count=segment_count, suffixes=vertical_suffixes(segment_count))
    parts, merged = merge_tiny_hair_face_parts(parts, owner_bbox)
    merged_tiny_buckets += merged
    parts = refine_large_hair_split_parts(parts, owner_bbox)
    return HairSplitResult(parts=parts, merged_tiny_buckets=merged_tiny_buckets)


def refine_large_hair_split_parts(
    parts: list[HairSplitPart],
    owner_bbox: tuple[list[float], list[float]],
) -> list[HairSplitPart]:
    refined: list[HairSplitPart] = []
    for part in parts:
        refined.extend(refine_large_hair_split_part(part, owner_bbox))
    return refined


def refine_large_hair_split_part(
    part: HairSplitPart,
    owner_bbox: tuple[list[float], list[float]],
) -> list[HairSplitPart]:
    if len(part.faces) < HEAD_HAIR_REFINEMENT_MIN_FACES:
        return [part]

    owner_min, owner_max = owner_bbox
    owner_dimensions = bbox_dimensions(owner_min, owner_max)
    part_min, part_max = faces_bbox(part.faces)
    part_dimensions = bbox_dimensions(part_min, part_max)
    longest = max(part_dimensions) if part_dimensions else 0.0
    if longest <= EPSILON:
        return [part]

    ranked_axes = sorted(range(3), key=lambda axis: part_dimensions[axis], reverse=True)
    axes: list[int] = []
    for axis in ranked_axes:
        if len(axes) >= HEAD_HAIR_REFINEMENT_MAX_AXES:
            break
        if part_dimensions[axis] < longest * HEAD_HAIR_REFINEMENT_MIN_AXIS_RATIO:
            continue
        if owner_dimensions[axis] > EPSILON and part_dimensions[axis] < owner_dimensions[axis] * 0.18:
            continue
        axes.append(axis)
    if not axes:
        return [part]

    refined = [part]
    for axis in axes:
        split_targets = [item for item in refined if should_refine_large_hair_split_part(item, owner_dimensions)]
        if not split_targets:
            break
        split_candidates = split_hair_parts_by_axis(
            split_targets,
            axis=axis,
            segment_count=2,
            suffixes=axis_suffixes(axis, 2),
        )
        untouched_parts = [item for item in refined if not should_refine_large_hair_split_part(item, owner_dimensions)]
        if len(split_candidates) <= len(split_targets):
            continue
        refined = untouched_parts + split_candidates
    return refined


def should_refine_large_hair_split_part(part: HairSplitPart, owner_dimensions: list[float]) -> bool:
    if len(part.faces) > HEAD_HAIR_REFINEMENT_TARGET_FACES:
        return True
    if len(part.faces) < HEAD_HAIR_REFINEMENT_MIN_BUCKET_FACES:
        return False
    part_min, part_max = faces_bbox(part.faces)
    part_dimensions = bbox_dimensions(part_min, part_max)
    large_axes = sum(
        1
        for axis, dimension in enumerate(part_dimensions)
        if owner_dimensions[axis] > EPSILON and dimension >= owner_dimensions[axis] * 0.35
    )
    return large_axes >= 2


def split_hair_parts_by_axis(
    parts: list[HairSplitPart],
    axis: int,
    segment_count: int,
    suffixes: list[str],
) -> list[HairSplitPart]:
    result: list[HairSplitPart] = []
    for part in parts:
        split_parts = split_faces_by_axis([(part.name, part.faces)], axis, segment_count, suffixes)
        if len(split_parts) == 1:
            split_name, split_faces = split_parts[0]
            if split_name == part.name and split_faces is part.faces:
                result.append(part)
            else:
                result.append(HairSplitPart(split_name, split_faces, split_axes=set(part.split_axes)))
            continue

        next_split_axes = set(part.split_axes)
        next_split_axes.add(axis)
        for split_name, split_faces in split_parts:
            result.append(HairSplitPart(split_name, split_faces, split_axes=set(next_split_axes)))
    return result


def merge_tiny_hair_face_parts(
    parts: list[HairSplitPart],
    owner_bbox: tuple[list[float], list[float]],
) -> tuple[list[HairSplitPart], int]:
    if len(parts) < 2:
        return parts, 0

    owner_min, owner_max = owner_bbox
    owner_dimensions = bbox_dimensions(owner_min, owner_max)
    owner_longest = max(owner_dimensions) if owner_dimensions else 0.0
    min_faces = max(HAIR_BUCKET_MIN_FACES, math.ceil(sum(len(part.faces) for part in parts) * HAIR_BUCKET_MIN_FACE_RATIO))

    kept: list[HairSplitPart] = []
    tiny: list[HairSplitPart] = []
    for part in parts:
        part_min, part_max = faces_bbox(part.faces)
        part_dimensions = bbox_dimensions(part_min, part_max)
        part_longest = max(part_dimensions) if part_dimensions else 0.0
        tiny_by_span = part_longest <= max(owner_longest * 0.12, MIN_CUBE_SIZE * 2.0)
        if len(part.faces) < min_faces or tiny_by_span:
            tiny.append(part)
        else:
            kept.append(part)

    if not kept or not tiny:
        return parts, 0

    merged_count = 0
    for tiny_part in tiny:
        tiny_centroid = points_centroid([point for face in tiny_part.faces for point in face.points])
        target_index = min(
            range(len(kept)),
            key=lambda index: squared_distance(
                tiny_centroid,
                points_centroid([point for face in kept[index].faces for point in face.points]),
            ),
        )
        kept[target_index].faces.extend(tiny_part.faces)
        kept[target_index].split_axes.update(tiny_part.split_axes)
        merged_count += 1

    return kept, merged_count


def expand_hair_bucket_accumulator(accumulator: BBoxAccumulator, split_axes: set[int]) -> bool:
    if not split_axes or accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False

    dimensions = bbox_dimensions(accumulator.min_xyz, accumulator.max_xyz)
    margin = hair_bucket_overlap_margin(dimensions)
    if margin <= EPSILON:
        return False

    expanded_min = accumulator.min_xyz.copy()
    expanded_max = accumulator.max_xyz.copy()
    for axis in sorted(split_axes):
        expanded_min[axis] -= margin
        expanded_max[axis] += margin

    accumulator.min_xyz = expanded_min
    accumulator.max_xyz = expanded_max
    base_index = len(accumulator.points_by_vertex)
    for index, corner in enumerate(bbox_corners(expanded_min, expanded_max)):
        accumulator.points_by_vertex[("hair_overlap", base_index + index)] = corner.copy()
    return True


def hair_bucket_overlap_margin(dimensions: list[float]) -> float:
    largest = max(dimensions) if dimensions else 0.0
    if largest <= EPSILON:
        return HAIR_BUCKET_OVERLAP_MIN
    return max(largest * HAIR_BUCKET_OVERLAP_RATIO, HAIR_BUCKET_OVERLAP_MIN)


def split_head_accessory_part_faces(
    part_name: str,
    faces: list[SplitFace],
    owner_bbox: tuple[list[float], list[float]],
) -> list[tuple[str, list[SplitFace]]]:
    if len(faces) < 2:
        return [(part_name, faces)]

    part_min, part_max = faces_bbox(faces)
    owner_min, owner_max = owner_bbox
    owner_dimensions = bbox_dimensions(owner_min, owner_max)
    part_dimensions = bbox_dimensions(part_min, part_max)
    ranked_axes = sorted(
        range(3),
        key=lambda axis: part_dimensions[axis] / max(owner_dimensions[axis], EPSILON),
        reverse=True,
    )

    parts = [(part_name, faces)]
    split_axes = 0
    for axis in ranked_axes:
        if split_axes >= 2:
            break
        if owner_dimensions[axis] <= EPSILON or part_dimensions[axis] < owner_dimensions[axis] * 0.45:
            continue
        segment_count = 3 if len(faces) < 1200 else 4
        parts = split_faces_by_axis(parts, axis=axis, segment_count=segment_count, suffixes=axis_suffixes(axis, segment_count))
        split_axes += 1
    return merge_tiny_face_parts(parts, HEAD_ACCESSORY_SPLIT_MIN_FACES)


def merge_tiny_face_parts(
    parts: list[tuple[str, list[SplitFace]]],
    min_faces: int,
) -> list[tuple[str, list[SplitFace]]]:
    if min_faces <= 0 or len(parts) < 2:
        return parts
    kept = [(name, part_faces) for name, part_faces in parts if len(part_faces) >= min_faces]
    tiny = [(name, part_faces) for name, part_faces in parts if len(part_faces) < min_faces]
    if not kept or not tiny:
        return parts

    for _name, part_faces in tiny:
        centroid = points_centroid([point for face in part_faces for point in face.points])
        target_index = min(
            range(len(kept)),
            key=lambda index: squared_distance(
                centroid,
                points_centroid([point for face in kept[index][1] for point in face.points]),
            ),
        )
        kept[target_index][1].extend(part_faces)
    return kept


def split_faces_by_axis(
    parts: list[tuple[str, list[SplitFace]]],
    axis: int,
    segment_count: int,
    suffixes: list[str],
) -> list[tuple[str, list[SplitFace]]]:
    result: list[tuple[str, list[SplitFace]]] = []
    for part_name, faces in parts:
        part_min, part_max = faces_bbox(faces)
        part_size = part_max[axis] - part_min[axis]
        if len(faces) < 2 or part_size <= EPSILON:
            result.append((part_name, faces))
            continue

        buckets: list[list[SplitFace]] = [[] for _ in range(segment_count)]
        for face in faces:
            centroid_value = points_centroid(face.points)[axis]
            ratio = (centroid_value - part_min[axis]) / part_size
            bucket_index = min(max(int(ratio * segment_count), 0), segment_count - 1)
            buckets[bucket_index].append(face)

        split_parts = [
            (f"{part_name}_{suffixes[index]}", bucket)
            for index, bucket in enumerate(buckets)
            if bucket
        ]
        result.extend(split_parts if len(split_parts) > 1 else [(part_name, faces)])
    return result


def horizontal_suffixes(segment_count: int) -> list[str]:
    if segment_count == 2:
        return ["left", "right"]
    if segment_count == 3:
        return ["left", "center", "right"]
    return ["left", "mid_left", "mid_right", "right"]


def vertical_suffixes(segment_count: int) -> list[str]:
    if segment_count == 2:
        return ["lower", "upper"]
    if segment_count == 3:
        return ["lower", "middle", "upper"]
    return ["lower", "mid_lower", "mid_upper", "upper"]


def depth_suffixes(segment_count: int) -> list[str]:
    if segment_count == 2:
        return ["front", "back"]
    if segment_count == 3:
        return ["front", "middle", "back"]
    return ["front", "mid_front", "mid_back", "back"]


def axis_suffixes(axis: int, segment_count: int) -> list[str]:
    if axis == 0:
        return horizontal_suffixes(segment_count)
    if axis == 1:
        return vertical_suffixes(segment_count)
    return depth_suffixes(segment_count)


def unique_part_name(base_name: str, accumulators: dict[str, BBoxAccumulator]) -> str:
    if base_name not in accumulators:
        return base_name
    suffix = 2
    while f"{base_name}_{suffix}" in accumulators:
        suffix += 1
    return f"{base_name}_{suffix}"


def is_face_feature_part(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in FACE_FEATURE_PART_PREFIXES)


def explicit_face_feature_accumulators_bbox(
    accumulators: dict[PartKey, BBoxAccumulator],
    owner_bone: int,
    owner_bbox: tuple[list[float], list[float]],
    config: FaceFeatureProtectionConfig,
    *,
    significant_only: bool = False,
) -> tuple[list[float], list[float]] | None:
    feature_accumulators = explicit_face_feature_accumulators(
        accumulators,
        owner_bone,
        owner_bbox,
        config,
        significant_only=significant_only,
    )
    if not feature_accumulators:
        return None
    return combined_bbox(feature_accumulators)


def explicit_face_feature_names(
    accumulators: dict[PartKey, BBoxAccumulator],
    owner_bone: int,
    owner_bbox: tuple[list[float], list[float]],
    config: FaceFeatureProtectionConfig,
) -> tuple[str, ...]:
    significant = explicit_face_feature_accumulator_items(
        accumulators,
        owner_bone,
        owner_bbox,
        config,
        significant_only=True,
    )
    items = significant or explicit_face_feature_accumulator_items(
        accumulators,
        owner_bone,
        owner_bbox,
        config,
        significant_only=False,
    )
    return tuple(sorted(part_key.name for part_key, _accumulator in items))


def explicit_face_feature_visibility_bbox(
    accumulators: dict[PartKey, BBoxAccumulator],
    owner_bone: int,
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int,
    config: FaceFeatureProtectionConfig,
) -> tuple[list[float], list[float]] | None:
    feature_accumulators = explicit_face_feature_accumulators(
        accumulators,
        owner_bone,
        owner_bbox,
        config,
        significant_only=True,
    ) or explicit_face_feature_accumulators(
        accumulators,
        owner_bone,
        owner_bbox,
        config,
        significant_only=False,
    )
    if not feature_accumulators:
        return None

    min_xyz, max_xyz = combined_bbox(feature_accumulators)
    depth_values = sorted(
        point[2]
        for accumulator in feature_accumulators
        for point in (list(accumulator.points_by_vertex.values()) or bbox_corners(accumulator.min_xyz, accumulator.max_xyz))
    )
    if depth_values:
        owner_min, owner_max = owner_bbox
        owner_depth = max(owner_max[2] - owner_min[2], EPSILON)
        if front_sign >= 0:
            min_xyz[2] = max(
                min_xyz[2],
                feature_visibility_depth_value(depth_values, front_sign, owner_depth, config),
            )
        else:
            max_xyz[2] = min(
                max_xyz[2],
                feature_visibility_depth_value(depth_values, front_sign, owner_depth, config),
            )
    return min_xyz, max_xyz


def explicit_face_feature_accumulators(
    accumulators: dict[PartKey, BBoxAccumulator],
    owner_bone: int,
    owner_bbox: tuple[list[float], list[float]],
    config: FaceFeatureProtectionConfig,
    *,
    significant_only: bool,
) -> list[BBoxAccumulator]:
    return [
        accumulator
        for _part_key, accumulator in explicit_face_feature_accumulator_items(
            accumulators,
            owner_bone,
            owner_bbox,
            config,
            significant_only=significant_only,
        )
    ]


def explicit_face_feature_accumulator_items(
    accumulators: dict[PartKey, BBoxAccumulator],
    owner_bone: int,
    owner_bbox: tuple[list[float], list[float]],
    config: FaceFeatureProtectionConfig,
    *,
    significant_only: bool,
) -> list[tuple[PartKey, BBoxAccumulator]]:
    return [
        (part_key, accumulator)
        for part_key, accumulator in accumulators.items()
        if part_key.owner_bone != owner_bone
        and is_face_feature_name(part_key.name)
        and accumulator.min_xyz is not None
        and accumulator.max_xyz is not None
        and accumulator_center_near_bbox(accumulator, owner_bbox)
        and (not significant_only or accumulator.faces >= config.min_faces)
    ]


def feature_visibility_depth_value(
    sorted_values: list[float],
    front_sign: int,
    owner_depth: float,
    config: FaceFeatureProtectionConfig,
) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) < 3:
        return sorted_values[0] if front_sign >= 0 else sorted_values[-1]

    gap_threshold = max(owner_depth * config.outlier_gap_ratio, MIN_CUBE_SIZE * 4.0)
    scan_count = max(1, int((len(sorted_values) - 1) * 0.35))
    if front_sign >= 0:
        best_index = 0
        best_gap = 0.0
        for index in range(scan_count):
            gap = sorted_values[index + 1] - sorted_values[index]
            if gap > best_gap:
                best_gap = gap
                best_index = index + 1
        return sorted_values[best_index] if best_gap >= gap_threshold else sorted_values[0]

    best_index = len(sorted_values) - 1
    best_gap = 0.0
    start = max(0, len(sorted_values) - 1 - scan_count)
    for index in range(len(sorted_values) - 2, start - 1, -1):
        gap = sorted_values[index + 1] - sorted_values[index]
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return sorted_values[best_index] if best_gap >= gap_threshold else sorted_values[-1]


def accumulator_center_near_bbox(
    accumulator: BBoxAccumulator,
    bbox: tuple[list[float], list[float]],
) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    min_xyz, max_xyz = bbox
    dimensions = bbox_dimensions(min_xyz, max_xyz)
    center = [(accumulator.min_xyz[index] + accumulator.max_xyz[index]) * 0.5 for index in range(3)]
    return all(
        min_xyz[index] - dimensions[index] * 0.25 <= center[index] <= max_xyz[index] + dimensions[index] * 0.25
        for index in range(3)
    )


def split_head_core_parts(
    owner_bone: int,
    head_core_faces: list[SplitFace],
    accumulators: dict[PartKey, BBoxAccumulator],
    report_accumulators: dict[str, BBoxAccumulator],
    report_methods: dict[str, str],
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int,
) -> None:
    if len(head_core_faces) < HEAD_CORE_SPLIT_MIN_FACES:
        return

    head_core_key = PartKey(owner_bone, "head_core")
    if head_core_key not in accumulators or "head_core" not in report_accumulators:
        return

    part_min, part_max = faces_bbox(head_core_faces)
    dimensions = bbox_dimensions(part_min, part_max)
    if max(dimensions, default=0.0) <= EPSILON:
        return

    split_parts: list[tuple[str, list[SplitFace]]] = [("head_core", head_core_faces)]
    split_parts = split_faces_by_axis(
        split_parts,
        axis=2,
        segment_count=2,
        suffixes=head_core_depth_suffixes(front_sign),
    )
    split_parts, _merged_depth = merge_tiny_named_parts(split_parts, HEAD_CORE_SPLIT_MIN_BUCKET_FACES)
    split_parts = split_faces_by_axis(
        split_parts,
        axis=1,
        segment_count=2,
        suffixes=vertical_suffixes(2),
    )
    split_parts, _merged_height = merge_tiny_named_parts(split_parts, HEAD_CORE_SPLIT_MIN_BUCKET_FACES)

    if len(split_parts) <= 1:
        return

    del accumulators[head_core_key]
    report_accumulators.pop("head_core", None)
    report_methods.pop("head_core", None)

    for part_name, part_faces in split_parts:
        accumulator = accumulators.setdefault(PartKey(owner_bone, part_name), BBoxAccumulator())
        accumulator.is_complex_split = True
        report_accumulator = report_accumulators.setdefault(part_name, BBoxAccumulator())
        for face in part_faces:
            accumulator.add_face(face.points, face.vertex_keys)
            report_accumulator.add_face(face.points, face.vertex_keys)
        report_methods[part_name] = "spatial_region"


def protect_explicit_face_feature_visibility(
    owner_bone: int,
    accumulators: dict[PartKey, BBoxAccumulator],
    report_accumulators: dict[str, BBoxAccumulator] | None,
    feature_bbox: tuple[list[float], list[float]],
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int,
    config: FaceFeatureProtectionConfig,
    bones: dict[int, BonePartition],
    protected_feature_names: tuple[str, ...] = (),
) -> list[FaceFeatureProtectionAction]:
    feature_min, feature_max = feature_bbox
    owner_min, owner_max = owner_bbox
    owner_dimensions = bbox_dimensions(owner_min, owner_max)
    depth_margin = max(owner_dimensions[2] * config.margin_ratio, EPSILON)
    height_margin = max(owner_dimensions[1] * config.margin_ratio, EPSILON)
    actions: list[FaceFeatureProtectionAction] = []
    bone = bones.get(owner_bone)
    owner_bone_name = bone.name if bone is not None else f"bone_{owner_bone}"

    for part_key, accumulator in list(accumulators.items()):
        if part_key.owner_bone != owner_bone:
            continue
        if accumulator.min_xyz is None or accumulator.max_xyz is None:
            continue

        report_accumulator = report_accumulators.get(part_key.name) if report_accumulators is not None else None
        if part_key.name == "head_core" or part_key.name.startswith("head_core_"):
            if not config.protect_head_core_front:
                continue
            if not accumulator_overlaps_bbox_axes(accumulator, feature_bbox, (0, 1)):
                continue
            before_bbox = accumulator_bbox(accumulator)
            target_value = feature_min[2] - depth_margin if front_sign >= 0 else feature_max[2] + depth_margin
            clamped = clamp_front_axis_behind_features(accumulator, feature_min, feature_max, depth_margin, front_sign)
            if clamped and report_accumulator is not None:
                clamp_front_axis_behind_features(report_accumulator, feature_min, feature_max, depth_margin, front_sign)
            if clamped:
                actions.append(
                    face_feature_protection_action(
                        owner_bone,
                        owner_bone_name,
                        part_key.name,
                        "clamp_head_core_behind_features",
                        before_bbox,
                        accumulator_bbox(accumulator),
                        feature_bbox,
                        protected_feature_names=protected_feature_names,
                        axis="z",
                        target_value=target_value,
                        margin=depth_margin,
                        front_sign=front_sign,
                        overlap_axes=("x", "y"),
                    )
                )
            continue

        if part_key.name.startswith("hair_front") and accumulator_overlaps_bbox_axes(accumulator, feature_bbox, (0, 1)):
            if not config.protect_hair_front:
                continue
            before_bbox = accumulator_bbox(accumulator)
            target_value = feature_max[1] + height_margin
            if clamp_hair_above_features(accumulator, target_value):
                if report_accumulator is not None:
                    clamp_hair_above_features(report_accumulator, target_value)
                actions.append(
                    face_feature_protection_action(
                        owner_bone,
                        owner_bone_name,
                        part_key.name,
                        "raise_hair_front_above_features",
                        before_bbox,
                        accumulator_bbox(accumulator),
                        feature_bbox,
                        protected_feature_names=protected_feature_names,
                        axis="y",
                        target_value=target_value,
                        margin=height_margin,
                        front_sign=front_sign,
                        overlap_axes=("x", "y"),
                    )
                )
    return actions

def apply_explicit_face_feature_visibility_protection(
    accumulators: dict[PartKey, BBoxAccumulator],
    bones: dict[int, BonePartition],
    config: FaceFeatureProtectionConfig,
) -> list[FaceFeatureProtectionAction]:
    if not config.enabled:
        return []
    actions: list[FaceFeatureProtectionAction] = []
    owner_bones = {
        part_key.owner_bone
        for part_key, accumulator in accumulators.items()
        if accumulator.min_xyz is not None
        and accumulator.max_xyz is not None
        and is_head_visibility_part_name(part_key.name)
    }
    for owner_bone in sorted(owner_bones):
        owner_accumulators = [
            accumulator
            for part_key, accumulator in accumulators.items()
            if part_key.owner_bone == owner_bone
            and accumulator.min_xyz is not None
            and accumulator.max_xyz is not None
        ]
        if not owner_accumulators:
            continue
        owner_bbox = combined_bbox(owner_accumulators)
        raw_feature_bbox = explicit_face_feature_accumulators_bbox(accumulators, owner_bone, owner_bbox, config)
        if raw_feature_bbox is None:
            continue
        front_sign = infer_face_feature_front_sign(owner_bbox, raw_feature_bbox)
        feature_bbox = explicit_face_feature_visibility_bbox(
            accumulators,
            owner_bone,
            owner_bbox,
            front_sign,
            config,
        ) or raw_feature_bbox
        protected_feature_names = explicit_face_feature_names(accumulators, owner_bone, owner_bbox, config)
        actions.extend(
            protect_explicit_face_feature_visibility(
                owner_bone,
                accumulators,
                None,
                feature_bbox,
                owner_bbox,
                front_sign,
                config,
                bones,
                protected_feature_names,
            )
        )
    return actions


def is_head_visibility_part_name(name: str) -> bool:
    return name == "head_core" or name.startswith("head_core_") or name.startswith("hair_front")


def accumulator_bbox(accumulator: BBoxAccumulator) -> tuple[list[float], list[float]]:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return accumulator.min_xyz.copy(), accumulator.max_xyz.copy()


def face_feature_protection_action(
    owner_bone: int,
    owner_bone_name: str,
    part_name: str,
    action: str,
    before_bbox: tuple[list[float], list[float]],
    after_bbox: tuple[list[float], list[float]],
    feature_bbox: tuple[list[float], list[float]],
    *,
    protected_feature_names: tuple[str, ...] = (),
    axis: str | None = None,
    target_value: float | None = None,
    margin: float | None = None,
    front_sign: int | None = None,
    overlap_axes: tuple[str, ...] = (),
) -> FaceFeatureProtectionAction:
    return FaceFeatureProtectionAction(
        owner_bone=owner_bone,
        owner_bone_name=owner_bone_name,
        adjusted_part_name=part_name,
        cube_name=f"{part_name}_cube",
        action=action,
        before_bbox=(before_bbox[0].copy(), before_bbox[1].copy()),
        after_bbox=(after_bbox[0].copy(), after_bbox[1].copy()),
        feature_bbox=(feature_bbox[0].copy(), feature_bbox[1].copy()),
        protected_feature_names=protected_feature_names,
        axis=axis,
        target_value=target_value,
        margin=margin,
        front_sign=front_sign,
        overlap_axes=overlap_axes,
    )

def infer_face_feature_front_sign(
    owner_bbox: tuple[list[float], list[float]],
    feature_bbox: tuple[list[float], list[float]],
) -> int:
    owner_min, owner_max = owner_bbox
    feature_min, feature_max = feature_bbox
    owner_center_z = (owner_min[2] + owner_max[2]) * 0.5
    feature_center_z = (feature_min[2] + feature_max[2]) * 0.5
    return 1 if feature_center_z >= owner_center_z else -1


def clamp_front_axis_behind_features(
    accumulator: BBoxAccumulator,
    feature_min: list[float],
    feature_max: list[float],
    margin: float,
    front_sign: int,
) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    changed = False
    if front_sign >= 0:
        target_max = feature_min[2] - margin
        if accumulator.max_xyz[2] > target_max and abs(accumulator.max_xyz[2] - target_max) > EPSILON:
            accumulator.max_xyz[2] = target_max
            if accumulator.min_xyz[2] > target_max:
                accumulator.min_xyz[2] = target_max
            changed = True
    else:
        target_min = feature_max[2] + margin
        if accumulator.min_xyz[2] < target_min and abs(accumulator.min_xyz[2] - target_min) > EPSILON:
            accumulator.min_xyz[2] = target_min
            if accumulator.max_xyz[2] < target_min:
                accumulator.max_xyz[2] = target_min
            changed = True
    if changed:
        reset_accumulator_points_to_bbox(accumulator, "face_feature_protected")
    return changed


def clamp_hair_above_features(accumulator: BBoxAccumulator, min_y: float) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    if accumulator.min_xyz[1] >= min_y or accumulator.max_xyz[1] <= min_y:
        return False
    accumulator.min_xyz[1] = min_y
    reset_accumulator_points_to_bbox(accumulator, "face_feature_protected")
    return True


def accumulator_overlaps_bbox_axes(
    accumulator: BBoxAccumulator,
    bbox: tuple[list[float], list[float]],
    axes: tuple[int, ...],
) -> bool:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return False
    min_xyz, max_xyz = bbox
    for axis in axes:
        if accumulator.max_xyz[axis] < min_xyz[axis] or accumulator.min_xyz[axis] > max_xyz[axis]:
            return False
    return True


def reset_accumulator_points_to_bbox(accumulator: BBoxAccumulator, key_prefix: str) -> None:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return
    accumulator.points_by_vertex = {
        (key_prefix, index): point.copy()
        for index, point in enumerate(bbox_corners(accumulator.min_xyz, accumulator.max_xyz))
    }


def head_core_depth_suffixes(front_sign: int) -> list[str]:
    if front_sign >= 0:
        return ["back", "front"]
    return ["front", "back"]


def merge_tiny_named_parts(
    parts: list[tuple[str, list[SplitFace]]],
    min_faces: int,
) -> tuple[list[tuple[str, list[SplitFace]]], int]:
    if min_faces <= 0 or len(parts) < 2:
        return parts, 0

    kept: list[list[Any]] = []
    tiny: list[tuple[str, list[SplitFace]]] = []
    for name, part_faces in parts:
        if len(part_faces) < min_faces:
            tiny.append((name, part_faces))
        else:
            kept.append([name, part_faces])

    if not kept or not tiny:
        return parts, 0

    merged_count = 0
    for _tiny_name, tiny_faces in tiny:
        tiny_centroid = points_centroid([point for face in tiny_faces for point in face.points])
        target_index = min(
            range(len(kept)),
            key=lambda index: squared_distance(
                tiny_centroid,
                points_centroid([point for face in kept[index][1] for point in face.points]),
            ),
        )
        kept[target_index][1].extend(tiny_faces)
        merged_count += 1

    return [(name, part_faces) for name, part_faces in kept], merged_count


def is_side_feature_part(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in SIDE_FEATURE_PART_PREFIXES)


def classify_head_component_spatial(
    faces: list[SplitFace],
    owner_bbox: tuple[list[float], list[float]],
    front_sign: int = DEFAULT_HEAD_FRONT_SIGN,
) -> str:
    centroid = points_centroid([point for face in faces for point in face.points])
    min_xyz, max_xyz = owner_bbox
    width = max_xyz[0] - min_xyz[0]
    depth = max_xyz[2] - min_xyz[2]
    height = max_xyz[1] - min_xyz[1]
    if depth > EPSILON:
        if front_sign >= 0:
            if centroid[2] >= max_xyz[2] - depth * 0.35:
                return "hair_front"
            if centroid[2] <= min_xyz[2] + depth * 0.35:
                return "hair_back"
        else:
            if centroid[2] <= min_xyz[2] + depth * 0.35:
                return "hair_front"
            if centroid[2] >= max_xyz[2] - depth * 0.35:
                return "hair_back"
    if width > EPSILON:
        if centroid[0] <= min_xyz[0] + width * 0.25:
            return "hair_side_l"
        if centroid[0] >= max_xyz[0] - width * 0.25:
            return "hair_side_r"
    if height > EPSILON and centroid[1] >= max_xyz[1] - height * 0.2:
        return "head_accessory"
    return "head_accessory"


def faces_bbox(faces: list[SplitFace]) -> tuple[list[float], list[float]]:
    accumulator = BBoxAccumulator()
    for face in faces:
        for point in face.points:
            accumulator.add_point(point)
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return accumulator.min_xyz, accumulator.max_xyz


def apply_cleanup(
    accumulators: dict[PartKey, BBoxAccumulator],
    bones: dict[int, BonePartition],
    cleanup: CleanupConfig,
    warnings: list[str],
    regular_faces_by_part: dict[PartKey, list[SplitFace]] | None = None,
    protected_part_keys: set[PartKey] | None = None,
) -> CleanupReport:
    report = CleanupReport()
    if not cleanup.delete_small_parts and not cleanup.merge_small_parts_to_parent:
        return report
    if cleanup.min_faces <= 0 and cleanup.min_bbox_volume <= 0:
        warnings.append("Cleanup was enabled but no small-part thresholds were set; skipped cleanup.")
        return report

    protected_part_keys = protected_part_keys or set()
    apply_regular_connected_component_cleanup(accumulators, bones, cleanup, regular_faces_by_part or {}, report, protected_part_keys)

    for part_key, accumulator in sorted(list(accumulators.items()), key=lambda item: (item[0].owner_bone, item[0].name)):
        if accumulators.get(part_key) is not accumulator:
            continue
        if accumulator.min_xyz is None or accumulator.max_xyz is None or accumulator.faces <= 0:
            continue

        reason = small_part_reason(accumulator, cleanup)
        if reason is None:
            continue
        if part_key in protected_part_keys:
            continue

        if cleanup.merge_small_parts_to_parent:
            target_bone = cleanup_target_parent(part_key.owner_bone, bones)
            if target_bone is not None:
                target = bones[target_bone]
                target_key = PartKey(target_bone, target.name)
                merge_accumulator(accumulators.setdefault(target_key, BBoxAccumulator()), accumulator)
                del accumulators[part_key]
                report.merged_parts.append(
                    cleanup_part_report(part_key, accumulator, bones, "merged_to_parent", reason, target_bone)
                )
                continue

        if cleanup.delete_small_parts:
            del accumulators[part_key]
            report.deleted_parts.append(cleanup_part_report(part_key, accumulator, bones, "deleted", reason, None))
        else:
            report.kept_small_parts.append(cleanup_part_report(part_key, accumulator, bones, "kept", reason, None))

    return report


def apply_regular_connected_component_cleanup(
    accumulators: dict[PartKey, BBoxAccumulator],
    bones: dict[int, BonePartition],
    cleanup: CleanupConfig,
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
    report: CleanupReport,
    protected_part_keys: set[PartKey] | None = None,
) -> None:
    pending_merges: list[tuple[int, BBoxAccumulator]] = []
    protected_part_keys = protected_part_keys or set()
    for part_key, faces in sorted(regular_faces_by_part.items(), key=lambda item: (item[0].owner_bone, item[0].name)):
        if part_key not in accumulators or len(faces) < 2:
            continue
        if part_key in protected_part_keys:
            continue

        component_indices = connected_cleanup_face_components(faces)
        if len(component_indices) < 2:
            continue

        kept_components: list[BBoxAccumulator] = []
        touched_small_component = False
        for component_number, component in enumerate(component_indices, start=1):
            component_accumulator = accumulator_from_split_faces([faces[index] for index in component])
            reason = small_part_reason(component_accumulator, cleanup)
            if reason is None:
                kept_components.append(component_accumulator)
                continue

            touched_small_component = True
            component_name = f"{part_key.name}_component_{component_number}"
            component_key = PartKey(part_key.owner_bone, component_name)
            target_bone = cleanup_target_parent(part_key.owner_bone, bones)
            if cleanup.merge_small_parts_to_parent and target_bone is not None:
                pending_merges.append((target_bone, component_accumulator))
                report.merged_parts.append(
                    cleanup_part_report(component_key, component_accumulator, bones, "merged_to_parent", reason, target_bone)
                )
            elif cleanup.delete_small_parts:
                report.deleted_parts.append(cleanup_part_report(component_key, component_accumulator, bones, "deleted", reason, None))
            else:
                kept_components.append(component_accumulator)
                report.kept_small_parts.append(cleanup_part_report(component_key, component_accumulator, bones, "kept", reason, None))

        if not touched_small_component:
            continue

        if kept_components:
            rebuilt = BBoxAccumulator()
            for component_accumulator in kept_components:
                merge_accumulator(rebuilt, component_accumulator)
            accumulators[part_key] = rebuilt
        else:
            del accumulators[part_key]

    for target_bone, component_accumulator in pending_merges:
        target = bones[target_bone]
        target_key = PartKey(target_bone, target.name)
        merge_accumulator(accumulators.setdefault(target_key, BBoxAccumulator()), component_accumulator)


def accumulator_from_split_faces(faces: list[SplitFace]) -> BBoxAccumulator:
    accumulator = BBoxAccumulator()
    for face in faces:
        accumulator.add_face(face.points, face.vertex_keys)
    return accumulator


def connected_cleanup_face_components(faces: list[SplitFace]) -> list[list[int]]:
    parent = list(range(len(faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_face_by_vertex: dict[tuple[str, int], int] = {}
    first_face_by_point: dict[tuple[float, float, float], int] = {}
    for face_index, face in enumerate(faces):
        for vertex_key in face.vertex_keys:
            other = first_face_by_vertex.get(vertex_key)
            if other is None:
                first_face_by_vertex[vertex_key] = face_index
            else:
                union(other, face_index)
        for point in face.points:
            point_key = rounded_point_key(point)
            other = first_face_by_point.get(point_key)
            if other is None:
                first_face_by_point[point_key] = face_index
            else:
                union(other, face_index)

    components: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        components.setdefault(find(face_index), []).append(face_index)
    return sorted(components.values(), key=lambda component: component[0] if component else 0)


def rounded_point_key(point: list[float]) -> tuple[float, float, float]:
    return (round(point[0], 9), round(point[1], 9), round(point[2], 9))


def cleanup_target_parent(owner_bone: int, bones: dict[int, BonePartition]) -> int | None:
    bone = bones.get(owner_bone)
    if bone is None or bone.parent not in bones:
        return None
    return bone.parent


def small_part_reason(accumulator: BBoxAccumulator, cleanup: CleanupConfig) -> str | None:
    reasons: list[str] = []
    if cleanup.min_faces > 0 and accumulator.faces < cleanup.min_faces:
        reasons.append(f"faces<{cleanup.min_faces}")
    volume = accumulator_bbox_volume(accumulator)
    if cleanup.min_bbox_volume > 0 and volume < cleanup.min_bbox_volume:
        reasons.append(f"bbox_volume<{cleanup.min_bbox_volume:g}")
    return ", ".join(reasons) if reasons else None


def merge_accumulator(target: BBoxAccumulator, source: BBoxAccumulator) -> None:
    if source.min_xyz is not None:
        target.add_point(source.min_xyz)
    if source.max_xyz is not None:
        target.add_point(source.max_xyz)
    target.faces += source.faces
    target.vertices.update(source.vertices)
    target.points_by_vertex.update(source.points_by_vertex)
    target.is_complex_split = target.is_complex_split or source.is_complex_split


def accumulator_bbox_volume(accumulator: BBoxAccumulator) -> float:
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return 0.0
    volume = 1.0
    for index in range(3):
        volume *= max(accumulator.max_xyz[index] - accumulator.min_xyz[index], 0.0)
    return volume


def cleanup_part_report(
    part_key: PartKey,
    accumulator: BBoxAccumulator,
    bones: dict[int, BonePartition],
    action: str,
    reason: str,
    target_bone: int | None,
) -> CleanupPartReport:
    bone = bones.get(part_key.owner_bone)
    target = bones.get(target_bone) if target_bone is not None else None
    return CleanupPartReport(
        owner_bone=part_key.owner_bone,
        owner_bone_name=bone.name if bone is not None else f"bone_{part_key.owner_bone}",
        name=part_key.name,
        action=action,
        reason=reason,
        faces=accumulator.faces,
        vertices=len(accumulator.vertices),
        bbox_volume=accumulator_bbox_volume(accumulator),
        target_bone=target_bone,
        target_bone_name=target.name if target is not None else None,
    )


def build_cuboids(
    accumulators: dict[PartKey, BBoxAccumulator],
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
    scale: float,
    offset: list[float],
    oriented_cubes: OrientedCubesConfig,
    vrm_humanoid_nodes: dict[str, set[int]],
    *,
    auto_orient: bool = False,
) -> tuple[list[Cuboid], list[OrientedCubeReport], list[OrientationDecisionReport]]:
    cuboids: list[Cuboid] = []
    reports: list[OrientedCubeReport] = []
    decisions: list[OrientationDecisionReport] = []
    for part_key, accumulator in sorted(accumulators.items(), key=lambda item: (item[0].owner_bone, item[0].name)):
        if accumulator.min_xyz is None or accumulator.max_xyz is None:
            continue
        bone = bones.get(part_key.owner_bone)
        owner_bone_name = bone.name if bone is not None else f"bone_{part_key.owner_bone}"
        world_matrix = world_matrices.get(part_key.owner_bone, identity_matrix())
        origin = to_blockbench_space(
            matrix_translation(world_matrix), scale, offset
        )
        rotation = [0.0, 0.0, 0.0]
        rotation_source = None

        from_xyz = to_blockbench_space(accumulator.min_xyz, scale, offset)
        to_xyz = to_blockbench_space(accumulator.max_xyz, scale, offset)
        original_from_xyz, original_to_xyz = ensure_min_cube_size(from_xyz, to_xyz)
        original_volume = box_volume(original_from_xyz, original_to_xyz)
        oriented_volume: float | None = None
        cube_name = f"{part_key.name}_cube"

        if should_orient_accumulator(part_key, accumulator, bones, oriented_cubes, vrm_humanoid_nodes):
            rotation_matrix = normalized_rotation_matrix(world_matrix)
            candidate_rotation = matrix_to_euler_xyz_degrees(rotation_matrix)
            candidate_source = "bone_world_matrix"
            if not has_nonzero_rotation(candidate_rotation) and not accumulator.is_complex_split:
                direction_matrix = bone_direction_rotation_matrix(part_key.owner_bone, bones, world_matrices)
                if direction_matrix is not None:
                    direction_rotation = matrix_to_euler_xyz_degrees(direction_matrix)
                    if has_nonzero_rotation(direction_rotation):
                        rotation_matrix = direction_matrix
                        candidate_rotation = direction_rotation
                        candidate_source = "bone_direction"
            if has_nonzero_rotation(candidate_rotation):
                from_xyz, to_xyz = oriented_accumulator_bounds(accumulator, scale, offset, origin, rotation_matrix)
                score_from_xyz, score_to_xyz = ensure_min_cube_size(from_xyz, to_xyz)
                oriented_volume = box_volume(score_from_xyz, score_to_xyz)
                rotation = candidate_rotation
                rotation_source = candidate_source
                decisions.append(
                    orientation_decision_report(
                        cube_name,
                        part_key,
                        owner_bone_name,
                        candidate_source,
                        True,
                        "accepted",
                        original_volume,
                        oriented_volume,
                        candidate_rotation,
                    )
                )
        elif auto_orient:
            auto_candidate, auto_decisions = auto_orient_accumulator(
                part_key,
                accumulator,
                bones,
                world_matrices,
                scale,
                offset,
                origin,
                original_from_xyz,
                original_to_xyz,
            )
            decisions.extend(auto_decisions)
            if auto_candidate is not None:
                from_xyz, to_xyz, rotation, rotation_source, oriented_volume = auto_candidate

        from_xyz, to_xyz = ensure_min_cube_size(
            from_xyz,
            to_xyz,
            allow_zero_axes=zero_thickness_axes(from_xyz, to_xyz),
        )
        cuboids.append(
            Cuboid(
                owner_bone=part_key.owner_bone,
                owner_bone_name=owner_bone_name,
                name=cube_name,
                from_xyz=from_xyz,
                to_xyz=to_xyz,
                origin=origin,
                faces=accumulator.faces,
                vertices=len(accumulator.vertices),
                rotation=rotation,
                rotation_source=rotation_source,
            )
        )
        if rotation_source is not None:
            reports.append(
                OrientedCubeReport(
                    name=cube_name,
                    owner_bone=part_key.owner_bone,
                    owner_bone_name=owner_bone_name,
                    rotation=rotation,
                    source=rotation_source,
                    reason="accepted",
                    original_bbox_volume=original_volume,
                    oriented_bbox_volume=oriented_volume,
                    cube_only_compatible=True,
                )
            )
    return cuboids, reports, decisions


def auto_orient_accumulator(
    part_key: PartKey,
    accumulator: BBoxAccumulator,
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
    scale: float,
    offset: list[float],
    origin: list[float],
    original_from_xyz: list[float],
    original_to_xyz: list[float],
) -> tuple[tuple[list[float], list[float], list[float], str, float] | None, list[OrientationDecisionReport]]:
    cube_name = f"{part_key.name}_cube"
    decisions: list[OrientationDecisionReport] = []
    bone = bones.get(part_key.owner_bone)
    if bone is None:
        return None, decisions
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return None, decisions
    if is_face_feature_name(part_key.name) or is_face_feature_name(bone.name):
        return None, decisions
    if accumulator.is_complex_split and (
        is_head_core_part(part_key.name) or is_face_feature_part(part_key.name) or is_side_feature_part(part_key.name)
    ):
        return None, decisions
    if accumulator.is_complex_split and accumulator.faces < AUTO_ORIENT_MIN_FACES:
        return None, decisions

    original_dimensions = bbox_dimensions(original_from_xyz, original_to_xyz)
    original_volume = box_volume_from_dimensions(original_dimensions)
    if max(original_dimensions) < AUTO_ORIENT_MIN_LONG_DIM or original_volume < AUTO_ORIENT_MIN_VOLUME:
        return None, decisions

    best_candidate: tuple[list[float], list[float], list[float], str, float | None, str] | None = None
    direction_matrix = bone_direction_rotation_matrix(part_key.owner_bone, bones, world_matrices)
    if direction_matrix is not None:
        direction_candidate = auto_orient_candidate(
            accumulator,
            scale,
            offset,
            origin,
            direction_matrix,
            original_volume,
            "auto_bone_direction",
        )
        decisions.append(
            orientation_decision_report(
                cube_name,
                part_key,
                bone.name,
                "auto_bone_direction",
                direction_candidate[5] == "accepted",
                direction_candidate[5],
                original_volume,
                direction_candidate[4],
                direction_candidate[2],
            )
        )
        if direction_candidate[5] == "accepted":
            return direction_candidate[:5], decisions

    if should_try_geometry_auto_orient(part_key):
        geometry_min_reduction = (
            HEAD_DETAIL_AUTO_ORIENT_MIN_VOLUME_REDUCTION
            if accumulator.is_complex_split and (is_hair_part(part_key.name) or part_key.name.startswith("head_accessory"))
            else AUTO_ORIENT_MIN_VOLUME_REDUCTION
        )
        for geometry_matrix in geometry_auto_orientation_matrices(accumulator, scale, offset):
            geometry_candidate = auto_orient_candidate(
                accumulator,
                scale,
                offset,
                origin,
                geometry_matrix,
                original_volume,
                "auto_geometry_pca",
                geometry_min_reduction,
            )
            if geometry_candidate is None:
                continue
            decisions.append(
                orientation_decision_report(
                    cube_name,
                    part_key,
                    bone.name,
                    "auto_geometry_pca",
                    geometry_candidate[5] == "accepted",
                    geometry_candidate[5],
                    original_volume,
                    geometry_candidate[4],
                    geometry_candidate[2],
                )
            )
            if geometry_candidate[5] != "accepted":
                continue
            if best_candidate is None or geometry_candidate[4] < best_candidate[4]:
                best_candidate = geometry_candidate

    return (None, decisions) if best_candidate is None else (best_candidate[:5], decisions)


def auto_orient_candidate(
    accumulator: BBoxAccumulator,
    scale: float,
    offset: list[float],
    origin: list[float],
    rotation_matrix: list[list[float]],
    original_volume: float,
    source: str,
    min_volume_reduction: float = AUTO_ORIENT_MIN_VOLUME_REDUCTION,
) -> tuple[list[float], list[float], list[float], str, float | None, str]:
    rotation = matrix_to_euler_xyz_degrees(rotation_matrix)
    if not has_nonzero_rotation(rotation):
        return [], [], rotation, source, None, "axis_aligned_candidate"
    from_xyz, to_xyz = oriented_accumulator_bounds(accumulator, scale, offset, origin, rotation_matrix)
    score_from_xyz, score_to_xyz = ensure_min_cube_size(from_xyz, to_xyz)
    oriented_volume = box_volume(score_from_xyz, score_to_xyz)
    if oriented_volume >= original_volume * (1.0 - min_volume_reduction):
        return from_xyz, to_xyz, rotation, source, oriented_volume, "insufficient_volume_reduction"
    return from_xyz, to_xyz, rotation, source, oriented_volume, "accepted"


def orientation_decision_report(
    cube_name: str,
    part_key: PartKey,
    owner_bone_name: str,
    source: str,
    accepted: bool,
    reason: str,
    original_volume: float,
    oriented_volume: float | None,
    rotation: list[float],
) -> OrientationDecisionReport:
    return OrientationDecisionReport(
        name=cube_name,
        owner_bone=part_key.owner_bone,
        owner_bone_name=owner_bone_name,
        source=source,
        accepted=accepted,
        reason=reason,
        original_bbox_volume=original_volume,
        oriented_bbox_volume=oriented_volume,
        cube_only_compatible=True,
        rotation=rotation,
    )


def should_try_geometry_auto_orient(part_key: PartKey) -> bool:
    return not (is_head_core_part(part_key.name) or is_face_feature_part(part_key.name) or is_side_feature_part(part_key.name))


def geometry_auto_orientation_matrices(
    accumulator: BBoxAccumulator,
    scale: float,
    offset: list[float],
) -> list[list[list[float]]]:
    source_points = list(accumulator.points_by_vertex.values()) or bbox_corners(accumulator.min_xyz, accumulator.max_xyz)
    points = [to_blockbench_space(point, scale, offset) for point in source_points]
    matrices: list[list[list[float]]] = []

    principal_axis = principal_axis_3d(points)
    if principal_axis is not None:
        matrix = rotation_matrix_from_y_axis(principal_axis)
        if matrix is not None:
            matrices.append(matrix)

    for matrix in planar_principal_rotation_matrices(points):
        matrices.append(matrix)
    return matrices


def should_orient_accumulator(
    part_key: PartKey,
    accumulator: BBoxAccumulator,
    bones: dict[int, BonePartition],
    config: OrientedCubesConfig,
    vrm_humanoid_nodes: dict[str, set[int]],
) -> bool:
    if not config.enabled:
        return False
    if config.scope == "complex_split_parts" and not accumulator.is_complex_split:
        return False
    if config.scope == "bone_cubes" and accumulator.is_complex_split:
        return False
    bone = bones.get(part_key.owner_bone)
    if bone is None:
        return False
    haystack = bone.name if config.case_sensitive else bone.name.casefold()
    for bone_name in config.bones or (DEFAULT_COMPLEX_SPLIT_BONE,):
        if part_key.owner_bone in vrm_humanoid_nodes.get(bone_name, set()):
            return True
        aliases = COMPLEX_BONE_ALIASES.get(bone_name, (bone_name,))
        for alias in aliases:
            needle = alias if config.case_sensitive else alias.casefold()
            if needle and needle in haystack:
                return True
    return False


def bone_direction_rotation_matrix(
    bone_index: int,
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
) -> list[list[float]] | None:
    bone = bones.get(bone_index)
    if bone is None:
        return None
    origin = matrix_translation(world_matrices.get(bone_index, identity_matrix()))
    target: list[float] | None = None
    for child_index in bone.children:
        if child_index in world_matrices:
            target = matrix_translation(world_matrices[child_index])
            break
    if target is None and bone.parent in world_matrices:
        parent = matrix_translation(world_matrices[bone.parent])
        direction = [origin[index] - parent[index] for index in range(3)]
    elif target is not None:
        direction = [target[index] - origin[index] for index in range(3)]
    else:
        return None
    return rotation_matrix_from_y_axis(direction)


__all__ = [
    "convert_model",
    "convert_result_to_dict",
    "format_convert_summary",
    "write_convert_report",
]
