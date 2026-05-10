from __future__ import annotations

import json
import math
import struct
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import (
    CleanupConfig,
    ComplexSplitConfig,
    HybridDetailSplitConfig,
    OrientedCubesConfig,
    ProcessingConfig,
    resolve_processing_config,
)
from .constants import MODE_TRIANGLES, MODE_TRIANGLE_FAN, MODE_TRIANGLE_STRIP
from .errors import ConvertError, InspectError
from .inspect import SUPPORTED_MODEL_SUFFIXES, build_parent_map, is_valid_index, load_gltf, read_accessor
from .partition import (
    BonePartition,
    BoneResolutionReport,
    PartitionReport,
    bone_resolution_to_dict,
    build_filtered_bone_partitions,
    choose_face_owner,
    fallback_joint_for_skin,
    read_faces,
    read_optional_accessor,
    resolve_bone_node,
)


EPSILON = 1e-6
MIN_CUBE_SIZE = 0.01
ZERO_THICKNESS_DIMENSION_RATIO = 0.05
ZERO_THICKNESS_MIN_PLANE_DIMENSION = MIN_CUBE_SIZE * 8.0
DEFAULT_COMPLEX_SPLIT_BONE = "head"
SUPPORTED_CONVERT_MODES = {"cuboid", "hybrid"}
HYBRID_SPECIAL_CUBE_BONES = ("head", "hair", "skirt", "coat", "accessory")
QUALITY_LARGEST_CUBES_LIMIT = 10
QUALITY_TINY_FRAGMENT_CUBES_LIMIT = 20
QUALITY_ELONGATED_DIMENSION_RATIO = 6.0
QUALITY_TINY_FRAGMENT_MAX_FACES = 2
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
HAIR_BUCKET_MIN_FACES = 4
HAIR_BUCKET_MIN_FACE_RATIO = 0.08
HAIR_BUCKET_OVERLAP_RATIO = 0.015
HAIR_BUCKET_OVERLAP_MIN = MIN_CUBE_SIZE * 0.25
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
    "head_accessory": ("ribbon", "accessory", "hat", "リボン", "头饰", "頭飾", "头带", "頭帶", "饰", "飾"),
    "head_core": ("face", "skin", "head", "顔", "肌", "脸", "臉", "皮肤", "皮膚"),
}
FACE_FEATURE_PART_PREFIXES = ("eye", "mouth", "brow", "eyelash", "nose")
SIDE_FEATURE_PART_PREFIXES = ("ear",)


@dataclass(frozen=True)
class PartKey:
    owner_bone: int
    name: str


@dataclass
class BBoxAccumulator:
    min_xyz: list[float] | None = None
    max_xyz: list[float] | None = None
    faces: int = 0
    vertices: set[tuple[str, int]] = field(default_factory=set)
    points_by_vertex: dict[tuple[str, int], list[float]] = field(default_factory=dict)
    is_complex_split: bool = False

    def add_face(self, points: list[list[float]], vertex_keys: list[tuple[str, int]]) -> None:
        if not points:
            return

        for point in points:
            self.add_point(point)
        self.vertices.update(vertex_keys)
        for point, vertex_key in zip(points, vertex_keys, strict=False):
            self.points_by_vertex.setdefault(vertex_key, point.copy())
        self.faces += 1

    def add_point(self, point: list[float]) -> None:
        if self.min_xyz is None or self.max_xyz is None:
            self.min_xyz = point.copy()
            self.max_xyz = point.copy()
            return

        for index, value in enumerate(point):
            self.min_xyz[index] = min(self.min_xyz[index], value)
            self.max_xyz[index] = max(self.max_xyz[index], value)


@dataclass
class Cuboid:
    owner_bone: int
    owner_bone_name: str
    name: str
    from_xyz: list[float]
    to_xyz: list[float]
    origin: list[float]
    faces: int
    vertices: int
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_source: str | None = None


@dataclass
class SplitFace:
    owner_bone: int
    bone_name: str
    points: list[list[float]]
    vertex_keys: list[tuple[str, int]]
    material_name: str | None


@dataclass
class AutoSpatialPart:
    name: str
    faces: list[SplitFace]
    min_xyz: list[float]
    max_xyz: list[float]


@dataclass
class HairSplitPart:
    name: str
    faces: list[SplitFace]
    split_axes: set[int] = field(default_factory=set)


@dataclass
class HairSplitResult:
    parts: list[HairSplitPart]
    merged_tiny_buckets: int = 0


@dataclass
class ComplexSplitSubpartReport:
    name: str
    method: str
    faces: int
    vertices: int


@dataclass
class ComplexSplitBoneReport:
    bone: int
    bone_name: str
    source_faces: int
    subparts: list[ComplexSplitSubpartReport]
    merged_tiny_components: int = 0
    deleted_tiny_components: int = 0
    merged_tiny_hair_buckets: int = 0
    expanded_hair_bucket_overlap: int = 0


@dataclass
class CleanupPartReport:
    owner_bone: int
    owner_bone_name: str
    name: str
    action: str
    reason: str
    faces: int
    vertices: int
    bbox_volume: float
    target_bone: int | None = None
    target_bone_name: str | None = None


@dataclass
class CleanupReport:
    deleted_parts: list[CleanupPartReport] = field(default_factory=list)
    merged_parts: list[CleanupPartReport] = field(default_factory=list)
    kept_small_parts: list[CleanupPartReport] = field(default_factory=list)


@dataclass
class OrientedCubeReport:
    name: str
    owner_bone: int
    owner_bone_name: str
    rotation: list[float]
    source: str


@dataclass(frozen=True)
class HybridModeReport:
    enabled: bool = False
    special_cube_bones: tuple[str, ...] = ()
    mesh_strategy: str = "none"
    cuboid_strategy: str = "one_bbox_per_resolved_bone"


@dataclass(frozen=True)
class SkippedUnskinnedMesh:
    node_index: int
    mesh_index: int


@dataclass
class ConvertResult:
    input_path: Path
    output_path: Path
    mode: str
    preset: str
    scale: float
    cubes: list[Cuboid]
    bone_resolution: BoneResolutionReport
    empty_bones: int
    small_cubes: int
    complex_split: list[ComplexSplitBoneReport]
    cleanup: CleanupReport
    oriented_cubes: list[OrientedCubeReport]
    hybrid: HybridModeReport
    hybrid_detail_split: HybridDetailSplitConfig
    skipped_unskinned_meshes: list[SkippedUnskinnedMesh]
    warnings: list[str]


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
            skipped_unskinned_meshes.append(SkippedUnskinnedMesh(node_index=node_index, mesh_index=mesh_index))
            warnings.append(f"Node {node_index} mesh {mesh_index} has no skin; skipped convert.")
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

    complex_split_report = apply_complex_split(split_faces, accumulators, config.complex_split, vrm_humanoid_nodes)
    complex_split_report.extend(
        apply_regular_detail_split(accumulators, regular_faces_by_part, bones, mode, config.hybrid_detail_split)
    )
    complex_split_report.extend(apply_auto_spatial_split(accumulators, regular_faces_by_part, bones, mode))
    cleanup_report = apply_cleanup(accumulators, bones, config.cleanup, warnings, regular_faces_by_part)
    populated = {
        part_key: accumulator
        for part_key, accumulator in accumulators.items()
        if accumulator.min_xyz is not None and accumulator.max_xyz is not None and accumulator.faces > 0
    }
    if not populated:
        raise ConvertError("no skinned triangle faces could be converted into cuboids")

    model_min, model_max = combined_bbox(populated.values())
    scale, offset = compute_scale_and_offset(model_min, model_max, target_height, warnings)
    cuboids, oriented_cube_report = build_cuboids(
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
        skipped_unskinned_meshes=skipped_unskinned_meshes,
        warnings=warnings,
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
    vrm_humanoid_nodes: dict[str, set[int]],
) -> list[ComplexSplitBoneReport]:
    faces_by_bone: dict[int, list[SplitFace]] = {}
    for face in split_faces:
        if face.points:
            faces_by_bone.setdefault(face.owner_bone, []).append(face)

    reports: list[ComplexSplitBoneReport] = []
    for owner_bone, faces in sorted(faces_by_bone.items()):
        if is_head_complex_split_bone(owner_bone, faces[0].bone_name, config, vrm_humanoid_nodes):
            reports.append(apply_head_complex_split(owner_bone, faces, accumulators, config))
        else:
            reports.append(apply_generic_complex_split(owner_bone, faces, accumulators, config))
    return reports


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
) -> ComplexSplitBoneReport:
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
    explicit_face_feature_bbox = explicit_face_feature_accumulators_bbox(accumulators, owner_bone, owner_bbox)
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
                split_result = split_hair_part_faces(part_name, part_faces, owner_bbox)
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
    split_head_core_parts(
        owner_bone,
        head_core_faces,
        accumulators,
        report_accumulators,
        report_methods,
        owner_bbox,
        front_sign,
    )

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
    )


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


def apply_regular_detail_split(
    accumulators: dict[PartKey, BBoxAccumulator],
    regular_faces_by_part: dict[PartKey, list[SplitFace]],
    bones: dict[int, BonePartition],
    mode: str,
    detail_split: HybridDetailSplitConfig,
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
        reports.append(
            ComplexSplitBoneReport(
                bone=part_key.owner_bone,
                bone_name=bone.name if bone is not None else part_key.name,
                source_faces=accumulator.faces,
                subparts=sorted(subparts, key=lambda item: item.name),
            )
        )
    return reports


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
    if not detail_split.by_connected_component or not should_split_single_material_components(part_name, faces):
        return [(sanitize_part_name(part_name), "regular_detail", faces)]
    return split_regular_faces_by_components(
        sanitize_part_name(part_name),
        faces,
        "regular_connected_component",
        detail_split,
    )


def should_split_single_material_components(part_name: str, faces: list[SplitFace]) -> bool:
    if any(matches_complex_alias(part_name, alias) for alias in ("hair", "skirt", "coat", "accessory")):
        return True
    material_part = classify_generic_material(faces)
    return material_part in {"hair", "head_accessory"}


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

        split_parts = auto_spatial_split_faces(part_key.name, faces, accumulator, model_height)
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


def accumulator_from_auto_spatial_part(part: AutoSpatialPart) -> BBoxAccumulator:
    accumulator = BBoxAccumulator(min_xyz=part.min_xyz.copy(), max_xyz=part.max_xyz.copy(), faces=len(part.faces))
    for face in part.faces:
        accumulator.vertices.update(face.vertex_keys)
        for point, vertex_key in zip(face.points, face.vertex_keys, strict=False):
            accumulator.points_by_vertex.setdefault(vertex_key, clamp_point_to_bbox(point, part.min_xyz, part.max_xyz))
    return accumulator


def clamp_point_to_bbox(point: list[float], min_xyz: list[float], max_xyz: list[float]) -> list[float]:
    return [min(max(point[index], min_xyz[index]), max_xyz[index]) for index in range(3)]


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


def bbox_dimensions(min_xyz: list[float], max_xyz: list[float]) -> list[float]:
    return [max(max_xyz[index] - min_xyz[index], 0.0) for index in range(3)]


def bbox_volume(min_xyz: list[float], max_xyz: list[float]) -> float:
    dimensions = bbox_dimensions(min_xyz, max_xyz)
    return box_volume_from_dimensions(dimensions)


def box_volume(min_xyz: list[float], max_xyz: list[float]) -> float:
    return box_volume_from_dimensions(bbox_dimensions(min_xyz, max_xyz))


def box_volume_from_dimensions(dimensions: list[float]) -> float:
    return dimensions[0] * dimensions[1] * dimensions[2]


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
        for pattern in ("eye", "目", "瞳", "brow", "眉", "eyelash", "まつげ", "睫", "mouth", "口")
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
) -> HairSplitResult:
    if len(faces) < 2:
        return HairSplitResult(parts=[HairSplitPart(part_name, faces)])

    part_min, part_max = faces_bbox(faces)
    owner_min, owner_max = owner_bbox
    part_width = part_max[0] - part_min[0]
    owner_width = owner_max[0] - owner_min[0]
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

    if len(faces) < 250 or part_height <= EPSILON or owner_height <= EPSILON or part_height < owner_height * 0.3:
        return HairSplitResult(parts=parts, merged_tiny_buckets=merged_tiny_buckets)

    segment_count = max(math.ceil(part_height / (owner_height * 0.3)), math.ceil(len(faces) / 2500))
    segment_count = max(2, min(segment_count, 4))
    parts = split_hair_parts_by_axis(parts, axis=1, segment_count=segment_count, suffixes=vertical_suffixes(segment_count))
    parts, merged = merge_tiny_hair_face_parts(parts, owner_bbox)
    merged_tiny_buckets += merged
    return HairSplitResult(parts=parts, merged_tiny_buckets=merged_tiny_buckets)


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
) -> tuple[list[float], list[float]] | None:
    feature_accumulators = [
        accumulator
        for part_key, accumulator in accumulators.items()
        if part_key.owner_bone != owner_bone
        and is_face_feature_name(part_key.name)
        and accumulator.min_xyz is not None
        and accumulator.max_xyz is not None
        and accumulator_center_near_bbox(accumulator, owner_bbox)
    ]
    if not feature_accumulators:
        return None
    return combined_bbox(feature_accumulators)


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


def points_centroid(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    return [sum(point[index] for point in points) / len(points) for index in range(3)]


def apply_cleanup(
    accumulators: dict[PartKey, BBoxAccumulator],
    bones: dict[int, BonePartition],
    cleanup: CleanupConfig,
    warnings: list[str],
    regular_faces_by_part: dict[PartKey, list[SplitFace]] | None = None,
) -> CleanupReport:
    report = CleanupReport()
    if not cleanup.delete_small_parts and not cleanup.merge_small_parts_to_parent:
        return report
    if cleanup.min_faces <= 0 and cleanup.min_bbox_volume <= 0:
        warnings.append("Cleanup was enabled but no small-part thresholds were set; skipped cleanup.")
        return report

    apply_regular_connected_component_cleanup(accumulators, bones, cleanup, regular_faces_by_part or {}, report)

    for part_key, accumulator in sorted(list(accumulators.items()), key=lambda item: (item[0].owner_bone, item[0].name)):
        if accumulators.get(part_key) is not accumulator:
            continue
        if accumulator.min_xyz is None or accumulator.max_xyz is None or accumulator.faces <= 0:
            continue

        reason = small_part_reason(accumulator, cleanup)
        if reason is None:
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
) -> None:
    pending_merges: list[tuple[int, BBoxAccumulator]] = []
    for part_key, faces in sorted(regular_faces_by_part.items(), key=lambda item: (item[0].owner_bone, item[0].name)):
        if part_key not in accumulators or len(faces) < 2:
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


def row_to_vec3(row: list[float | int]) -> list[float]:
    return [float(row[0] if len(row) > 0 else 0.0), float(row[1] if len(row) > 1 else 0.0), float(row[2] if len(row) > 2 else 0.0)]


def combined_bbox(accumulators: Any) -> tuple[list[float], list[float]]:
    model_min: list[float] | None = None
    model_max: list[float] | None = None
    for accumulator in accumulators:
        if accumulator.min_xyz is None or accumulator.max_xyz is None:
            continue
        if model_min is None or model_max is None:
            model_min = accumulator.min_xyz.copy()
            model_max = accumulator.max_xyz.copy()
            continue
        for index in range(3):
            model_min[index] = min(model_min[index], accumulator.min_xyz[index])
            model_max[index] = max(model_max[index], accumulator.max_xyz[index])

    if model_min is None or model_max is None:
        raise ConvertError("no valid cuboid bounds were produced")
    return model_min, model_max


def compute_scale_and_offset(
    model_min: list[float], model_max: list[float], target_height: float, warnings: list[str]
) -> tuple[float, list[float]]:
    height = model_max[1] - model_min[1]
    if height <= EPSILON:
        warnings.append("Model height is zero or nearly zero; target-height scaling was skipped.")
        scale = 1.0
    else:
        scale = target_height / height

    return scale, [
        (model_min[0] + model_max[0]) * 0.5,
        model_min[1],
        (model_min[2] + model_max[2]) * 0.5,
    ]


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
) -> tuple[list[Cuboid], list[OrientedCubeReport]]:
    cuboids: list[Cuboid] = []
    reports: list[OrientedCubeReport] = []
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
                rotation = candidate_rotation
                rotation_source = candidate_source
        elif auto_orient:
            auto_candidate = auto_orient_accumulator(
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
            if auto_candidate is not None:
                from_xyz, to_xyz, rotation, rotation_source = auto_candidate

        from_xyz, to_xyz = ensure_min_cube_size(
            from_xyz,
            to_xyz,
            allow_zero_axes=zero_thickness_axes(from_xyz, to_xyz),
        )
        cube_name = f"{part_key.name}_cube"
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
                )
            )
    return cuboids, reports


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
) -> tuple[list[float], list[float], list[float], str] | None:
    bone = bones.get(part_key.owner_bone)
    if bone is None:
        return None
    if accumulator.min_xyz is None or accumulator.max_xyz is None:
        return None
    if is_face_feature_name(part_key.name) or is_face_feature_name(bone.name):
        return None
    if accumulator.is_complex_split and (
        is_head_core_part(part_key.name) or is_face_feature_part(part_key.name) or is_side_feature_part(part_key.name)
    ):
        return None
    if accumulator.is_complex_split and accumulator.faces < AUTO_ORIENT_MIN_FACES:
        return None

    original_dimensions = bbox_dimensions(original_from_xyz, original_to_xyz)
    original_volume = box_volume_from_dimensions(original_dimensions)
    if max(original_dimensions) < AUTO_ORIENT_MIN_LONG_DIM or original_volume < AUTO_ORIENT_MIN_VOLUME:
        return None

    best_candidate: tuple[list[float], list[float], list[float], str, float] | None = None
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
        if direction_candidate is not None:
            return direction_candidate[:4]

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
            if best_candidate is None or geometry_candidate[4] < best_candidate[4]:
                best_candidate = geometry_candidate

    return None if best_candidate is None else best_candidate[:4]


def auto_orient_candidate(
    accumulator: BBoxAccumulator,
    scale: float,
    offset: list[float],
    origin: list[float],
    rotation_matrix: list[list[float]],
    original_volume: float,
    source: str,
    min_volume_reduction: float = AUTO_ORIENT_MIN_VOLUME_REDUCTION,
) -> tuple[list[float], list[float], list[float], str, float] | None:
    rotation = matrix_to_euler_xyz_degrees(rotation_matrix)
    if not has_nonzero_rotation(rotation):
        return None
    from_xyz, to_xyz = oriented_accumulator_bounds(accumulator, scale, offset, origin, rotation_matrix)
    score_from_xyz, score_to_xyz = ensure_min_cube_size(from_xyz, to_xyz)
    oriented_volume = box_volume(score_from_xyz, score_to_xyz)
    if oriented_volume >= original_volume * (1.0 - min_volume_reduction):
        return None
    return from_xyz, to_xyz, rotation, source, oriented_volume


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


def principal_axis_3d(points: list[list[float]]) -> list[float] | None:
    if len(points) < 3:
        return None
    centroid = points_centroid(points)
    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for point in points:
        delta = [point[index] - centroid[index] for index in range(3)]
        for row in range(3):
            for column in range(3):
                covariance[row][column] += delta[row] * delta[column]
    if sum(covariance[index][index] for index in range(3)) <= EPSILON:
        return None

    dominant_axis = max(range(3), key=lambda index: covariance[index][index])
    axis = [0.0, 0.0, 0.0]
    axis[dominant_axis] = 1.0
    for _ in range(16):
        next_axis = [sum(covariance[row][column] * axis[column] for column in range(3)) for row in range(3)]
        normalized = normalized_vec3(next_axis)
        if normalized is None:
            return None
        axis = normalized
    return axis


def planar_principal_rotation_matrices(points: list[list[float]]) -> list[list[list[float]]]:
    matrices: list[list[list[float]]] = []
    xy_angle = principal_angle_2d(points, 0, 1)
    if xy_angle is not None:
        matrices.append(rotation_matrix_z(xy_angle))
    yz_angle = principal_angle_2d(points, 1, 2)
    if yz_angle is not None:
        matrices.append(rotation_matrix_x(yz_angle))
    xz_angle = principal_angle_2d(points, 0, 2)
    if xz_angle is not None:
        matrices.append(rotation_matrix_y(-xz_angle))
    return matrices


def principal_angle_2d(points: list[list[float]], first_axis: int, second_axis: int) -> float | None:
    if len(points) < 3:
        return None
    center_first = sum(point[first_axis] for point in points) / len(points)
    center_second = sum(point[second_axis] for point in points) / len(points)
    variance_first = 0.0
    variance_second = 0.0
    covariance = 0.0
    for point in points:
        first = point[first_axis] - center_first
        second = point[second_axis] - center_second
        variance_first += first * first
        variance_second += second * second
        covariance += first * second
    if variance_first + variance_second <= EPSILON:
        return None
    return 0.5 * math.atan2(2.0 * covariance, variance_first - variance_second)


def rotation_matrix_x(angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]


def rotation_matrix_y(angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]


def rotation_matrix_z(angle: float) -> list[list[float]]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]


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


def rotation_matrix_from_y_axis(direction: list[float]) -> list[list[float]] | None:
    y_axis = normalized_vec3(direction)
    if y_axis is None:
        return None
    reference = [0.0, 0.0, 1.0]
    if abs(dot_vec3(y_axis, reference)) > 0.95:
        reference = [1.0, 0.0, 0.0]
    x_axis = normalized_vec3(cross_vec3(y_axis, reference))
    if x_axis is None:
        return None
    z_axis = normalized_vec3(cross_vec3(x_axis, y_axis))
    if z_axis is None:
        return None
    columns = [x_axis, y_axis, z_axis]
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def normalized_vec3(values: list[float]) -> list[float] | None:
    length = math.sqrt(sum(value * value for value in values))
    if length <= EPSILON:
        return None
    return [value / length for value in values]


def cross_vec3(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def dot_vec3(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def oriented_accumulator_bounds(
    accumulator: BBoxAccumulator,
    scale: float,
    offset: list[float],
    origin: list[float],
    rotation_matrix: list[list[float]],
) -> tuple[list[float], list[float]]:
    points = list(accumulator.points_by_vertex.values()) or bbox_corners(accumulator.min_xyz, accumulator.max_xyz)
    local_points = [
        inverse_rotate_around_origin(to_blockbench_space(point, scale, offset), origin, rotation_matrix)
        for point in points
    ]
    return points_bbox(local_points)


def bbox_corners(min_xyz: list[float] | None, max_xyz: list[float] | None) -> list[list[float]]:
    if min_xyz is None or max_xyz is None:
        return []
    return [
        [x, y, z]
        for x in (min_xyz[0], max_xyz[0])
        for y in (min_xyz[1], max_xyz[1])
        for z in (min_xyz[2], max_xyz[2])
    ]


def points_bbox(points: list[list[float]]) -> tuple[list[float], list[float]]:
    if not points:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    min_xyz = points[0].copy()
    max_xyz = points[0].copy()
    for point in points[1:]:
        for index in range(3):
            min_xyz[index] = min(min_xyz[index], point[index])
            max_xyz[index] = max(max_xyz[index], point[index])
    return min_xyz, max_xyz


def inverse_rotate_around_origin(
    point: list[float], origin: list[float], rotation_matrix: list[list[float]]
) -> list[float]:
    delta = [point[index] - origin[index] for index in range(3)]
    return [
        origin[index] + sum(rotation_matrix[row][index] * delta[row] for row in range(3))
        for index in range(3)
    ]


def normalized_rotation_matrix(matrix: list[list[float]]) -> list[list[float]]:
    columns = [[matrix[row][column] for row in range(3)] for column in range(3)]
    normalized_columns: list[list[float]] = []
    for column in columns:
        length = math.sqrt(sum(value * value for value in column))
        if length <= EPSILON:
            normalized_columns.append([0.0, 0.0, 0.0])
        else:
            normalized_columns.append([value / length for value in column])
    return [[normalized_columns[column][row] for column in range(3)] for row in range(3)]


def matrix_to_euler_xyz_degrees(matrix: list[list[float]]) -> list[float]:
    sy = math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])
    if sy > EPSILON:
        x = math.atan2(matrix[2][1], matrix[2][2])
        y = math.atan2(-matrix[2][0], sy)
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        y = math.atan2(-matrix[2][0], sy)
        z = 0.0
    return [math.degrees(value) for value in (x, y, z)]


def has_nonzero_rotation(rotation: list[float]) -> bool:
    return any(abs(value) > 1e-5 for value in rotation)


def count_small_cubes(cuboids: list[Cuboid]) -> int:
    small = 0
    for cuboid in cuboids:
        if any((cuboid.to_xyz[index] - cuboid.from_xyz[index]) <= MIN_CUBE_SIZE + EPSILON for index in range(3)):
            small += 1
    return small


def to_blockbench_space(point: list[float], scale: float, offset: list[float]) -> list[float]:
    return [(point[index] - offset[index]) * scale for index in range(3)]


def ensure_min_cube_size(
    from_xyz: list[float],
    to_xyz: list[float],
    allow_zero_axes: set[int] | None = None,
) -> tuple[list[float], list[float]]:
    result_from = from_xyz.copy()
    result_to = to_xyz.copy()
    allow_zero_axes = allow_zero_axes or set()
    for index in range(3):
        if result_from[index] > result_to[index]:
            result_from[index], result_to[index] = result_to[index], result_from[index]
        size = result_to[index] - result_from[index]
        if index in allow_zero_axes:
            center = (result_from[index] + result_to[index]) * 0.5
            result_from[index] = center
            result_to[index] = center
            continue
        if size < MIN_CUBE_SIZE:
            center = (result_from[index] + result_to[index]) * 0.5
            half = MIN_CUBE_SIZE * 0.5
            result_from[index] = center - half
            result_to[index] = center + half
    return result_from, result_to


def zero_thickness_axes(from_xyz: list[float], to_xyz: list[float]) -> set[int]:
    dimensions = bbox_dimensions(from_xyz, to_xyz)
    ranked_dimensions = sorted(dimensions, reverse=True)
    if ranked_dimensions[0] <= EPSILON:
        return set()
    if ranked_dimensions[1] < ZERO_THICKNESS_MIN_PLANE_DIMENSION:
        return set()

    candidates = {
        index
        for index, dimension in enumerate(dimensions)
        if dimension <= EPSILON
        or (
            dimension < MIN_CUBE_SIZE
            and dimension / ranked_dimensions[0] <= ZERO_THICKNESS_DIMENSION_RATIO
        )
    }
    return candidates if len(candidates) == 1 else set()


def build_bbmodel(
    name: str,
    bones: dict[int, BonePartition],
    world_matrices: dict[int, list[list[float]]],
    cuboids: list[Cuboid],
    scale: float,
    offset: list[float],
) -> dict[str, Any]:
    cubes_by_bone: dict[int, list[int]] = {}
    for cube_index, cuboid in enumerate(cuboids):
        cubes_by_bone.setdefault(cuboid.owner_bone, []).append(cube_index)
    cube_uuids = {
        cube_index: stable_uuid("cube", f"{cuboid.owner_bone}:{cuboid.name}:{cube_index}")
        for cube_index, cuboid in enumerate(cuboids)
    }
    group_uuids = {bone_index: stable_uuid("group", f"{bone_index}:{bone.name}") for bone_index, bone in bones.items()}

    elements = [cube_to_element(cuboid, cube_uuids[cube_index]) for cube_index, cuboid in enumerate(cuboids)]
    groups = [
        group_to_dict(
            bone,
            group_uuids[bone_index],
            to_blockbench_space(matrix_translation(world_matrices.get(bone_index, identity_matrix())), scale, offset),
        )
        for bone_index, bone in sorted(bones.items())
    ]
    roots = [bone for bone in sorted(bones.values(), key=lambda item: item.node_index) if bone.parent is None]
    outliner = [build_outliner_entry(root, bones, group_uuids, cube_uuids, cubes_by_bone) for root in roots]

    return {
        "meta": {"format_version": "5.0", "model_format": "free", "box_uv": False},
        "name": name,
        "model_identifier": "",
        "visible_box": [1, 1, 0],
        "variable_placeholders": "",
        "variable_placeholder_buttons": [],
        "timeline_setups": [],
        "unhandled_root_fields": {},
        "resolution": {"width": 16, "height": 16},
        "elements": elements,
        "groups": groups,
        "outliner": outliner,
        "textures": [],
        "animations": [],
    }


def cube_to_element(cuboid: Cuboid, cube_uuid: str) -> dict[str, Any]:
    element = {
        "name": cuboid.name,
        "box_uv": False,
        "render_order": "default",
        "locked": False,
        "allow_mirror_modeling": True,
        "from": rounded_vec(cuboid.from_xyz),
        "to": rounded_vec(cuboid.to_xyz),
        "autouv": 0,
        "color": 1,
        "origin": rounded_vec(cuboid.origin),
        "faces": default_cube_faces(),
        "type": "cube",
        "uuid": cube_uuid,
    }
    if has_nonzero_rotation(cuboid.rotation):
        element["rotation"] = rounded_vec(cuboid.rotation)
    return element


def group_to_dict(bone: BonePartition, group_uuid: str, origin: list[float]) -> dict[str, Any]:
    return {
        "uuid": group_uuid,
        "export": True,
        "locked": False,
        "origin": rounded_vec(origin),
        "rotation": [0, 0, 0],
        "color": 0,
        "name": bone.name,
        "children": [],
        "reset": False,
        "shade": True,
        "mirror_uv": False,
        "selected": False,
        "visibility": True,
        "autouv": 0,
        "isOpen": True,
        "primary_selected": False,
    }


def build_outliner_entry(
    bone: BonePartition,
    bones: dict[int, BonePartition],
    group_uuids: dict[int, str],
    cube_uuids: dict[int, str],
    cubes_by_bone: dict[int, list[int]],
) -> dict[str, Any]:
    children: list[str | dict[str, Any]] = []
    for child_index in sorted(bone.children):
        child = bones.get(child_index)
        if child is not None:
            children.append(build_outliner_entry(child, bones, group_uuids, cube_uuids, cubes_by_bone))
    for cube_index in cubes_by_bone.get(bone.node_index, []):
        children.append(cube_uuids[cube_index])

    return {"uuid": group_uuids[bone.node_index], "isOpen": True, "children": children}


def default_cube_faces() -> dict[str, dict[str, list[float]]]:
    return {
        "north": {"uv": [0, 0, 16, 16]},
        "east": {"uv": [0, 0, 16, 16]},
        "south": {"uv": [0, 0, 16, 16]},
        "west": {"uv": [0, 0, 16, 16]},
        "up": {"uv": [0, 0, 16, 16]},
        "down": {"uv": [0, 0, 16, 16]},
    }


def stable_uuid(kind: str, value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"gltf2bb:{kind}:{value}"))


def rounded_vec(values: list[float]) -> list[float]:
    result = []
    for value in values:
        if not math.isfinite(value):
            raise ConvertError("generated non-finite Blockbench coordinate")
        rounded = round(value, 6)
        result.append(0.0 if rounded == -0.0 else rounded)
    return result


def compute_world_matrices(nodes: list[dict[str, Any]], parent_map: dict[int, int]) -> dict[int, list[list[float]]]:
    cache: dict[int, list[list[float]]] = {}
    visiting: set[int] = set()

    def compute(node_index: int) -> list[list[float]]:
        if node_index in cache:
            return cache[node_index]
        if node_index in visiting:
            return identity_matrix()
        visiting.add(node_index)
        local = node_local_matrix(nodes[node_index])
        parent = parent_map.get(node_index)
        if parent is not None and is_valid_index(nodes, parent):
            world = multiply_matrix(compute(parent), local)
        else:
            world = local
        visiting.remove(node_index)
        cache[node_index] = world
        return world

    for node_index in range(len(nodes)):
        compute(node_index)
    return cache


def node_local_matrix(node: dict[str, Any]) -> list[list[float]]:
    matrix = node.get("matrix")
    if matrix is not None:
        if len(matrix) != 16:
            return identity_matrix()
        return [
            [float(matrix[0]), float(matrix[4]), float(matrix[8]), float(matrix[12])],
            [float(matrix[1]), float(matrix[5]), float(matrix[9]), float(matrix[13])],
            [float(matrix[2]), float(matrix[6]), float(matrix[10]), float(matrix[14])],
            [float(matrix[3]), float(matrix[7]), float(matrix[11]), float(matrix[15])],
        ]

    translation = [float(value) for value in node.get("translation", [0.0, 0.0, 0.0])]
    rotation = [float(value) for value in node.get("rotation", [0.0, 0.0, 0.0, 1.0])]
    scale = [float(value) for value in node.get("scale", [1.0, 1.0, 1.0])]
    return multiply_matrix(multiply_matrix(translation_matrix(translation), quaternion_matrix(rotation)), scale_matrix(scale))


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_matrix(translation: list[float]) -> list[list[float]]:
    matrix = identity_matrix()
    matrix[0][3] = translation[0] if len(translation) > 0 else 0.0
    matrix[1][3] = translation[1] if len(translation) > 1 else 0.0
    matrix[2][3] = translation[2] if len(translation) > 2 else 0.0
    return matrix


def scale_matrix(scale: list[float]) -> list[list[float]]:
    matrix = identity_matrix()
    matrix[0][0] = scale[0] if len(scale) > 0 else 1.0
    matrix[1][1] = scale[1] if len(scale) > 1 else 1.0
    matrix[2][2] = scale[2] if len(scale) > 2 else 1.0
    return matrix


def quaternion_matrix(rotation: list[float]) -> list[list[float]]:
    x = rotation[0] if len(rotation) > 0 else 0.0
    y = rotation[1] if len(rotation) > 1 else 0.0
    z = rotation[2] if len(rotation) > 2 else 0.0
    w = rotation[3] if len(rotation) > 3 else 1.0
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= EPSILON:
        return identity_matrix()
    x /= length
    y /= length
    z /= length
    w /= length

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_matrix(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)]
        for row in range(4)
    ]


def transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def matrix_translation(matrix: list[list[float]]) -> list[float]:
    return [matrix[0][3], matrix[1][3], matrix[2][3]]


def convert_result_to_dict(result: ConvertResult) -> dict[str, Any]:
    return {
        "file": str(result.input_path),
        "output": str(result.output_path),
        "mode": result.mode,
        "preset": result.preset,
        "scale": result.scale,
        "totals": {
            "cubes": len(result.cubes),
            "original_bones": result.bone_resolution.original_bones,
            "kept_bones": result.bone_resolution.kept_bones,
            "merged_bones": len(result.bone_resolution.merged_to_parent),
            "ignored_bones": len(result.bone_resolution.ignored),
            "empty_bones": result.empty_bones,
            "small_cubes": result.small_cubes,
            "complex_split_bones": len(result.complex_split),
            "hybrid_special_cube_bones": len(result.hybrid.special_cube_bones),
            "deleted_small_parts": len(result.cleanup.deleted_parts),
            "merged_small_parts": len(result.cleanup.merged_parts),
            "kept_small_parts": len(result.cleanup.kept_small_parts),
            "oriented_cubes": len(result.oriented_cubes),
        },
        "bone_resolution": bone_resolution_to_dict(result.bone_resolution),
        "hybrid": hybrid_to_dict(result.hybrid),
        "hybrid_detail_split": hybrid_detail_split_to_dict(result.hybrid_detail_split),
        "complex_split": [complex_split_to_dict(item) for item in result.complex_split],
        "cleanup": cleanup_to_dict(result.cleanup),
        "oriented_cubes": [oriented_cube_to_dict(item) for item in result.oriented_cubes],
        "quality": quality_to_dict(result),
        "cubes": [cuboid_to_dict(cuboid) for cuboid in result.cubes],
        "warnings": result.warnings,
}


def hybrid_to_dict(item: HybridModeReport) -> dict[str, Any]:
    return {
        "enabled": item.enabled,
        "special_cube_bones": list(item.special_cube_bones),
        "mesh_strategy": item.mesh_strategy,
        "cuboid_strategy": item.cuboid_strategy,
    }


def hybrid_detail_split_to_dict(item: HybridDetailSplitConfig) -> dict[str, Any]:
    return {
        "enabled": item.enabled,
        "min_faces": item.min_faces,
        "max_long_dim_ratio": item.max_long_dim_ratio,
        "by_material": item.by_material,
        "by_connected_component": item.by_connected_component,
        "min_material_faces": item.min_material_faces,
        "min_material_ratio": item.min_material_ratio,
        "min_component_faces": item.min_component_faces,
        "min_component_ratio": item.min_component_ratio,
    }


def complex_split_to_dict(item: ComplexSplitBoneReport) -> dict[str, Any]:
    return {
        "bone": item.bone,
        "bone_name": item.bone_name,
        "source_faces": item.source_faces,
        "subparts": [complex_split_subpart_to_dict(subpart) for subpart in item.subparts],
        "merged_tiny_components": item.merged_tiny_components,
        "deleted_tiny_components": item.deleted_tiny_components,
        "merged_tiny_hair_buckets": item.merged_tiny_hair_buckets,
        "expanded_hair_bucket_overlap": item.expanded_hair_bucket_overlap,
    }


def complex_split_subpart_to_dict(item: ComplexSplitSubpartReport) -> dict[str, Any]:
    return {
        "name": item.name,
        "method": item.method,
        "faces": item.faces,
        "vertices": item.vertices,
    }


def cleanup_to_dict(item: CleanupReport) -> dict[str, Any]:
    return {
        "deleted_parts": [cleanup_part_to_dict(part) for part in item.deleted_parts],
        "merged_parts": [cleanup_part_to_dict(part) for part in item.merged_parts],
        "kept_small_parts": [cleanup_part_to_dict(part) for part in item.kept_small_parts],
    }


def cleanup_part_to_dict(item: CleanupPartReport) -> dict[str, Any]:
    return {
        "owner_bone": item.owner_bone,
        "owner_bone_name": item.owner_bone_name,
        "name": item.name,
        "action": item.action,
        "reason": item.reason,
        "faces": item.faces,
        "vertices": item.vertices,
        "bbox_volume": item.bbox_volume,
        "target_bone": item.target_bone,
        "target_bone_name": item.target_bone_name,
    }


def oriented_cube_to_dict(item: OrientedCubeReport) -> dict[str, Any]:
    return {
        "name": item.name,
        "owner_bone": item.owner_bone,
        "owner_bone_name": item.owner_bone_name,
        "rotation": rounded_vec(item.rotation),
        "source": item.source,
    }


def quality_to_dict(result: ConvertResult) -> dict[str, Any]:
    return {
        "largest_cubes": [quality_cube_to_dict(cuboid) for cuboid in largest_quality_cubes(result.cubes)],
        "unrotated_elongated_cubes": [
            quality_unrotated_elongated_cube_to_dict(cuboid, reason)
            for cuboid, reason in unrotated_elongated_quality_cubes(result.cubes)
        ],
        "tiny_fragment_cubes": [
            quality_tiny_fragment_cube_to_dict(cuboid)
            for cuboid in tiny_fragment_quality_cubes(result.cubes)
        ],
        "skipped_unskinned_meshes_summary": skipped_unskinned_meshes_summary_to_dict(
            result.skipped_unskinned_meshes
        ),
    }


def largest_quality_cubes(cuboids: list[Cuboid]) -> list[Cuboid]:
    return sorted(
        cuboids,
        key=lambda cuboid: (cube_volume(cuboid), max(cube_dimensions(cuboid), default=0.0), cuboid.faces),
        reverse=True,
    )[:QUALITY_LARGEST_CUBES_LIMIT]


def unrotated_elongated_quality_cubes(cuboids: list[Cuboid]) -> list[tuple[Cuboid, str]]:
    flagged: list[tuple[Cuboid, str]] = []
    for cuboid in cuboids:
        reason = unrotated_elongated_reason(cuboid)
        if reason is not None:
            flagged.append((cuboid, reason))
    return sorted(flagged, key=lambda item: cube_elongation_ratio(item[0]), reverse=True)


def tiny_fragment_quality_cubes(cuboids: list[Cuboid]) -> list[Cuboid]:
    model_height = model_height_from_cubes(cuboids)
    max_dimension = max(MIN_CUBE_SIZE * 2.0, model_height * 0.03)
    tiny = [
        cuboid
        for cuboid in cuboids
        if cuboid.faces <= QUALITY_TINY_FRAGMENT_MAX_FACES
        and max(cube_dimensions(cuboid), default=0.0) <= max_dimension
    ]
    return sorted(tiny, key=lambda cuboid: (cube_volume(cuboid), cuboid.faces, cuboid.name))[
        :QUALITY_TINY_FRAGMENT_CUBES_LIMIT
    ]


def quality_cube_to_dict(cuboid: Cuboid) -> dict[str, Any]:
    dimensions = cube_dimensions(cuboid)
    return {
        "name": cuboid.name,
        "owner_bone": cuboid.owner_bone,
        "owner_bone_name": cuboid.owner_bone_name,
        "dimensions": rounded_vec(dimensions),
        "volume": rounded_number(box_volume_from_dimensions(dimensions)),
        "faces": cuboid.faces,
        "rotation_source": cuboid.rotation_source,
    }


def quality_unrotated_elongated_cube_to_dict(cuboid: Cuboid, reason: str) -> dict[str, Any]:
    data = quality_cube_to_dict(cuboid)
    data["reason"] = reason
    return data


def quality_tiny_fragment_cube_to_dict(cuboid: Cuboid) -> dict[str, Any]:
    dimensions = cube_dimensions(cuboid)
    return {
        "name": cuboid.name,
        "owner_bone": cuboid.owner_bone,
        "owner_bone_name": cuboid.owner_bone_name,
        "faces": cuboid.faces,
        "dimensions": rounded_vec(dimensions),
        "volume": rounded_number(box_volume_from_dimensions(dimensions)),
    }


def skipped_unskinned_meshes_summary_to_dict(items: list[SkippedUnskinnedMesh]) -> dict[str, Any]:
    return {
        "count": len(items),
        "node_indices": sorted({item.node_index for item in items}),
        "mesh_indices": sorted({item.mesh_index for item in items}),
    }


def unrotated_elongated_reason(cuboid: Cuboid) -> str | None:
    if has_nonzero_rotation(cuboid.rotation):
        return None
    ratio = cube_elongation_ratio(cuboid)
    if ratio < QUALITY_ELONGATED_DIMENSION_RATIO:
        return None
    return f"long_dim/second_dim={ratio:.2f}>={QUALITY_ELONGATED_DIMENSION_RATIO:.2f}; no rotation"


def cube_elongation_ratio(cuboid: Cuboid) -> float:
    dimensions = sorted(cube_dimensions(cuboid), reverse=True)
    if not dimensions or dimensions[0] <= EPSILON:
        return 0.0
    if len(dimensions) < 2 or dimensions[1] <= EPSILON:
        return math.inf
    return dimensions[0] / dimensions[1]


def cube_dimensions(cuboid: Cuboid) -> list[float]:
    return bbox_dimensions(cuboid.from_xyz, cuboid.to_xyz)


def cube_volume(cuboid: Cuboid) -> float:
    return box_volume_from_dimensions(cube_dimensions(cuboid))


def model_height_from_cubes(cuboids: list[Cuboid]) -> float:
    if not cuboids:
        return 0.0
    min_y = min(min(cuboid.from_xyz[1], cuboid.to_xyz[1]) for cuboid in cuboids)
    max_y = max(max(cuboid.from_xyz[1], cuboid.to_xyz[1]) for cuboid in cuboids)
    return max(max_y - min_y, 0.0)


def rounded_number(value: float) -> float:
    if not math.isfinite(value):
        raise ConvertError("generated non-finite quality metric")
    rounded = round(value, 6)
    return 0.0 if rounded == -0.0 else rounded


def cuboid_to_dict(cuboid: Cuboid) -> dict[str, Any]:
    data = {
        "owner_bone": cuboid.owner_bone,
        "name": cuboid.name,
        "from": rounded_vec(cuboid.from_xyz),
        "to": rounded_vec(cuboid.to_xyz),
        "origin": rounded_vec(cuboid.origin),
        "faces": cuboid.faces,
        "vertices": cuboid.vertices,
    }
    if has_nonzero_rotation(cuboid.rotation):
        data["rotation"] = rounded_vec(cuboid.rotation)
        data["rotation_source"] = cuboid.rotation_source
    return data


def write_convert_report(result: ConvertResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(convert_result_to_dict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def format_convert_summary(result: ConvertResult) -> str:
    lines = [
        f"File: {result.input_path}",
        f"Mode: {result.mode}",
        f"Preset: {result.preset}",
        f"Scale: {result.scale:.6g}",
        f"Cubes: {len(result.cubes)}",
        f"Bones: {result.bone_resolution.kept_bones} kept / {result.bone_resolution.original_bones} original",
        f"Merged bones: {len(result.bone_resolution.merged_to_parent)}",
        f"Ignored bones: {len(result.bone_resolution.ignored)}",
        f"Empty bones: {result.empty_bones}",
        f"Small cubes: {result.small_cubes}",
        f"Complex splits: {len(result.complex_split)}",
        f"Hybrid strategy: {result.hybrid.mesh_strategy if result.hybrid.enabled else 'off'}",
        f"Cleanup deleted parts: {len(result.cleanup.deleted_parts)}",
        f"Cleanup merged parts: {len(result.cleanup.merged_parts)}",
        f"Oriented cubes: {len(result.oriented_cubes)}",
        f"Output: {result.output_path}",
    ]
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)
