from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ProcessingConfig, resolve_processing_config
from .constants import MODE_TRIANGLES, MODE_TRIANGLE_FAN, MODE_TRIANGLE_STRIP
from .errors import InspectError, PartitionError
from .inspect import SUPPORTED_MODEL_SUFFIXES, build_parent_map, is_valid_index, load_gltf, read_accessor


EPSILON = 1e-6


@dataclass
class BonePartition:
    node_index: int
    name: str
    parent: int | None
    children: list[int]
    skin_indices: set[int] = field(default_factory=set)
    faces: int = 0
    vertices: set[tuple[str, int]] = field(default_factory=set)
    primitives: set[str] = field(default_factory=set)
    fallback_faces: int = 0


@dataclass
class PrimitivePartition:
    node_index: int
    mesh_index: int
    primitive_index: int
    skin_index: int
    material_id: int | None
    faces: int
    assigned_faces: int = 0
    fallback_faces: int = 0
    unassigned_faces: int = 0
    unweighted_vertices: int = 0
    invalid_joint_references: int = 0
    invalid_face_indices: int = 0
    owners: dict[int, int] = field(default_factory=dict)
    owner_vertices: dict[int, set[int]] = field(default_factory=dict)


@dataclass
class ResolvedBoneInfo:
    node_index: int
    name: str
    action: str
    reason: str
    resolved_to: int | None
    resolved_to_name: str | None


@dataclass
class BoneResolutionReport:
    preset: str
    original_bones: int
    kept_bones: int
    resolved_bones: dict[int, int]
    merged_to_parent: list[ResolvedBoneInfo] = field(default_factory=list)
    ignored: list[ResolvedBoneInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PartitionReport:
    path: Path
    scenes: int
    nodes: int
    meshes: int
    skins: int
    animations: int
    materials: int
    bones: dict[int, BonePartition]
    bone_resolution: BoneResolutionReport
    primitives: list[PrimitivePartition]
    warnings: list[str] = field(default_factory=list)
    skipped_primitives: int = 0


def partition_model(
    path: Path,
    *,
    preset: str | None = None,
    config_path: Path | None = None,
    processing_config: ProcessingConfig | None = None,
) -> PartitionReport:
    if not path.exists():
        raise PartitionError(f"input file does not exist: {path}")
    if not path.is_file():
        raise PartitionError(f"input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise PartitionError(f"expected a .gltf, .glb, or .vrm file, got: {path.name}")

    try:
        gltf, binary_chunk = load_gltf(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error, InspectError) as exc:
        raise PartitionError(f"failed to read glTF data from {path}: {exc}") from exc

    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    skins = gltf.get("skins", [])
    accessors = gltf.get("accessors", [])
    parent_map = build_parent_map(nodes)
    config = processing_config or resolve_processing_config(preset, config_path)
    bones, bone_resolution = build_filtered_bone_partitions(nodes, skins, parent_map, config)
    report = PartitionReport(
        path=path,
        scenes=len(gltf.get("scenes", [])),
        nodes=len(nodes),
        meshes=len(meshes),
        skins=len(skins),
        animations=len(gltf.get("animations", [])),
        materials=len(gltf.get("materials", [])),
        bones=bones,
        bone_resolution=bone_resolution,
        primitives=[],
    )
    report.warnings.extend(bone_resolution.warnings)
    buffer_cache: dict[int, bytes] = {}

    for node_index, node in enumerate(nodes):
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if mesh_index is None:
            continue
        if not is_valid_index(meshes, mesh_index):
            report.warnings.append(f"Node {node_index} references missing mesh {mesh_index}.")
            continue
        if skin_index is None:
            report.skipped_primitives += len(meshes[mesh_index].get("primitives", []))
            report.warnings.append(f"Node {node_index} mesh {mesh_index} has no skin; skipped partition.")
            continue
        if not is_valid_index(skins, skin_index):
            report.skipped_primitives += len(meshes[mesh_index].get("primitives", []))
            report.warnings.append(f"Node {node_index} references missing skin {skin_index}; skipped partition.")
            continue

        skin = skins[skin_index]
        skin_joints = skin.get("joints", [])
        fallback_bone = resolve_bone_node(
            fallback_joint_for_skin(skin, skin_joints, parent_map), bone_resolution.resolved_bones
        )
        for primitive_index, primitive in enumerate(meshes[mesh_index].get("primitives", [])):
            primitive_report = partition_primitive(
                gltf,
                path,
                binary_chunk,
                buffer_cache,
                accessors,
                primitive,
                node_index,
                mesh_index,
                primitive_index,
                skin_index,
                skin_joints,
                fallback_bone,
                bone_resolution.resolved_bones,
                report,
            )
            if primitive_report is not None:
                report.primitives.append(primitive_report)

    return report


def build_filtered_bone_partitions(
    nodes: list[dict[str, Any]],
    skins: list[dict[str, Any]],
    parent_map: dict[int, int],
    config: ProcessingConfig,
) -> tuple[dict[int, BonePartition], BoneResolutionReport]:
    original_bones = build_bone_partitions(nodes, skins, parent_map)
    return apply_bone_filter(original_bones, config)


def build_bone_partitions(
    nodes: list[dict[str, Any]], skins: list[dict[str, Any]], parent_map: dict[int, int]
) -> dict[int, BonePartition]:
    bones: dict[int, BonePartition] = {}
    child_map: dict[int, list[int]] = {}
    for child, parent in parent_map.items():
        child_map.setdefault(parent, []).append(child)

    for skin_index, skin in enumerate(skins):
        joint_set = set(skin.get("joints", []))
        for node_index in skin.get("joints", []):
            if not is_valid_index(nodes, node_index):
                continue
            node = nodes[node_index]
            parent = parent_map.get(node_index)
            bone = bones.setdefault(
                node_index,
                BonePartition(
                    node_index=node_index,
                    name=node.get("name") or f"bone_{node_index}",
                    parent=parent if parent in joint_set else None,
                    children=[child for child in child_map.get(node_index, []) if child in joint_set],
                ),
            )
            bone.skin_indices.add(skin_index)
    return bones


def apply_bone_filter(
    original_bones: dict[int, BonePartition], config: ProcessingConfig
) -> tuple[dict[int, BonePartition], BoneResolutionReport]:
    actions: dict[int, tuple[str, str]] = {}
    for bone_index, bone in original_bones.items():
        action = bone_filter_action(bone.name, config)
        if action is not None:
            actions[bone_index] = action

    kept_indices = set(original_bones) - set(actions)
    warnings: list[str] = []
    if not kept_indices and original_bones:
        root = first_root_bone(original_bones)
        kept_indices.add(root)
        actions.pop(root, None)
        warnings.append(
            f"All bones matched preset {config.preset!r} filtering rules; kept root bone {root} "
            "to preserve a valid hierarchy."
        )

    resolved_bones = {bone_index: bone_index for bone_index in kept_indices}
    for bone_index in sorted(actions):
        target = nearest_kept_ancestor(original_bones[bone_index].parent, original_bones, kept_indices)
        if target is None:
            target = first_root_bone(original_bones, kept_indices) if kept_indices else None
        if target is not None:
            resolved_bones[bone_index] = target

    filtered = copy_kept_bones(original_bones, kept_indices)
    resolution = BoneResolutionReport(
        preset=config.preset,
        original_bones=len(original_bones),
        kept_bones=len(filtered),
        resolved_bones=resolved_bones,
        warnings=warnings,
    )

    if config.bone_filter.report_merged_bones:
        for bone_index, (action, reason) in sorted(actions.items()):
            bone = original_bones[bone_index]
            target = resolved_bones.get(bone_index)
            target_bone = filtered.get(target) if target is not None else None
            info = ResolvedBoneInfo(
                node_index=bone_index,
                name=bone.name,
                action=action,
                reason=reason,
                resolved_to=target,
                resolved_to_name=target_bone.name if target_bone is not None else None,
            )
            if action == "merged_to_parent":
                resolution.merged_to_parent.append(info)
            else:
                resolution.ignored.append(info)

    return filtered, resolution


def bone_filter_action(name: str, config: ProcessingConfig) -> tuple[str, str] | None:
    reason = match_name_contains(
        name,
        config.bone_filter.merge_to_parent_name_contains,
        config.bone_filter.case_sensitive,
    )
    if reason is not None:
        return "merged_to_parent", reason

    reason = match_name_regex(
        name,
        config.bone_filter.merge_to_parent_name_regex,
        config.bone_filter.case_sensitive,
    )
    if reason is not None:
        return "merged_to_parent", reason

    reason = match_name_contains(
        name,
        config.bone_filter.ignore_name_contains,
        config.bone_filter.case_sensitive,
    )
    if reason is not None:
        return "ignored", reason

    return None


def match_name_contains(name: str, patterns: tuple[str, ...], case_sensitive: bool) -> str | None:
    haystack = name if case_sensitive else name.casefold()
    for pattern in patterns:
        needle = pattern if case_sensitive else pattern.casefold()
        if needle in haystack:
            return f"name_contains:{pattern}"
    return None


def match_name_regex(name: str, patterns: tuple[str, ...], case_sensitive: bool) -> str | None:
    flags = 0 if case_sensitive else re.IGNORECASE
    for pattern in patterns:
        if re.search(pattern, name, flags):
            return f"name_regex:{pattern}"
    return None


def first_root_bone(
    bones: dict[int, BonePartition], candidates: set[int] | None = None
) -> int:
    allowed = candidates if candidates is not None else set(bones)
    roots = [bone.node_index for bone in bones.values() if bone.node_index in allowed and bone.parent not in allowed]
    if roots:
        return min(roots)
    return min(allowed)


def nearest_kept_ancestor(
    parent: int | None,
    bones: dict[int, BonePartition],
    kept_indices: set[int],
) -> int | None:
    current = parent
    seen: set[int] = set()
    while current is not None and current not in seen:
        seen.add(current)
        if current in kept_indices:
            return current
        current_bone = bones.get(current)
        current = current_bone.parent if current_bone is not None else None
    return None


def copy_kept_bones(
    original_bones: dict[int, BonePartition], kept_indices: set[int]
) -> dict[int, BonePartition]:
    filtered: dict[int, BonePartition] = {}
    for bone_index in sorted(kept_indices):
        bone = original_bones[bone_index]
        filtered[bone_index] = BonePartition(
            node_index=bone.node_index,
            name=bone.name,
            parent=nearest_kept_ancestor(bone.parent, original_bones, kept_indices),
            children=[],
            skin_indices=set(bone.skin_indices),
        )

    for bone in filtered.values():
        if bone.parent in filtered:
            filtered[bone.parent].children.append(bone.node_index)
    for bone in filtered.values():
        bone.children.sort()
    return filtered


def resolve_bone_node(node_index: int | None, resolved_bones: dict[int, int]) -> int | None:
    if node_index is None:
        return None
    return resolved_bones.get(node_index)


def fallback_joint_for_skin(
    skin: dict[str, Any], skin_joints: list[int], parent_map: dict[int, int]
) -> int | None:
    skeleton = skin.get("skeleton")
    if skeleton in skin_joints:
        return skeleton

    joint_set = set(skin_joints)
    for joint in skin_joints:
        if parent_map.get(joint) not in joint_set:
            return joint
    return skin_joints[0] if skin_joints else None


def partition_primitive(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessors: list[dict[str, Any]],
    primitive: dict[str, Any],
    node_index: int,
    mesh_index: int,
    primitive_index: int,
    skin_index: int,
    skin_joints: list[int],
    fallback_bone: int | None,
    resolved_bones: dict[int, int],
    report: PartitionReport,
) -> PrimitivePartition | None:
    attributes = primitive.get("attributes", {})
    position_accessor = attributes.get("POSITION")
    if not is_valid_index(accessors, position_accessor):
        report.skipped_primitives += 1
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} has no valid POSITION accessor; skipped partition."
        )
        return None

    vertex_count = int(accessors[position_accessor].get("count", 0))
    mode = int(primitive.get("mode", MODE_TRIANGLES))
    if mode not in {MODE_TRIANGLES, MODE_TRIANGLE_STRIP, MODE_TRIANGLE_FAN}:
        report.skipped_primitives += 1
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} uses non-triangle mode {mode}; skipped partition."
        )
        return None

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
    primitive_report = PrimitivePartition(
        node_index=node_index,
        mesh_index=mesh_index,
        primitive_index=primitive_index,
        skin_index=skin_index,
        material_id=primitive.get("material"),
        faces=len(faces),
    )
    if not faces:
        return primitive_report

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
    if joints is None or weights is None:
        primitive_report.unweighted_vertices = vertex_count
    else:
        primitive_report.unweighted_vertices = count_unweighted_vertices(weights, vertex_count)

    primitive_key = f"node={node_index}/mesh={mesh_index}/primitive={primitive_index}"
    for face in faces:
        owner, fallback_used, invalid_refs, invalid_face_indices = choose_face_owner(
            face, joints, weights, skin_joints, fallback_bone, vertex_count, resolved_bones
        )
        primitive_report.invalid_joint_references += invalid_refs
        primitive_report.invalid_face_indices += invalid_face_indices

        if owner is None:
            primitive_report.unassigned_faces += 1
            continue

        primitive_report.assigned_faces += 1
        primitive_report.owners[owner] = primitive_report.owners.get(owner, 0) + 1
        primitive_report.owner_vertices.setdefault(owner, set()).update(face)
        if fallback_used:
            primitive_report.fallback_faces += 1

        bone = report.bones.get(owner)
        if bone is None:
            report.warnings.append(
                f"Mesh {mesh_index} primitive {primitive_index} assigned face to joint node {owner}, "
                "but that node is not present in the skeleton report."
            )
            continue
        bone.faces += 1
        bone.vertices.update((primitive_key, vertex_index) for vertex_index in face)
        bone.primitives.add(primitive_key)
        if fallback_used:
            bone.fallback_faces += 1

    return primitive_report


def read_faces(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessors: list[dict[str, Any]],
    primitive: dict[str, Any],
    vertex_count: int,
    report: PartitionReport,
    mesh_index: int,
    primitive_index: int,
) -> list[tuple[int, int, int]]:
    indices_accessor = primitive.get("indices")
    if indices_accessor is None:
        index_stream = list(range(vertex_count))
    elif is_valid_index(accessors, indices_accessor):
        try:
            rows = read_accessor(gltf, path, binary_chunk, buffer_cache, indices_accessor)
        except InspectError as exc:
            report.warnings.append(
                f"Mesh {mesh_index} primitive {primitive_index} could not read indices accessor "
                f"{indices_accessor}: {exc}; falling back to non-indexed faces."
            )
            index_stream = list(range(vertex_count))
        else:
            index_stream = [int(row[0]) for row in rows or []]
    else:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} references missing indices accessor "
            f"{indices_accessor}; falling back to non-indexed faces."
        )
        index_stream = list(range(vertex_count))

    mode = int(primitive.get("mode", MODE_TRIANGLES))
    if mode == MODE_TRIANGLES:
        return [tuple(index_stream[index : index + 3]) for index in range(0, len(index_stream) - 2, 3)]
    if mode == MODE_TRIANGLE_STRIP:
        return [
            (index_stream[index], index_stream[index + 1], index_stream[index + 2])
            for index in range(0, max(len(index_stream) - 2, 0))
        ]
    if mode == MODE_TRIANGLE_FAN:
        return [
            (index_stream[0], index_stream[index + 1], index_stream[index + 2])
            for index in range(0, max(len(index_stream) - 2, 0))
        ]

    return []


def read_optional_accessor(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessor_index: int | None,
    semantic: str,
    report: PartitionReport,
    mesh_index: int,
    primitive_index: int,
) -> list[list[float | int]] | None:
    if accessor_index is None:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} has no {semantic}; using fallback owner."
        )
        return None
    try:
        return read_accessor(gltf, path, binary_chunk, buffer_cache, accessor_index)
    except InspectError as exc:
        report.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} could not read {semantic} accessor "
            f"{accessor_index}: {exc}; using fallback owner."
        )
        return None


def count_unweighted_vertices(weights: list[list[float | int]], vertex_count: int) -> int:
    unweighted = max(vertex_count - len(weights), 0)
    for row in weights[:vertex_count]:
        if sum(float(value) for value in row) <= EPSILON:
            unweighted += 1
    return unweighted


def choose_face_owner(
    face: tuple[int, int, int],
    joints: list[list[float | int]] | None,
    weights: list[list[float | int]] | None,
    skin_joints: list[int],
    fallback_bone: int | None,
    vertex_count: int,
    resolved_bones: dict[int, int] | None = None,
) -> tuple[int | None, bool, int, int]:
    scores: dict[int, float] = {}
    invalid_joint_references = 0
    invalid_face_indices = 0

    if joints is not None and weights is not None:
        for vertex_index in face:
            if vertex_index < 0 or vertex_index >= vertex_count:
                invalid_face_indices += 1
                continue
            if vertex_index >= len(joints) or vertex_index >= len(weights):
                continue

            joint_row = joints[vertex_index]
            weight_row = weights[vertex_index]
            weight_sum = sum(float(weight) for weight in weight_row)
            if weight_sum <= EPSILON:
                continue

            for joint, weight in zip(joint_row, weight_row, strict=False):
                normalized_weight = float(weight) / weight_sum
                if normalized_weight <= EPSILON:
                    continue
                joint_index = int(joint)
                if not 0 <= joint_index < len(skin_joints):
                    invalid_joint_references += 1
                    continue
                owner = skin_joints[joint_index]
                if resolved_bones is not None:
                    resolved_owner = resolved_bones.get(owner)
                    if resolved_owner is None:
                        continue
                    owner = resolved_owner
                scores[owner] = scores.get(owner, 0.0) + normalized_weight

    if scores:
        return max(scores, key=scores.get), False, invalid_joint_references, invalid_face_indices
    return fallback_bone, fallback_bone is not None, invalid_joint_references, invalid_face_indices


def partition_report_to_dict(report: PartitionReport) -> dict[str, Any]:
    total_faces = sum(primitive.faces for primitive in report.primitives)
    assigned_faces = sum(primitive.assigned_faces for primitive in report.primitives)
    fallback_faces = sum(primitive.fallback_faces for primitive in report.primitives)
    unassigned_faces = sum(primitive.unassigned_faces for primitive in report.primitives)
    unweighted_vertices = sum(primitive.unweighted_vertices for primitive in report.primitives)
    invalid_joint_references = sum(primitive.invalid_joint_references for primitive in report.primitives)
    invalid_face_indices = sum(primitive.invalid_face_indices for primitive in report.primitives)
    empty_bones = sum(1 for bone in report.bones.values() if bone.faces == 0)

    return {
        "file": str(report.path),
        "preset": report.bone_resolution.preset,
        "scenes": report.scenes,
        "nodes": report.nodes,
        "meshes": report.meshes,
        "skins": report.skins,
        "animations": report.animations,
        "materials": report.materials,
        "totals": {
            "bones": len(report.bones),
            "original_bones": report.bone_resolution.original_bones,
            "kept_bones": report.bone_resolution.kept_bones,
            "merged_bones": len(report.bone_resolution.merged_to_parent),
            "ignored_bones": len(report.bone_resolution.ignored),
            "empty_bones": empty_bones,
            "processed_primitives": len(report.primitives),
            "skipped_primitives": report.skipped_primitives,
            "faces": total_faces,
            "assigned_faces": assigned_faces,
            "fallback_faces": fallback_faces,
            "unassigned_faces": unassigned_faces,
            "unweighted_vertices": unweighted_vertices,
            "invalid_joint_references": invalid_joint_references,
            "invalid_face_indices": invalid_face_indices,
        },
        "bone_resolution": bone_resolution_to_dict(report.bone_resolution),
        "bones": [bone_to_dict(bone) for bone in sorted(report.bones.values(), key=lambda item: item.node_index)],
        "primitives": [primitive_to_dict(primitive, report.bones) for primitive in report.primitives],
        "warnings": report.warnings,
    }


def bone_resolution_to_dict(resolution: BoneResolutionReport) -> dict[str, Any]:
    return {
        "preset": resolution.preset,
        "original_bones": resolution.original_bones,
        "kept_bones": resolution.kept_bones,
        "merged_to_parent": [resolved_bone_to_dict(bone) for bone in resolution.merged_to_parent],
        "ignored": [resolved_bone_to_dict(bone) for bone in resolution.ignored],
        "warnings": resolution.warnings,
    }


def resolved_bone_to_dict(info: ResolvedBoneInfo) -> dict[str, Any]:
    return {
        "node_index": info.node_index,
        "name": info.name,
        "action": info.action,
        "reason": info.reason,
        "resolved_to": info.resolved_to,
        "resolved_to_name": info.resolved_to_name,
    }


def bone_to_dict(bone: BonePartition) -> dict[str, Any]:
    return {
        "node_index": bone.node_index,
        "name": bone.name,
        "parent": bone.parent,
        "children": bone.children,
        "skin_indices": sorted(bone.skin_indices),
        "faces": bone.faces,
        "vertices": len(bone.vertices),
        "primitives": len(bone.primitives),
        "fallback_faces": bone.fallback_faces,
    }


def primitive_to_dict(primitive: PrimitivePartition, bones: dict[int, BonePartition]) -> dict[str, Any]:
    owners = []
    for bone_index, faces in sorted(primitive.owners.items(), key=lambda item: (-item[1], item[0])):
        bone = bones.get(bone_index)
        owners.append(
            {
                "bone_node_index": bone_index,
                "bone_name": bone.name if bone is not None else f"bone_{bone_index}",
                "faces": faces,
                "vertices": len(primitive.owner_vertices.get(bone_index, set())),
            }
        )

    return {
        "node_index": primitive.node_index,
        "mesh_index": primitive.mesh_index,
        "primitive_index": primitive.primitive_index,
        "skin_index": primitive.skin_index,
        "material_id": primitive.material_id,
        "faces": primitive.faces,
        "assigned_faces": primitive.assigned_faces,
        "fallback_faces": primitive.fallback_faces,
        "unassigned_faces": primitive.unassigned_faces,
        "unweighted_vertices": primitive.unweighted_vertices,
        "invalid_joint_references": primitive.invalid_joint_references,
        "invalid_face_indices": primitive.invalid_face_indices,
        "owners": owners,
    }


def write_partition_report(report: PartitionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(partition_report_to_dict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def format_partition_summary(report: PartitionReport) -> str:
    data = partition_report_to_dict(report)
    totals = data["totals"]
    lines = [
        f"File: {report.path}",
        f"Preset: {data['preset']}",
        f"Skins: {report.skins}",
        f"Bones: {totals['kept_bones']} kept / {totals['original_bones']} original",
        f"Merged bones: {totals['merged_bones']}",
        f"Ignored bones: {totals['ignored_bones']}",
        f"Empty bones: {totals['empty_bones']}",
        f"Processed primitives: {totals['processed_primitives']}",
        f"Skipped primitives: {totals['skipped_primitives']}",
        f"Faces: {totals['faces']}",
        f"Assigned faces: {totals['assigned_faces']}",
        f"Fallback faces: {totals['fallback_faces']}",
        f"Unassigned faces: {totals['unassigned_faces']}",
        f"Invalid joint references: {totals['invalid_joint_references']}",
    ]

    bones_with_faces = [bone for bone in data["bones"] if bone["faces"] > 0]
    if bones_with_faces:
        lines.extend(["", "Bone face ownership:"])
        for bone in sorted(bones_with_faces, key=lambda item: (-item["faces"], item["node_index"])):
            lines.append(
                "  "
                f"node={bone['node_index']} name={bone['name']} "
                f"faces={bone['faces']} vertices={bone['vertices']} fallback={bone['fallback_faces']}"
            )

    if report.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in report.warnings)

    return "\n".join(lines)
