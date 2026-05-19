from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import FaceFeatureProtectionConfig, HybridDetailSplitConfig, UnskinnedMeshesConfig
from ..errors import ConvertError
from ..partition import bone_resolution_to_dict
from .bbmodel import rounded_vec
from .constants import (
    EPSILON,
    MIN_CUBE_SIZE,
    QUALITY_ELONGATED_DIMENSION_RATIO,
    QUALITY_LARGEST_CUBES_LIMIT,
    QUALITY_TINY_FRAGMENT_CUBES_LIMIT,
    QUALITY_TINY_FRAGMENT_MAX_FACES,
)
from .geometry import box_volume_from_dimensions, cube_dimensions, cube_volume, model_height_from_cubes, rounded_number
from .types import (
    AssignedUnskinnedMesh,
    CleanupPartReport,
    CleanupReport,
    ComplexSplitBoneReport,
    ComplexSplitSubpartReport,
    ConvertResult,
    Cuboid,
    FaceFeatureProtectionAction,
    HybridModeReport,
    OrientedCubeReport,
    OrientationDecisionReport,
    SkippedUnskinnedMesh,
)


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
            "assigned_unskinned_meshes": len(result.assigned_unskinned_meshes),
            "skipped_unskinned_meshes": len(result.skipped_unskinned_meshes),
        },
        "bone_resolution": bone_resolution_to_dict(result.bone_resolution),
        "hybrid": hybrid_to_dict(result.hybrid),
        "hybrid_detail_split": hybrid_detail_split_to_dict(result.hybrid_detail_split),
        "face_feature_protection": face_feature_protection_to_dict(
            result.face_feature_protection,
            result.face_feature_protection_actions,
        ),
        "unskinned_meshes": unskinned_meshes_to_dict(
            result.unskinned_meshes,
            result.assigned_unskinned_meshes,
            result.skipped_unskinned_meshes,
        ),
        "complex_split": [complex_split_to_dict(item) for item in result.complex_split],
        "cleanup": cleanup_to_dict(result.cleanup),
        "oriented_cubes": [oriented_cube_to_dict(item) for item in result.oriented_cubes],
        "orientation_decisions": [orientation_decision_to_dict(item) for item in result.orientation_decisions],
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


def face_feature_protection_to_dict(
    config: FaceFeatureProtectionConfig,
    actions: list[FaceFeatureProtectionAction],
) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "min_faces": config.min_faces,
        "margin_ratio": config.margin_ratio,
        "outlier_gap_ratio": config.outlier_gap_ratio,
        "protect_hair_front": config.protect_hair_front,
        "protect_head_core_front": config.protect_head_core_front,
        "actions": [face_feature_protection_action_to_dict(action) for action in actions],
    }


def face_feature_protection_action_to_dict(item: FaceFeatureProtectionAction) -> dict[str, Any]:
    return {
        "owner_bone": item.owner_bone,
        "owner_bone_name": item.owner_bone_name,
        "adjusted_part_name": item.adjusted_part_name,
        "cube_name": item.cube_name,
        "action": item.action,
        "before_bbox": bbox_to_dict(item.before_bbox),
        "after_bbox": bbox_to_dict(item.after_bbox),
        "feature_bbox": bbox_to_dict(item.feature_bbox),
        "protected_feature_names": list(item.protected_feature_names),
        "axis": item.axis,
        "target_value": rounded_number_or_error(item.target_value) if item.target_value is not None else None,
        "margin": rounded_number_or_error(item.margin) if item.margin is not None else None,
        "front_sign": item.front_sign,
        "overlap_axes": list(item.overlap_axes),
    }


def bbox_to_dict(bbox: tuple[list[float], list[float]]) -> dict[str, Any]:
    return {"min": rounded_vec(bbox[0]), "max": rounded_vec(bbox[1])}


def unskinned_meshes_to_dict(
    config: UnskinnedMeshesConfig,
    assigned: list[AssignedUnskinnedMesh],
    skipped: list[SkippedUnskinnedMesh],
) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "strategy": config.strategy,
        "ignore_material_name_contains": list(config.ignore_material_name_contains),
        "ignore_node_name_contains": list(config.ignore_node_name_contains),
        "ignore_mesh_name_contains": list(config.ignore_mesh_name_contains),
        "case_sensitive": config.case_sensitive,
        "assigned": [assigned_unskinned_mesh_to_dict(item) for item in assigned],
        "skipped": [skipped_unskinned_mesh_to_dict(item) for item in skipped],
    }


def assigned_unskinned_mesh_to_dict(item: AssignedUnskinnedMesh) -> dict[str, Any]:
    return {
        "node_index": item.node_index,
        "mesh_index": item.mesh_index,
        "primitive_index": item.primitive_index,
        "owner_bone": item.owner_bone,
        "owner_bone_name": item.owner_bone_name,
        "part_name": item.part_name,
        "strategy": item.strategy,
        "reason": item.reason,
        "faces": item.faces,
        "vertices": item.vertices,
    }


def skipped_unskinned_mesh_to_dict(item: SkippedUnskinnedMesh) -> dict[str, Any]:
    return {
        "node_index": item.node_index,
        "mesh_index": item.mesh_index,
        "primitive_index": item.primitive_index,
        "reason": item.reason,
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
        "reason": item.reason,
        "original_bbox_volume": rounded_number(item.original_bbox_volume) if item.original_bbox_volume is not None else None,
        "oriented_bbox_volume": rounded_number(item.oriented_bbox_volume) if item.oriented_bbox_volume is not None else None,
        "cube_only_compatible": item.cube_only_compatible,
    }


def orientation_decision_to_dict(item: OrientationDecisionReport) -> dict[str, Any]:
    return {
        "name": item.name,
        "owner_bone": item.owner_bone,
        "owner_bone_name": item.owner_bone_name,
        "source": item.source,
        "accepted": item.accepted,
        "reason": item.reason,
        "rotation": rounded_vec(item.rotation),
        "original_bbox_volume": rounded_number(item.original_bbox_volume),
        "oriented_bbox_volume": rounded_number(item.oriented_bbox_volume) if item.oriented_bbox_volume is not None else None,
        "cube_only_compatible": item.cube_only_compatible,
    }


def quality_to_dict(result: ConvertResult) -> dict[str, Any]:
    return {
        "cube_only": cube_only_quality_to_dict(result),
        "cube_count_by_owner_bone": cube_count_by_owner_bone_to_dict(result.cubes),
        "largest_cubes": [quality_cube_to_dict(cuboid) for cuboid in largest_quality_cubes(result.cubes)],
        "oversized_cubes": [
            quality_oversized_cube_to_dict(cuboid)
            for cuboid in largest_quality_cubes(result.cubes)
        ],
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
        "split_diagnostics": split_diagnostics_to_dict(result),
        "cube_budget_warnings": cube_budget_warnings_to_dict(result),
    }


def cube_only_quality_to_dict(result: ConvertResult) -> dict[str, Any]:
    return {
        "cube_only": True,
        "cube_count": len(result.cubes),
        "mesh_element_count": 0,
        "mesh_element_names": [],
        "vertex_element_count": 0,
        "vertex_element_names": [],
    }


def cube_count_by_owner_bone_to_dict(cuboids: list[Cuboid]) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str]] = Counter((cuboid.owner_bone, cuboid.owner_bone_name) for cuboid in cuboids)
    return [
        {"owner_bone": owner_bone, "owner_bone_name": owner_bone_name, "cubes": count}
        for (owner_bone, owner_bone_name), count in sorted(counts.items(), key=lambda item: (item[0][1], item[0][0]))
    ]


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


def quality_oversized_cube_to_dict(cuboid: Cuboid) -> dict[str, Any]:
    data = quality_cube_to_dict(cuboid)
    data["reason"] = "largest_by_volume"
    return data


def split_diagnostics_to_dict(result: ConvertResult) -> list[dict[str, Any]]:
    cube_counts = Counter(cuboid.owner_bone for cuboid in result.cubes)
    return [
        {
            "bone": item.bone,
            "bone_name": item.bone_name,
            "source_faces": item.source_faces,
            "subpart_count": len(item.subparts),
            "output_cubes": cube_counts[item.bone],
            "methods": sorted({subpart.method for subpart in item.subparts}),
            "merged_tiny_components": item.merged_tiny_components,
            "deleted_tiny_components": item.deleted_tiny_components,
            "merged_tiny_hair_buckets": item.merged_tiny_hair_buckets,
            "expanded_hair_bucket_overlap": item.expanded_hair_bucket_overlap,
        }
        for item in result.complex_split
    ]


def cube_budget_warnings_to_dict(result: ConvertResult) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if len(result.cubes) > result.cube_budget_warning_threshold:
        warnings.append(
            {
                "scope": "model",
                "cube_count": len(result.cubes),
                "threshold": result.cube_budget_warning_threshold,
                "reason": "cube_count_exceeds_budget",
            }
        )
    for item in cube_count_by_owner_bone_to_dict(result.cubes):
        if item["cubes"] > result.cube_owner_budget_warning_threshold:
            warnings.append(
                {
                    "scope": "owner_bone",
                    "owner_bone": item["owner_bone"],
                    "owner_bone_name": item["owner_bone_name"],
                    "cube_count": item["cubes"],
                    "threshold": result.cube_owner_budget_warning_threshold,
                    "reason": "owner_cube_count_exceeds_budget",
                }
            )
    return warnings


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
        "reasons": sorted({item.reason for item in items}),
    }


def unrotated_elongated_reason(cuboid: Cuboid) -> str | None:
    from .geometry import has_nonzero_rotation

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


def cuboid_to_dict(cuboid: Cuboid) -> dict[str, Any]:
    from .geometry import has_nonzero_rotation

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
        f"Unskinned meshes: {len(result.assigned_unskinned_meshes)} assigned / {len(result.skipped_unskinned_meshes)} skipped",
        f"Face feature protection actions: {len(result.face_feature_protection_actions)}",
        f"Output: {result.output_path}",
    ]
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def rounded_number_or_error(value: float) -> float:
    if not math.isfinite(value):
        raise ConvertError("generated non-finite quality metric")
    rounded = round(value, 6)
    return 0.0 if rounded == -0.0 else rounded
