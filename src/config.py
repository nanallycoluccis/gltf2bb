from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True)
class BoneFilterConfig:
    ignore_name_contains: tuple[str, ...] = ()
    merge_to_parent_name_contains: tuple[str, ...] = ()
    merge_to_parent_name_regex: tuple[str, ...] = ()
    case_sensitive: bool = False
    report_merged_bones: bool = True


@dataclass(frozen=True)
class ConnectedComponentsConfig:
    enabled: bool = True
    min_faces: int = 0
    merge_tiny_components_to_nearest: bool = True
    delete_tiny_components: bool = False


@dataclass(frozen=True)
class ComplexSplitConfig:
    enabled: bool = False
    bones: tuple[str, ...] = ()
    case_sensitive: bool = False
    connected_components: ConnectedComponentsConfig = field(default_factory=ConnectedComponentsConfig)


@dataclass(frozen=True)
class CleanupConfig:
    delete_small_parts: bool = False
    min_faces: int = 0
    min_bbox_volume: float = 0.0
    merge_small_parts_to_parent: bool = False


@dataclass(frozen=True)
class OrientedCubesConfig:
    enabled: bool = False
    bones: tuple[str, ...] = ()
    scope: str = "complex_split_parts"
    case_sensitive: bool = False


@dataclass(frozen=True)
class HybridDetailSplitConfig:
    enabled: bool = True
    min_faces: int = 200
    max_long_dim_ratio: float = 0.16
    by_material: bool = True
    by_connected_component: bool = True
    min_material_faces: int = 48
    min_material_ratio: float = 0.10
    min_component_faces: int = 16
    min_component_ratio: float = 0.05


@dataclass(frozen=True)
class ProcessingConfig:
    preset: str = "default"
    bone_filter: BoneFilterConfig = field(default_factory=BoneFilterConfig)
    complex_split: ComplexSplitConfig = field(default_factory=ComplexSplitConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    oriented_cubes: OrientedCubesConfig = field(default_factory=OrientedCubesConfig)
    hybrid_detail_split: HybridDetailSplitConfig = field(default_factory=HybridDetailSplitConfig)


BUILTIN_PRESETS: dict[str, ProcessingConfig] = {
    "default": ProcessingConfig(),
    "humanoid": ProcessingConfig(
        preset="humanoid",
        bone_filter=BoneFilterConfig(
            ignore_name_contains=("IK", "physics", "control", "ctrl"),
            merge_to_parent_name_contains=("twist", "helper", "secondary", "spring"),
        ),
    ),
    "mmd_humanoid": ProcessingConfig(
        preset="mmd_humanoid",
        bone_filter=BoneFilterConfig(
            ignore_name_contains=("IK", "physics", "control", "ctrl", "操作", "物理"),
            merge_to_parent_name_contains=("twist", "helper", "secondary", "spring", "補助", "捩"),
        ),
    ),
}


def resolve_processing_config(
    preset: str | None = None,
    config_path: Path | None = None,
) -> ProcessingConfig:
    data: dict[str, Any] = {}
    if config_path is not None:
        data = read_config_file(config_path)

    preset_name = preset or config_preset_name(data) or "default"
    config = preset_config(preset_name)
    if data:
        config = merge_config_data(config, data)
    validate_config(config)
    return config


def preset_config(name: str) -> ProcessingConfig:
    try:
        return BUILTIN_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_PRESETS))
        raise ConfigError(f"unknown preset {name!r}; available presets: {available}") from exc


def read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"config path is not a file: {path}")
    if path.suffix.lower() not in {".json", ""}:
        raise ConfigError("config files currently support JSON only; use --preset for built-in presets")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"failed to read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config root must be a JSON object")
    return data


def config_preset_name(data: dict[str, Any]) -> str | None:
    preset = data.get("preset")
    if preset is None:
        skeleton = data.get("skeleton")
        if isinstance(skeleton, dict):
            preset = skeleton.get("preset")
    if preset is None:
        return None
    if not isinstance(preset, str):
        raise ConfigError("preset must be a string")
    return preset


def merge_config_data(config: ProcessingConfig, data: dict[str, Any]) -> ProcessingConfig:
    bone_filter = config.bone_filter
    complex_split = config.complex_split
    cleanup = config.cleanup
    oriented_cubes = config.oriented_cubes
    hybrid_detail_split = config.hybrid_detail_split

    raw_bone_filter = data.get("bone_filter")
    if isinstance(raw_bone_filter, dict):
        bone_filter = merge_bone_filter_data(bone_filter, raw_bone_filter)
    elif raw_bone_filter is not None:
        raise ConfigError("bone_filter must be an object")

    raw_bone_merge = data.get("bone_merge")
    if isinstance(raw_bone_merge, dict):
        bone_filter = merge_bone_filter_data(bone_filter, raw_bone_merge)
    elif raw_bone_merge is not None:
        raise ConfigError("bone_merge must be an object")

    raw_complex_split = data.get("complex_split")
    if isinstance(raw_complex_split, dict):
        complex_split = merge_complex_split_data(complex_split, raw_complex_split)
    elif raw_complex_split is not None:
        raise ConfigError("complex_split must be an object")

    raw_cleanup = data.get("cleanup")
    if isinstance(raw_cleanup, dict):
        cleanup = merge_cleanup_data(cleanup, raw_cleanup)
    elif raw_cleanup is not None:
        raise ConfigError("cleanup must be an object")

    raw_oriented_cubes = data.get("oriented_cubes")
    if isinstance(raw_oriented_cubes, dict):
        oriented_cubes = merge_oriented_cubes_data(oriented_cubes, raw_oriented_cubes)
    elif raw_oriented_cubes is not None:
        raise ConfigError("oriented_cubes must be an object")

    raw_hybrid_detail_split = data.get("hybrid_detail_split")
    if isinstance(raw_hybrid_detail_split, dict):
        hybrid_detail_split = merge_hybrid_detail_split_data(hybrid_detail_split, raw_hybrid_detail_split)
    elif raw_hybrid_detail_split is not None:
        raise ConfigError("hybrid_detail_split must be an object")

    return replace(
        config,
        bone_filter=bone_filter,
        complex_split=complex_split,
        cleanup=cleanup,
        oriented_cubes=oriented_cubes,
        hybrid_detail_split=hybrid_detail_split,
    )


def merge_bone_filter_data(config: BoneFilterConfig, data: dict[str, Any]) -> BoneFilterConfig:
    if data.get("enabled") is False:
        config = BoneFilterConfig()

    updates: dict[str, Any] = {}
    for key in (
        "ignore_name_contains",
        "merge_to_parent_name_contains",
        "merge_to_parent_name_regex",
    ):
        if key in data:
            updates[key] = parse_string_list(data[key], key)

    if "case_sensitive" in data:
        if not isinstance(data["case_sensitive"], bool):
            raise ConfigError("case_sensitive must be a boolean")
        updates["case_sensitive"] = data["case_sensitive"]
    if "report_merged_bones" in data:
        if not isinstance(data["report_merged_bones"], bool):
            raise ConfigError("report_merged_bones must be a boolean")
        updates["report_merged_bones"] = data["report_merged_bones"]

    return replace(config, **updates)


def merge_complex_split_data(config: ComplexSplitConfig, data: dict[str, Any]) -> ComplexSplitConfig:
    updates: dict[str, Any] = {}
    if "enabled" in data:
        if not isinstance(data["enabled"], bool):
            raise ConfigError("complex_split.enabled must be a boolean")
        updates["enabled"] = data["enabled"]
    if "bones" in data:
        updates["bones"] = parse_string_list(data["bones"], "complex_split.bones")
    if "case_sensitive" in data:
        if not isinstance(data["case_sensitive"], bool):
            raise ConfigError("complex_split.case_sensitive must be a boolean")
        updates["case_sensitive"] = data["case_sensitive"]
    if "connected_components" in data:
        raw_connected_components = data["connected_components"]
        if not isinstance(raw_connected_components, dict):
            raise ConfigError("complex_split.connected_components must be an object")
        updates["connected_components"] = merge_connected_components_data(
            config.connected_components,
            raw_connected_components,
        )

    result = replace(config, **updates)
    if result.enabled and not result.bones:
        return replace(result, bones=("head",))
    return result


def merge_connected_components_data(
    config: ConnectedComponentsConfig,
    data: dict[str, Any],
) -> ConnectedComponentsConfig:
    updates: dict[str, Any] = {}
    for key in ("enabled", "merge_tiny_components_to_nearest", "delete_tiny_components"):
        if key in data:
            if not isinstance(data[key], bool):
                raise ConfigError(f"complex_split.connected_components.{key} must be a boolean")
            updates[key] = data[key]

    if "min_faces" in data:
        min_faces = data["min_faces"]
        if not isinstance(min_faces, int) or isinstance(min_faces, bool) or min_faces < 0:
            raise ConfigError("complex_split.connected_components.min_faces must be a non-negative integer")
        updates["min_faces"] = min_faces

    return replace(config, **updates)


def merge_cleanup_data(config: CleanupConfig, data: dict[str, Any]) -> CleanupConfig:
    updates: dict[str, Any] = {}
    for key in ("delete_small_parts", "merge_small_parts_to_parent"):
        if key in data:
            if not isinstance(data[key], bool):
                raise ConfigError(f"cleanup.{key} must be a boolean")
            updates[key] = data[key]

    if "min_faces" in data:
        min_faces = data["min_faces"]
        if not isinstance(min_faces, int) or isinstance(min_faces, bool) or min_faces < 0:
            raise ConfigError("cleanup.min_faces must be a non-negative integer")
        updates["min_faces"] = min_faces

    if "min_bbox_volume" in data:
        min_bbox_volume = data["min_bbox_volume"]
        if not isinstance(min_bbox_volume, (int, float)) or isinstance(min_bbox_volume, bool) or min_bbox_volume < 0:
            raise ConfigError("cleanup.min_bbox_volume must be a non-negative number")
        updates["min_bbox_volume"] = float(min_bbox_volume)

    return replace(config, **updates)


def merge_oriented_cubes_data(config: OrientedCubesConfig, data: dict[str, Any]) -> OrientedCubesConfig:
    updates: dict[str, Any] = {}
    if "enabled" in data:
        if not isinstance(data["enabled"], bool):
            raise ConfigError("oriented_cubes.enabled must be a boolean")
        updates["enabled"] = data["enabled"]
    if "bones" in data:
        updates["bones"] = parse_string_list(data["bones"], "oriented_cubes.bones")
    if "scope" in data:
        if data["scope"] not in {"complex_split_parts", "bone_cubes", "matching_bones"}:
            raise ConfigError(
                "oriented_cubes.scope currently supports 'complex_split_parts', 'bone_cubes', or 'matching_bones'"
            )
        updates["scope"] = data["scope"]
    if "case_sensitive" in data:
        if not isinstance(data["case_sensitive"], bool):
            raise ConfigError("oriented_cubes.case_sensitive must be a boolean")
        updates["case_sensitive"] = data["case_sensitive"]

    result = replace(config, **updates)
    if result.enabled and not result.bones:
        return replace(result, bones=("head",))
    return result


def merge_hybrid_detail_split_data(
    config: HybridDetailSplitConfig,
    data: dict[str, Any],
) -> HybridDetailSplitConfig:
    updates: dict[str, Any] = {}
    for key in ("enabled", "by_material", "by_connected_component"):
        if key in data:
            if not isinstance(data[key], bool):
                raise ConfigError(f"hybrid_detail_split.{key} must be a boolean")
            updates[key] = data[key]

    for key in ("min_faces", "min_material_faces", "min_component_faces"):
        if key in data:
            updates[key] = parse_non_negative_int(data[key], f"hybrid_detail_split.{key}")

    for key in ("max_long_dim_ratio", "min_material_ratio", "min_component_ratio"):
        if key in data:
            updates[key] = parse_non_negative_number(data[key], f"hybrid_detail_split.{key}")

    return replace(config, **updates)


def parse_string_list(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(item for item in value if item)


def parse_non_negative_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{key} must be a non-negative integer")
    return value


def parse_non_negative_number(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{key} must be a non-negative number")
    return float(value)


def validate_config(config: ProcessingConfig) -> None:
    flags = 0 if config.bone_filter.case_sensitive else re.IGNORECASE
    for pattern in config.bone_filter.merge_to_parent_name_regex:
        try:
            re.compile(pattern, flags)
        except re.error as exc:
            raise ConfigError(f"invalid merge_to_parent_name_regex pattern {pattern!r}: {exc}") from exc


def preset_names() -> list[str]:
    return sorted(BUILTIN_PRESETS)
