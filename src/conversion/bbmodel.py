from __future__ import annotations

import math
import uuid
from typing import Any

from ..errors import ConvertError
from ..partition import BonePartition
from .geometry import has_nonzero_rotation, identity_matrix, matrix_translation, to_blockbench_space
from .types import Cuboid


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
