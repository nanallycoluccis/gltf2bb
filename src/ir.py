from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PrimitiveStats:
    mesh_index: int
    primitive_index: int
    mode: int
    vertices: int
    faces: int
    joints_vertices: int
    weighted_vertices: int
    unweighted_vertices: int
    invalid_joint_vertices: int
    material_id: int | None


@dataclass
class InspectStats:
    path: Path
    scenes: int
    nodes: int
    meshes: int
    primitives: int
    skins: int
    animations: int
    materials: int
    vertices: int = 0
    faces: int = 0
    joints_vertices: int = 0
    weighted_vertices: int = 0
    missing_joints: int = 0
    missing_weights: int = 0
    invalid_joint_vertices: int = 0
    skin_joints: int = 0
    unique_joint_nodes: int = 0
    root_joints: int = 0
    named_joints: int = 0
    unnamed_joints: int = 0
    inverse_bind_matrices: int = 0
    skins_with_inverse_bind_matrices: int = 0
    warnings: list[str] = field(default_factory=list)
    primitives_detail: list[PrimitiveStats] = field(default_factory=list)
