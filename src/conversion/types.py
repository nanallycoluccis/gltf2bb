from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import FaceFeatureProtectionConfig, HybridDetailSplitConfig, UnskinnedMeshesConfig
from ..partition import BoneResolutionReport


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
class FaceFeatureProtectionAction:
    owner_bone: int
    owner_bone_name: str
    cube_name: str
    action: str
    before_bbox: tuple[list[float], list[float]]
    after_bbox: tuple[list[float], list[float]]
    feature_bbox: tuple[list[float], list[float]]


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
    primitive_index: int | None = None
    reason: str = "disabled"


@dataclass(frozen=True)
class AssignedUnskinnedMesh:
    node_index: int
    mesh_index: int
    primitive_index: int
    owner_bone: int
    owner_bone_name: str
    part_name: str
    strategy: str
    reason: str
    faces: int
    vertices: int


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
    unskinned_meshes: UnskinnedMeshesConfig
    face_feature_protection: FaceFeatureProtectionConfig
    face_feature_protection_actions: list[FaceFeatureProtectionAction]
    assigned_unskinned_meshes: list[AssignedUnskinnedMesh]
    skipped_unskinned_meshes: list[SkippedUnskinnedMesh]
    warnings: list[str]
    cube_budget_warning_threshold: int = 64
    cube_owner_budget_warning_threshold: int = 16
