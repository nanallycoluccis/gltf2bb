from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path
from typing import Any

from .config import preset_names
from .constants import (
    COMPONENT_BYTE_SIZES,
    COMPONENT_STRUCT_FORMATS,
    GLB_BIN_CHUNK_TYPE,
    GLB_JSON_CHUNK_TYPE,
    GLB_MAGIC,
    GLB_VERSION,
    MODE_TRIANGLES,
    MODE_TRIANGLE_FAN,
    MODE_TRIANGLE_STRIP,
    TYPE_COMPONENT_COUNTS,
)
from .errors import ConfigError, ConvertError, InspectError, PartitionError
from .ir import InspectStats, PrimitiveStats


SUPPORTED_MODEL_SUFFIXES = {".gltf", ".glb", ".vrm"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gltf2bb",
        description="Prepare glTF/GLB models for Blockbench workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Print model, skeleton, and skin weight statistics.",
    )
    inspect_parser.add_argument("model", type=Path, help="Path to a .gltf, .glb, or .vrm file.")

    partition_parser = subparsers.add_parser(
        "partition",
        help="Assign mesh faces to dominant skin bones and write a partition report.",
    )
    partition_parser.add_argument("model", type=Path, help="Path to a .gltf, .glb, or .vrm file.")
    partition_parser.add_argument(
        "--report",
        type=Path,
        help="Path to write report.json. If omitted, the report JSON is printed to stdout.",
    )
    partition_parser.add_argument(
        "--preset",
        choices=preset_names(),
        default="default",
        help="Built-in bone filtering preset.",
    )
    partition_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Values override the selected preset.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a skinned glTF/GLB model to a Blockbench project.",
    )
    convert_parser.add_argument("model", type=Path, help="Path to a .gltf, .glb, or .vrm file.")
    convert_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write output .bbmodel. Defaults to out/<model>.bbmodel.",
    )
    convert_parser.add_argument(
        "--mode",
        choices=["cuboid", "hybrid"],
        default="cuboid",
        help="Conversion mode. 'hybrid' keeps body cuboids and uses special cubes for complex bones.",
    )
    convert_parser.add_argument(
        "--target-height",
        type=float,
        default=32.0,
        help="Scale output so the assigned mesh bbox is this high in Blockbench units.",
    )
    convert_parser.add_argument(
        "--preset",
        choices=preset_names(),
        default="default",
        help="Built-in bone filtering preset.",
    )
    convert_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Values override the selected preset.",
    )
    convert_parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write convert report JSON.",
    )
    convert_parser.add_argument(
        "--complex-split",
        action="append",
        help="Split a complex owner bone into editable cuboid subparts (for example: head, hair, skirt, coat).",
    )

    args = parser.parse_args(argv)

    if args.command == "inspect":
        try:
            stats = inspect_model(args.model)
        except InspectError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(format_stats(stats))
        return 0

    if args.command == "partition":
        from .partition import (
            format_partition_summary,
            partition_model,
            partition_report_to_dict,
            write_partition_report,
        )

        try:
            report = partition_model(args.model, preset=args.preset, config_path=args.config)
        except (ConfigError, InspectError, PartitionError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.report is None:
            print(json.dumps(partition_report_to_dict(report), indent=2, ensure_ascii=False))
            return 0

        write_partition_report(report, args.report)
        print(format_partition_summary(report))
        print(f"Report: {args.report}")
        return 0

    if args.command == "convert":
        from .convert import convert_model, format_convert_summary

        output = args.output or Path("out") / f"{args.model.stem}.bbmodel"
        try:
            result = convert_model(
                args.model,
                output,
                mode=args.mode,
                target_height=args.target_height,
                preset=args.preset,
                config_path=args.config,
                complex_split=args.complex_split,
                report_path=args.report,
            )
        except (ConfigError, InspectError, PartitionError, ConvertError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(format_convert_summary(result))
        return 0

    parser.print_help()
    return 0


def inspect_model(path: Path) -> InspectStats:
    if not path.exists():
        raise InspectError(f"input file does not exist: {path}")
    if not path.is_file():
        raise InspectError(f"input path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise InspectError(f"expected a .gltf, .glb, or .vrm file, got: {path.name}")

    try:
        gltf, binary_chunk = load_gltf(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, struct.error) as exc:
        raise InspectError(f"failed to read glTF data from {path}: {exc}") from exc

    stats = InspectStats(
        path=path,
        scenes=len(gltf.get("scenes", [])),
        nodes=len(gltf.get("nodes", [])),
        meshes=len(gltf.get("meshes", [])),
        primitives=sum(len(mesh.get("primitives", [])) for mesh in gltf.get("meshes", [])),
        skins=len(gltf.get("skins", [])),
        animations=len(gltf.get("animations", [])),
        materials=len(gltf.get("materials", [])),
    )

    buffer_cache: dict[int, bytes] = {}
    mesh_skin_joint_counts = collect_mesh_skin_joint_counts(gltf, stats)
    collect_skeleton_stats(gltf, stats)
    collect_mesh_stats(gltf, path, binary_chunk, buffer_cache, mesh_skin_joint_counts, stats)
    return stats


def load_gltf(path: Path) -> tuple[dict[str, Any], bytes | None]:
    if path.suffix.lower() in {".glb", ".vrm"}:
        return load_glb(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    return data, None


def load_glb(path: Path) -> tuple[dict[str, Any], bytes | None]:
    data = path.read_bytes()
    if len(data) < 12:
        raise InspectError("GLB file is too small to contain a valid header")

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        raise InspectError("GLB magic header is invalid")
    if version != GLB_VERSION:
        raise InspectError(f"unsupported GLB version: {version}")
    if total_length != len(data):
        raise InspectError(
            f"GLB length mismatch: header says {total_length} bytes, file has {len(data)} bytes"
        )

    offset = 12
    json_chunk: bytes | None = None
    binary_chunk: bytes | None = None

    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length

        if chunk_type == GLB_JSON_CHUNK_TYPE:
            json_chunk = chunk
        elif chunk_type == GLB_BIN_CHUNK_TYPE:
            binary_chunk = chunk

    if json_chunk is None:
        raise InspectError("GLB file does not contain a JSON chunk")

    return json.loads(json_chunk.decode("utf-8")), binary_chunk


def collect_mesh_skin_joint_counts(gltf: dict[str, Any], stats: InspectStats) -> dict[int, list[int]]:
    mesh_skin_joint_counts: dict[int, list[int]] = {}
    skins = gltf.get("skins", [])

    for node_index, node in enumerate(gltf.get("nodes", [])):
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if mesh_index is None or skin_index is None:
            continue
        if not is_valid_index(skins, skin_index):
            stats.warnings.append(f"Node {node_index} references missing skin {skin_index}.")
            continue
        mesh_skin_joint_counts.setdefault(mesh_index, []).append(len(skins[skin_index].get("joints", [])))

    return mesh_skin_joint_counts


def collect_skeleton_stats(gltf: dict[str, Any], stats: InspectStats) -> None:
    nodes = gltf.get("nodes", [])
    accessors = gltf.get("accessors", [])
    parent_map = build_parent_map(nodes)
    unique_joints: set[int] = set()
    root_joints: set[int] = set()
    inverse_bind_matrices = 0
    skins_with_inverse_bind_matrices = 0

    for skin_index, skin in enumerate(gltf.get("skins", [])):
        joints = skin.get("joints", [])
        joint_set = set(joints)
        stats.skin_joints += len(joints)
        unique_joints.update(joints)

        for joint in joints:
            if not is_valid_index(nodes, joint):
                stats.warnings.append(f"Skin {skin_index} references missing joint node {joint}.")
                continue
            if parent_map.get(joint) not in joint_set:
                root_joints.add(joint)

        inverse_bind_accessor = skin.get("inverseBindMatrices")
        if inverse_bind_accessor is not None:
            skins_with_inverse_bind_matrices += 1
            if is_valid_index(accessors, inverse_bind_accessor):
                count = int(accessors[inverse_bind_accessor].get("count", 0))
                inverse_bind_matrices += count
                if count != len(joints):
                    stats.warnings.append(
                        f"Skin {skin_index} inverseBindMatrices count {count} "
                        f"does not match joints count {len(joints)}."
                    )
            else:
                stats.warnings.append(
                    f"Skin {skin_index} references missing inverseBindMatrices accessor "
                    f"{inverse_bind_accessor}."
                )

    stats.unique_joint_nodes = len(unique_joints)
    stats.root_joints = len(root_joints)
    stats.named_joints = sum(
        1 for joint in unique_joints if is_valid_index(nodes, joint) and nodes[joint].get("name")
    )
    stats.unnamed_joints = stats.unique_joint_nodes - stats.named_joints
    stats.inverse_bind_matrices = inverse_bind_matrices
    stats.skins_with_inverse_bind_matrices = skins_with_inverse_bind_matrices


def collect_mesh_stats(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    mesh_skin_joint_counts: dict[int, list[int]],
    stats: InspectStats,
) -> None:
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])

    for mesh_index, mesh in enumerate(meshes):
        max_skin_joints = max(mesh_skin_joint_counts.get(mesh_index, [0]))
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            position_accessor = attributes.get("POSITION")
            if not is_valid_index(accessors, position_accessor):
                stats.warnings.append(
                    f"Mesh {mesh_index} primitive {primitive_index} has no valid POSITION accessor."
                )
                continue

            vertices = int(accessors[position_accessor].get("count", 0))
            mode = int(primitive.get("mode", MODE_TRIANGLES))
            faces = count_faces(accessors, primitive, vertices, mode, stats, mesh_index, primitive_index)

            joints_accessor = attributes.get("JOINTS_0")
            weights_accessor = attributes.get("WEIGHTS_0")
            joints_vertices = count_matching_accessor(
                accessors, joints_accessor, vertices, "JOINTS_0", stats, mesh_index, primitive_index
            )
            weights_count = count_matching_accessor(
                accessors, weights_accessor, vertices, "WEIGHTS_0", stats, mesh_index, primitive_index
            )

            weighted_vertices, invalid_joint_vertices = read_weight_stats(
                gltf,
                path,
                binary_chunk,
                buffer_cache,
                weights_accessor,
                joints_accessor,
                vertices,
                max_skin_joints,
                stats,
                mesh_index,
                primitive_index,
            )
            if weights_accessor is not None and weighted_vertices is None:
                weighted_vertices = weights_count
            elif weighted_vertices is None:
                weighted_vertices = 0

            primitive_stats = PrimitiveStats(
                mesh_index=mesh_index,
                primitive_index=primitive_index,
                mode=mode,
                vertices=vertices,
                faces=faces,
                joints_vertices=joints_vertices,
                weighted_vertices=weighted_vertices,
                unweighted_vertices=max(vertices - weighted_vertices, 0),
                invalid_joint_vertices=invalid_joint_vertices,
                material_id=primitive.get("material"),
            )
            stats.primitives_detail.append(primitive_stats)

            stats.vertices += vertices
            stats.faces += faces
            stats.joints_vertices += joints_vertices
            stats.weighted_vertices += weighted_vertices
            stats.missing_joints += max(vertices - joints_vertices, 0)
            stats.missing_weights += max(vertices - weighted_vertices, 0)
            stats.invalid_joint_vertices += invalid_joint_vertices


def count_faces(
    accessors: list[dict[str, Any]],
    primitive: dict[str, Any],
    vertices: int,
    mode: int,
    stats: InspectStats,
    mesh_index: int,
    primitive_index: int,
) -> int:
    indices_accessor = primitive.get("indices")
    if indices_accessor is not None and is_valid_index(accessors, indices_accessor):
        element_count = int(accessors[indices_accessor].get("count", 0))
    else:
        element_count = vertices
        if indices_accessor is not None:
            stats.warnings.append(
                f"Mesh {mesh_index} primitive {primitive_index} references missing indices accessor "
                f"{indices_accessor}. Falling back to POSITION count."
            )

    if mode == MODE_TRIANGLES:
        return element_count // 3
    if mode in {MODE_TRIANGLE_STRIP, MODE_TRIANGLE_FAN}:
        return max(element_count - 2, 0)

    stats.warnings.append(
        f"Mesh {mesh_index} primitive {primitive_index} uses non-triangle mode {mode}; faces counted as 0."
    )
    return 0


def count_matching_accessor(
    accessors: list[dict[str, Any]],
    accessor_index: int | None,
    vertices: int,
    semantic: str,
    stats: InspectStats,
    mesh_index: int,
    primitive_index: int,
) -> int:
    if accessor_index is None:
        return 0
    if not is_valid_index(accessors, accessor_index):
        stats.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} references missing {semantic} accessor "
            f"{accessor_index}."
        )
        return 0

    count = int(accessors[accessor_index].get("count", 0))
    if count != vertices:
        stats.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} {semantic} count {count} "
            f"does not match POSITION count {vertices}."
        )
    return min(count, vertices)


def read_weight_stats(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    weights_accessor: int | None,
    joints_accessor: int | None,
    vertices: int,
    max_skin_joints: int,
    stats: InspectStats,
    mesh_index: int,
    primitive_index: int,
) -> tuple[int | None, int]:
    if weights_accessor is None:
        return 0, 0

    try:
        weights = read_accessor(gltf, path, binary_chunk, buffer_cache, weights_accessor)
    except InspectError as exc:
        stats.warnings.append(
            f"Mesh {mesh_index} primitive {primitive_index} could not read WEIGHTS_0 accessor "
            f"{weights_accessor}: {exc}"
        )
        return None, 0

    if weights is None:
        return None, 0

    weight_rows = weights[:vertices]
    weighted_vertices = sum(1 for row in weight_rows if sum(float(value) for value in row) > 1e-6)
    invalid_joint_vertices = 0

    if joints_accessor is not None and max_skin_joints > 0:
        try:
            joints = read_accessor(gltf, path, binary_chunk, buffer_cache, joints_accessor)
        except InspectError as exc:
            stats.warnings.append(
                f"Mesh {mesh_index} primitive {primitive_index} could not read JOINTS_0 accessor "
                f"{joints_accessor}: {exc}"
            )
        else:
            if joints is not None:
                joint_rows = joints[:vertices]
                for joint_row, weight_row in zip(joint_rows, weight_rows, strict=False):
                    for joint, weight in zip(joint_row, weight_row, strict=False):
                        if float(weight) > 1e-6 and int(joint) >= max_skin_joints:
                            invalid_joint_vertices += 1
                            break

    return weighted_vertices, invalid_joint_vertices


def read_accessor(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    accessor_index: int,
) -> list[list[float | int]] | None:
    accessors = gltf.get("accessors", [])
    buffer_views = gltf.get("bufferViews", [])
    if not is_valid_index(accessors, accessor_index):
        raise InspectError(f"missing accessor {accessor_index}")

    accessor = accessors[accessor_index]
    if "sparse" in accessor:
        raise InspectError("sparse accessors are not supported by M1 inspect")

    buffer_view_index = accessor.get("bufferView")
    if buffer_view_index is None:
        return [[0] * component_count(accessor) for _ in range(int(accessor.get("count", 0)))]
    if not is_valid_index(buffer_views, buffer_view_index):
        raise InspectError(f"missing bufferView {buffer_view_index}")

    buffer_view = buffer_views[buffer_view_index]
    buffer_index = int(buffer_view.get("buffer", 0))
    buffer_data = get_buffer_data(gltf, path, binary_chunk, buffer_cache, buffer_index)

    count = int(accessor.get("count", 0))
    components = component_count(accessor)
    component_type = int(accessor.get("componentType"))
    component_size = COMPONENT_BYTE_SIZES.get(component_type)
    struct_format = COMPONENT_STRUCT_FORMATS.get(component_type)
    if component_size is None or struct_format is None:
        raise InspectError(f"unsupported accessor componentType {component_type}")

    view_offset = int(buffer_view.get("byteOffset", 0))
    accessor_offset = int(accessor.get("byteOffset", 0))
    start = view_offset + accessor_offset
    default_stride = component_size * components
    stride = int(buffer_view.get("byteStride", default_stride))
    if stride < default_stride:
        raise InspectError(
            f"accessor {accessor_index} byteStride {stride} is smaller than element size {default_stride}"
        )
    normalized = bool(accessor.get("normalized", False))

    rows: list[list[float | int]] = []
    for row_index in range(count):
        row_offset = start + row_index * stride
        row_values: list[float | int] = []
        for component_index in range(components):
            value_offset = row_offset + component_index * component_size
            if value_offset + component_size > len(buffer_data):
                raise InspectError(
                    f"accessor {accessor_index} reads beyond buffer {buffer_index} length"
                )
            value = struct.unpack_from("<" + struct_format, buffer_data, value_offset)[0]
            if normalized:
                value = normalize_component(value, component_type)
            row_values.append(value)
        rows.append(row_values)
    return rows


def get_buffer_data(
    gltf: dict[str, Any],
    path: Path,
    binary_chunk: bytes | None,
    buffer_cache: dict[int, bytes],
    buffer_index: int,
) -> bytes:
    if buffer_index in buffer_cache:
        return buffer_cache[buffer_index]

    buffers = gltf.get("buffers", [])
    if not is_valid_index(buffers, buffer_index):
        raise InspectError(f"missing buffer {buffer_index}")

    buffer = buffers[buffer_index]
    uri = buffer.get("uri")
    if uri is None:
        if binary_chunk is None:
            raise InspectError(f"buffer {buffer_index} has no URI and no GLB binary chunk is available")
        data = binary_chunk
    elif uri.startswith("data:"):
        data = decode_data_uri(uri)
    else:
        buffer_path = path.parent / uri
        if not buffer_path.exists():
            raise InspectError(f"external buffer does not exist: {buffer_path}")
        data = buffer_path.read_bytes()

    expected_length = buffer.get("byteLength")
    if expected_length is not None and len(data) < int(expected_length):
        raise InspectError(
            f"buffer {buffer_index} is shorter than declared byteLength "
            f"({len(data)} < {expected_length})"
        )

    buffer_cache[buffer_index] = data
    return data


def decode_data_uri(uri: str) -> bytes:
    try:
        header, payload = uri.split(",", 1)
    except ValueError as exc:
        raise InspectError("invalid data URI buffer") from exc

    if ";base64" in header:
        try:
            return base64.b64decode(payload)
        except ValueError as exc:
            raise InspectError("invalid base64 data URI buffer") from exc
    return payload.encode("utf-8")


def component_count(accessor: dict[str, Any]) -> int:
    accessor_type = accessor.get("type")
    count = TYPE_COMPONENT_COUNTS.get(accessor_type)
    if count is None:
        raise InspectError(f"unsupported accessor type {accessor_type!r}")
    return count


def normalize_component(value: float | int, component_type: int) -> float:
    if component_type == 5120:
        return max(float(value) / 127.0, -1.0)
    if component_type == 5121:
        return float(value) / 255.0
    if component_type == 5122:
        return max(float(value) / 32767.0, -1.0)
    if component_type == 5123:
        return float(value) / 65535.0
    return float(value)


def build_parent_map(nodes: list[dict[str, Any]]) -> dict[int, int]:
    parent_map: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child_index in node.get("children", []):
            parent_map[child_index] = parent_index
    return parent_map


def is_valid_index(items: list[Any], index: Any) -> bool:
    return isinstance(index, int) and 0 <= index < len(items)


def format_stats(stats: InspectStats) -> str:
    lines = [
        f"File: {stats.path}",
        f"Scenes: {stats.scenes}",
        f"Nodes: {stats.nodes}",
        f"Meshes: {stats.meshes}",
        f"Mesh primitives: {stats.primitives}",
        f"Skins: {stats.skins}",
        f"Animations: {stats.animations}",
        f"Materials: {stats.materials}",
        f"Vertices: {stats.vertices}",
        f"Faces: {stats.faces}",
        "",
        "Skeleton statistics:",
        f"  Skin joints: {stats.skin_joints}",
        f"  Unique joint nodes: {stats.unique_joint_nodes}",
        f"  Root joints: {stats.root_joints}",
        f"  Named joints: {stats.named_joints}",
        f"  Unnamed joints: {stats.unnamed_joints}",
        f"  Inverse bind matrices: {stats.inverse_bind_matrices}",
        f"  Skins with inverse bind matrices: {stats.skins_with_inverse_bind_matrices}",
        "",
        "Weight statistics:",
        f"  Vertices with JOINTS_0: {stats.joints_vertices}",
        f"  Missing JOINTS_0: {stats.missing_joints}",
        f"  Weighted vertices: {stats.weighted_vertices}",
        f"  Missing weights: {stats.missing_weights}",
        f"  Vertices with invalid weighted joints: {stats.invalid_joint_vertices}",
    ]

    if stats.primitives_detail:
        lines.extend(["", "Primitives:"])
        for primitive in stats.primitives_detail:
            material = "none" if primitive.material_id is None else str(primitive.material_id)
            lines.append(
                "  "
                f"mesh={primitive.mesh_index} primitive={primitive.primitive_index} "
                f"mode={primitive.mode} vertices={primitive.vertices} faces={primitive.faces} "
                f"weighted={primitive.weighted_vertices} unweighted={primitive.unweighted_vertices} "
                f"material={material}"
            )

    if stats.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in stats.warnings)

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
