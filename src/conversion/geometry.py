from __future__ import annotations

import math
from typing import Any

from ..errors import ConvertError
from .constants import (
    EPSILON,
    MIN_CUBE_SIZE,
    ZERO_THICKNESS_DIMENSION_RATIO,
    ZERO_THICKNESS_MIN_PLANE_DIMENSION,
)
from .types import BBoxAccumulator, Cuboid


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


def clamp_point_to_bbox(point: list[float], min_xyz: list[float], max_xyz: list[float]) -> list[float]:
    return [min(max(point[index], min_xyz[index]), max_xyz[index]) for index in range(3)]


def bbox_dimensions(min_xyz: list[float], max_xyz: list[float]) -> list[float]:
    return [max_xyz[index] - min_xyz[index] for index in range(3)]


def bbox_volume(min_xyz: list[float], max_xyz: list[float]) -> float:
    return box_volume(min_xyz, max_xyz)


def box_volume(min_xyz: list[float], max_xyz: list[float]) -> float:
    return box_volume_from_dimensions(bbox_dimensions(min_xyz, max_xyz))


def box_volume_from_dimensions(dimensions: list[float]) -> float:
    volume = 1.0
    for dimension in dimensions:
        volume *= max(dimension, 0.0)
    return volume


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


def compute_world_matrices(nodes: list[dict[str, Any]], parent_map: dict[int, int]) -> dict[int, list[list[float]]]:
    cache: dict[int, list[list[float]]] = {}
    visiting: set[int] = set()

    def compute(node_index: int) -> list[list[float]]:
        from ..inspect import is_valid_index

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


def points_centroid(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    return [sum(point[index] for point in points) / len(points) for index in range(3)]


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
