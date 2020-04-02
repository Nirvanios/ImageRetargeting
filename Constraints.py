from typing import Tuple

import numpy as np

import Utils
from Point import PointType


def boundary_constraint_fun(points: np.ndarray, type_map: np.ndarray, target_shape: Tuple) -> int:
    points = points.reshape((-1, 2))
    sum_U = 0
    sum_B = 0
    sum_R = 0
    sum_L = 0
    for index, point in enumerate(points):
        if type_map[index].has_flag(PointType.BORDER):
            if type_map[index].has_flag(PointType.UP_B):
                sum_U += abs(point[1] - 0)
            elif type_map[index].has_flag(PointType.BOTTOM_B):
                sum_B += abs(point[1] - (target_shape[0] - 1))
            elif type_map[index].has_flag(PointType.LEFT_B):
                sum_L += abs(point[0] - 0)
            elif type_map[index].has_flag(PointType.RIGHT_B):
                sum_R += abs(point[0] - (target_shape[1] - 1))
    points = points.reshape(len(points) * 2)
    return sum_U + sum_B + sum_L + sum_R


def saliency_constraint_fun(points: np.ndarray, point_attributes: np.ndarray, obj_count: int) -> int:
    points = points.reshape((-1, 2))
    object_sums = np.zeros((obj_count, 2))
    point_counts = np.zeros(obj_count)
    pp = [[] for _ in range(obj_count)]

    for index, point in enumerate(points):
        if point_attributes[index].type.has_flag(PointType.SALIENCY):
            for object_parameter in point_attributes[index].object_parameters:
                object_sums[object_parameter.id] += point
                point_counts[object_parameter.id] += 1
                pp[object_parameter.id].append(point)
    object_centers = np.array([n / point_counts[i] for i, n in enumerate(object_sums)])

    object_sums = np.zeros(obj_count)
    for index, point in enumerate(points):
        if point_attributes[index].type.has_flag(PointType.SALIENCY):
            for object_parameter in point_attributes[index].object_parameters:
                tmp_point = Utils.pol2cart(
                    object_parameter.r * object_parameter.scale, object_parameter.theta,
                    object_centers[object_parameter.id])
                object_sums[object_parameter.id] += np.linalg.norm(point - tmp_point)

    points = points.reshape(len(points) * 2)
    return object_sums.sum()


def structure_constraint_energy_fun() -> int:
    return 0


def length_constraint_energy_fun() -> int:
    return 0
