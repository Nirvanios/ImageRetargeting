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


def saliency_constraint_fun(points: np.ndarray, point_attributes: np.ndarray) -> int:
    points = points.reshape((-1, 2))
    object_sums = np.zeros((1, 2))
    point_counts = np.zeros(1)

    for index, point in enumerate(points):
        if point_attributes[index].type.has_flag(PointType.SALIENCY):
            object_sums[point_attributes[index].object_parameter.id] += point
            point_counts[point_attributes[index].object_parameter.id] += 1
    object_centers = np.array([n / point_counts[i] for i, n in enumerate(object_sums)])

    object_sums = np.zeros((1, 2))
    for index, point in enumerate(points):
        if point_attributes[index].type.has_flag(PointType.SALIENCY):
            object_sums[point_attributes[index].object_parameter.id] += np.linalg.norm(
                point - Utils.pol2cart(
                    point_attributes[index].object_parameter.r * point_attributes[index].object_parameter.scale,
                    point_attributes[index].object_parameter.theta, object_centers[point_attributes[index].object_parameter.id]))

    points = points.reshape(len(points) * 2)
    return object_sums.sum()


def structure_constraint_energy_fun() -> int:
    return 0


def length_constraint_energy_fun() -> int:
    return 0
