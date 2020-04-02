from typing import Tuple

import numpy as np

import Utils
from Point import PointType


def boundary_constraint_fun(points: np.ndarray, type_map: np.ndarray, target_shape: Tuple) -> float:
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


def saliency_constraint_fun(points: np.ndarray, point_attributes: np.ndarray, obj_count: int) -> float:
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


def length_constraint_energy_fun(points: np.ndarray, point_attributes: np.ndarray, edges: np.ndarray,
                                 saliency_objects: np.ndarray, src_shape: np.ndarray,
                                 target_shape: np.ndarray) -> float:
    points = points.reshape((-1, 2))
    sum = 0
    for edge in edges:
        scale_factor = np.zeros(2)
        sum_scales = np.zeros(2)
        sum_lengths = np.zeros(2)
        point1 = points[edge[0]]
        point2 = points[edge[1]]
        length = abs(point1 - point2)
        norm = np.linalg.norm(point1 - point2)
        for saliency_object in saliency_objects:
            is_in_obj = Utils.is_edge_in_object(point1, point2, saliency_object)
            if is_in_obj[0] or is_in_obj[1]:
                extremes = np.array(saliency_object.get_extremes())
                lengths = abs(np.subtract(extremes[:, 0], extremes[:, 1]))
                sum_scales += saliency_object.scale * lengths * is_in_obj
                sum_lengths += lengths * is_in_obj
        scale_factor = (target_shape[::-1] - sum_scales) / (src_shape[::-1] - sum_lengths)
        l_ij = np.sqrt(np.power((scale_factor * length), 2).sum())
        l_ij2 = np.power(l_ij, 2)
        sum += np.power(np.power(norm, 2) - l_ij2, 2) / l_ij2

    points = points.reshape(len(points) * 2)
    return sum



def structure_constraint_energy_fun() -> int:
    return 0
