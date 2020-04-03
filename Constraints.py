from dataclasses import dataclass
from typing import List, Any

import numpy as np
from timeit import default_timer as timer

import Utils
from Point import PointType


@dataclass
class ConstraintAttributes:
    point_attributes: np.ndarray
    src_shape: np.ndarray
    target_shape: np.ndarray
    obj_count: int
    saliency_objects: np.ndarray
    edges: np.ndarray


def boundary_constraint_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))
    sum_U = 0
    sum_B = 0
    sum_R = 0
    sum_L = 0
    for index, point in enumerate(points):
        if attributes.point_attributes[index].type.has_flag(PointType.BORDER):
            if attributes.point_attributes[index].type.has_flag(PointType.UP_B):
                sum_U += abs(point[1] - 0)
            elif attributes.point_attributes[index].type.has_flag(PointType.BOTTOM_B):
                sum_B += abs(point[1] - (attributes.target_shape[0] - 1))
            elif attributes.point_attributes[index].type.has_flag(PointType.LEFT_B):
                sum_L += abs(point[0] - 0)
            elif attributes.point_attributes[index].type.has_flag(PointType.RIGHT_B):
                sum_R += abs(point[0] - (attributes.target_shape[1] - 1))
    points = points.reshape(len(points) * 2)
    return sum_U + sum_B + sum_L + sum_R


def saliency_constraint_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))
    object_sums = np.zeros((attributes.obj_count, 2))
    point_counts = np.zeros(attributes.obj_count)
    pp = [[] for _ in range(attributes.obj_count)]

    for index, point in enumerate(points):
        if attributes.point_attributes[index].type.has_flag(PointType.SALIENCY):
            for object_parameter in attributes.point_attributes[index].object_parameters:
                object_sums[object_parameter.id] += point
                point_counts[object_parameter.id] += 1
                pp[object_parameter.id].append(point)
    object_centers = np.array([n / point_counts[i] for i, n in enumerate(object_sums)])

    object_sums = np.zeros(attributes.obj_count)
    for index, point in enumerate(points):
        if attributes.point_attributes[index].type.has_flag(PointType.SALIENCY):
            for object_parameter in attributes.point_attributes[index].object_parameters:
                tmp_point = Utils.pol2cart(
                    object_parameter.r * object_parameter.scale, object_parameter.theta,
                    object_centers[object_parameter.id])
                object_sums[object_parameter.id] += np.linalg.norm(point - tmp_point)

    points = points.reshape(len(points) * 2)
    return object_sums.sum() * 0


def length_constraint_energy_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))
    extremes = np.array([np.array(saliency_object.get_extremes()) for saliency_object in attributes.saliency_objects])
    extremes_lengths = abs(np.subtract(extremes[:, :, 0], extremes[:, :, 1]))
    scales = np.array([[s.scale, s.scale] for s in attributes.saliency_objects])
    point_scales = np.zeros_like(points)
    for index, point in enumerate(points):
        is_in = np.logical_and(extremes[:, :, 0] <= point, point <= extremes[:, :, 1])
        a = attributes.target_shape - np.sum(scales * extremes_lengths * is_in, axis=0)
        b = attributes.src_shape - np.sum(extremes_lengths * is_in, axis=0)
        point_scales[index] = a / b

    lengths = abs(points[attributes.edges][:, 0] - points[attributes.edges][:, 1])
    norms = np.linalg.norm(lengths, axis=1)
    scale_factors = (point_scales[attributes.edges[:, 0]] + point_scales[attributes.edges[:, 1]]) / 2
    l_ij = np.sqrt(np.sum(np.power(scale_factors * lengths, 2), axis=1))
    l_ij2 = np.power(l_ij, 2)
    sum = np.sum(np.power(np.power(norms, 2) - l_ij2, 2) / l_ij2)

    return sum * 0



def structure_constraint_energy_fun() -> int:
    return 0
