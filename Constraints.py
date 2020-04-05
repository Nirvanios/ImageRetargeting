from dataclasses import dataclass
from typing import List, Any

import numpy as np

from Point import PointType


@dataclass
class ConstraintAttributes:
    point_attributes: np.ndarray
    point_scales: np.ndarray
    src_shape: np.ndarray
    target_shape: np.ndarray
    obj_count: int
    saliency_objects: np.ndarray
    edges: np.ndarray
    saliency_objects_indices: np.ndarray
    saliency_objects_relative_pos: np.ndarray
    border_points_indices: List


def boundary_constraint_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))
    extremes = attributes.target_shape - 1
    sum = np.abs(
        points[attributes.border_points_indices[0]][:, 1] - 0).sum() + np.abs(
        points[attributes.border_points_indices[1]][:, 1] - extremes[0]).sum() + np.abs(
        points[attributes.border_points_indices[2]][:, 0] - 0).sum() + np.abs(
        points[attributes.border_points_indices[3]][:, 0] - extremes[1]).sum()
    points = points.reshape(len(points) * 2)
    return sum


def saliency_constraint_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))
    sums = 0
    for index in range(attributes.obj_count):
        object_center = np.average(points[attributes.saliency_objects_indices[index]], axis=0)

        desired_position = attributes.saliency_objects_relative_pos[index] + object_center
        difference = points[attributes.saliency_objects_indices[index]] - desired_position
        norms = np.linalg.norm(difference, axis=1)
        sums += norms.sum()

    points = points.reshape(len(points) * 2)
    return sums


def length_constraint_energy_fun(points: np.ndarray, args: List[Any]) -> float:
    attributes = args[0]
    points = points.reshape((-1, 2))

    lengths = abs(points[attributes.edges][:, 0] - points[attributes.edges][:, 1])
    norms = np.linalg.norm(lengths, axis=1)
    scale_factors = (attributes.point_scales[attributes.edges[:, 0]] + attributes.point_scales[
        attributes.edges[:, 1]]) / 2
    l_ij = np.sqrt(np.sum(np.power(scale_factors * lengths, 2), axis=1))
    l_ij2 = np.power(l_ij, 2)
    sum = np.sum(np.power(np.power(norms, 2) - l_ij2, 2) / l_ij2)

    return sum


def structure_constraint_energy_fun() -> int:
    return 0
