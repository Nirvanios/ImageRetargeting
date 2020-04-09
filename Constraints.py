from dataclasses import dataclass
from typing import List

import numpy as np


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


def boundary_constraint_fun(points: np.ndarray, target_shape: np.ndarray, border_points_indices: np.ndarray) -> float:
    points = points.reshape((-1, 2))
    extremes = target_shape - 1
    sum = np.abs(
        points[border_points_indices[0]][:, 1] - 0).sum() + np.abs(
        points[border_points_indices[1]][:, 1] - extremes[0]).sum() + np.abs(
        points[border_points_indices[2]][:, 0] - 0).sum() + np.abs(
        points[border_points_indices[3]][:, 0] - extremes[1]).sum()
    points = points.reshape(len(points) * 2)
    return sum


def boundary_constraint_jac(points: np.ndarray, target_shape: np.ndarray, border_points_indices: np.ndarray,
                            jacob_matrix: np.ndarray) -> np.ndarray:
    points = points.reshape((-1, 2))

    j_points = jacob_matrix.reshape((-1, *points.shape)) + points
    extremes = target_shape - 1
    sum = np.abs(
        j_points[:, border_points_indices[0], 1] - 0).sum(axis=1) + np.abs(
        j_points[:, border_points_indices[1], 1] - extremes[0]).sum(axis=1) + np.abs(
        j_points[:, border_points_indices[2], 0] - 0).sum(axis=1) + np.abs(
        j_points[:, border_points_indices[3], 0] - extremes[1]).sum(axis=1)
    points = points.reshape(len(points) * 2)
    return sum


def saliency_constraint_fun(points: np.ndarray, saliency_objects_indices: np.ndarray,
                            saliency_objects_relative_pos: np.ndarray, obj_count: int) -> float:
    points = points.reshape((-1, 2))
    sums = 0
    for index in range(obj_count):
        object_center = np.mean(points[np.array(saliency_objects_indices[index])], axis=0)

        desired_position = saliency_objects_relative_pos[index] + object_center
        difference = points[np.array(saliency_objects_indices[index])] - desired_position
        norms = np.linalg.norm(difference, axis=1)
        sums += norms.sum()

    points = points.reshape(len(points) * 2)
    return sums


def saliency_constraint_jac(points: np.ndarray, saliency_objects_indices: np.ndarray,
                            saliency_objects_relative_pos: np.ndarray, obj_count: int,
                            jacob_matrix: np.ndarray) -> np.ndarray:
    points = points.reshape((-1, 2))
    j_points = jacob_matrix.reshape((-1, *points.shape)) + points
    sums = np.zeros(points.shape[0]*2, dtype='float64')
    for index in range(obj_count):
        object_center = np.mean(j_points[:, np.array(saliency_objects_indices[index])], axis=1)

        desired_position = np.broadcast_to(saliency_objects_relative_pos[index], (points.shape[0]*2, len(saliency_objects_relative_pos[index]), 2)) + np.repeat(object_center, len(saliency_objects_relative_pos[index]), axis=0).reshape((points.shape[0]*2, len(saliency_objects_relative_pos[index]), 2))
        difference = j_points[:, np.array(saliency_objects_indices[index])] - desired_position
        norms = np.linalg.norm(difference, axis=2)
        sums += norms.sum(axis=1)

    points = points.reshape(len(points) * 2)
    return sums


def length_constraint_energy_fun(points: np.ndarray, edges: np.ndarray, point_scales: np.ndarray) -> float:
    points = points.reshape((-1, 2))

    lengths = abs(points[edges][:, 0] - points[edges][:, 1])
    norms = np.linalg.norm(lengths, axis=1)
    scale_factors = (point_scales[edges[:, 0]] + point_scales[
        edges[:, 1]]) / 2
    l_ij = np.sqrt(np.sum(np.power(scale_factors * lengths, 2), axis=1))
    l_ij2 = np.power(l_ij, 2)
    sum = np.sum(np.power(np.power(norms, 2) - l_ij2, 2) / l_ij2)

    return sum


def length_constraint_energy_jac(points: np.ndarray, edges: np.ndarray, point_scales: np.ndarray,
                                 jacob_matrix: np.ndarray) -> np.ndarray:
    points = points.reshape((-1, 2))
    j_points = jacob_matrix.reshape((-1, *points.shape)) + points
    length_constraint_energy_fun(j_points[0], edges, point_scales)

    lengths = abs(j_points[:, edges][:, :, 0] - j_points[:, edges][:, :, 1])
    norms = np.linalg.norm(lengths, axis=2)
    scale_factors = (point_scales[edges[:, 0]] + point_scales[edges[:, 1]]) / 2
    l_ij = np.sqrt(np.sum(np.power(np.broadcast_to(scale_factors, (j_points.shape[0], *scale_factors.shape)) * lengths, 2), axis=2))
    l_ij2 = np.power(l_ij, 2)
    sum = np.sum(np.power(np.power(norms, 2) - l_ij2, 2) / l_ij2, axis=1)



    return sum


def structure_constraint_energy_fun() -> int:
    return 0
