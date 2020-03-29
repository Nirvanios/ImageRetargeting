import numpy as np
from typing import List, Tuple
from  Point import PointType

def boundary_constraint_fun(points: np.ndarray, type_map: np.ndarray, target_shape : Tuple) -> int:
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

def saliency_constraint_fun() -> int:
    return 0

def structure_constraint_energy_fun() -> int:
    return 0

def length_constraint_energy_fun() -> int:
    return 0

