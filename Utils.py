import math
import sys
from typing import Iterable, Callable, Any, TypeVar, Type, Tuple

import numpy as np
import scipy.spatial

T = TypeVar('T')


def find_if(container: Iterable[Type[T]], predicate: Callable[[Any], bool]) -> Type[T]:
    try:
        return next(n for idx, n in enumerate(container) if predicate(n))
    except StopIteration:
        return None


def pol2cart(r: float, theta: float, center: Tuple[float, float] = (0., 0.)) -> Tuple[float, float]:
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x + center[0], y + center[1]


def cart2pol(x: float, y: float, center: Tuple[float, float] = (0., 0.)) -> Tuple[float, float]:
    eps = np.finfo(float).eps if x - center[0] == 0 else 0
    r = math.sqrt(math.pow(x - center[0], 2) + math.pow(y - center[1], 2))
    theta = math.atan((y - center[1]) / (x - center[0] + eps))
    if (y - center[1]) <= 0 and (x - center[0]) < 0 or (y - center[1]) >= 0 > (x - center[0]):
        theta += math.radians(180)
    elif (y - center[1]) < 0 < (x - center[0]):
        theta += math.radians(360)
    return r, theta


def lines_length(lines: np.ndarray) -> float:
    tmp_lines = lines.copy()
    for index1 in range(len(lines)):
        line1 = tmp_lines[index1]
        for index2 in range(len(lines)):
            line2 = tmp_lines[index2]
            if index2 == index1:
                continue
            if line2[1] >= line1[0] and line2[1] <= line1[1] and line2[0] >= line1[0] and line2[0] <= line1[1]:
                tmp_lines[index2][0] = tmp_lines[index2][1] = 0
            elif line2[0] >= line1[0] and line2[0] <= line1[1]:
                tmp_lines[index2][0] = line1[1]
            elif line2[1] >= line1[0] and line2[1] <= line1[1]:
                tmp_lines[index2][1] = line1[0]
    sum = 0
    for line in tmp_lines:
        sum += abs(line[1] - line[0])
    return sum


def get_edges(delaunay: scipy.spatial.Delaunay) -> np.ndarray:
    edges = set()
    indices, indptr = delaunay.vertex_neighbor_vertices
    for k in range(indices.shape[0] - 1):
        for j in indptr[indices[k]:indices[k + 1]]:
            edges.add(tuple((k, j)) if k <= j else tuple((j, k)))
    return np.array([list(n) for n in edges])


def get_edge_neighbours(delaunay: scipy.spatial.Delaunay, edges: np.ndarray, points: np.ndarray) -> np.ndarray:
    neighbours = []
    indices, indptr = delaunay.vertex_neighbor_vertices
    for edge in edges:
        n1 = indptr[indices[edge[0]]:indices[edge[0] + 1]]
        n2 = indptr[indices[edge[1]]:indices[edge[1] + 1]]
        n = np.intersect1d(n1, n2)

        if n.shape[0] > 2:
            orientaions = []
            for _n in n:
                orientaions.append(compute_orientations(points, np.append(np.array(_n), edge).reshape((1, -1))))
            distances = np.array([sys.float_info.max] * 2)
            nn = np.zeros(2)
            for index, orientaion in enumerate(orientaions):
                d = np.linalg.norm(points[edge[0]] - points[n[index]]) + np.linalg.norm(
                    points[edge[1]] - points[n[index]])
                if orientaion < 0 and distances[0] > d:
                    nn[0] = n[index]
                    distances[0] = d
                elif orientaion > 0 and distances[1] > d:
                    nn[1] = n[index]
                    distances[1] = d
            n = nn

        n = np.broadcast_to(n, 2)
        for _n in n:
            neighbours.append(np.append(_n, edge))
    return np.array(neighbours).astype(np.uint)


def compute_orientations(points: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    return np.cross(points[neighbours[:, 0]] - points[neighbours[:, 1]],
                    points[neighbours[:, 0]] - points[neighbours[:, 2]])


def compute_orientations_jacob(points: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    return np.cross(points[:, neighbours[:, 0]] - points[:, neighbours[:, 1]],
                    points[:, neighbours[:, 0]] - points[:, neighbours[:, 2]]).reshape((points.shape[0], -1, 2))
