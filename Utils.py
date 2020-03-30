import math
from typing import Iterable, Callable, Any, TypeVar, Type, Tuple

import numpy as np

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
    r = math.sqrt(math.pow(x - center[0], 2) + math.pow(y - center[1], 2))
    theta = math.atan((y - center[1]) / (x - center[0]))
    return r, theta


def lines_length(lines: np.ndarray) -> float:
    tmp_lines = lines.copy()
    for index1 in range(len(lines)):
        line1 = tmp_lines[index1]
        for index2 in range(len(lines)):
            line2 = tmp_lines[index2]
            if index2 <= index1:
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

