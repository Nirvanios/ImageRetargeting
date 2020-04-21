from typing import Tuple

import numpy as np

class Line:

    def __init__(self, point1: Tuple[int, int], point2: Tuple[int, int]):
        coefficients = np.polyfit([point1[0], point2[0]], [point1[1], point2[1]], 1)
        self.a = coefficients[0]
        self.b = coefficients[1]

    def get_y(self, x: float) -> float:
        return self.a * x + self.b

    def get_x(self, y: float) -> float:
        return (y - self.b) / self.a



