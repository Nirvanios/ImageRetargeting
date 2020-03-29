from enum import Enum


class PointType(Enum):
    NONE = 0
    CORNER = 1
    BORDER = 2
    SALIENCY = 3
    OTHER = 4


class Point:
    def __init__(self, x: int = 0, y: int = 0, type: PointType = PointType.NONE):
        self.x = x
        self.y = y
        self.type = type
