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

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Point index out of range")

    def __str__(self):
        return "({}, {})".format(self.x, self.y)
