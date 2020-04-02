from enum import Flag, auto

class PointType(Flag):
    NONE = 0
    CORNER = auto()
    BORDER = auto()
    SALIENCY = auto()
    UP_B = auto()
    BOTTOM_B = auto()
    LEFT_B = auto()
    RIGHT_B = auto()
    OTHER = auto()

    def has_flag(self, flag: "PointType"):
        return self & flag == flag


class Point:
    def __init__(self, x: int = 0, y: int = 0, type: PointType = PointType.NONE):
        self.x = x
        self.y = y
        self.type = type
        self.object_parameters = []

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
