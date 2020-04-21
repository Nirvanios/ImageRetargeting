import random
from typing import List, Tuple

import numpy as np

from Line import Line
from Point import Point


class PatchArray:
    patch_array: List[List[List[Point]]]
    step_size: int
    img_shape: Tuple

    def __init__(self, step: int, image: np.ndarray, lines: List[Line]):
        """
        Constructs patches with points
        :param step: Step size between patches
        :param image: Source image
        """
        # init class atributtes
        random.seed()
        self.step_size = step
        self.img_shape = image.shape
        self.lines = lines
        self.line_points_indices = []
        x_size = self.img_shape[1] // step
        x_size += 0 if self.img_shape[1] % step == 0 else 1
        y_size = self.img_shape[0] // step
        y_size += 0 if self.img_shape[0] % step == 0 else 1
        self.patch_array = [[[] for _ in range(x_size)] for _ in range(y_size)]

        # Fill patch array
        for x in range(self.img_shape[1]):
            for y in range(self.img_shape[0]):
                if image[y, x] == 255:
                    self.patch_array[y // step][x // step].append(Point(x, y))

    def filter_points(self):
        """
        Filters points, only one remains in each patch, creates border points and fills empty patches with random point
        :return: None
        """
        for r_index, row in enumerate(self.patch_array):
            for c_index, patch in enumerate(row):
                cleared = False
                length = len(patch)
                # if border patch then clear and add border pixel
                if r_index == 0 or c_index == 0 or (r_index + 1) * self.step_size >= self.img_shape[0] or (
                        c_index + 1) * self.step_size >= self.img_shape[1]:
                    tmp = Point(min((c_index) * self.step_size, self.img_shape[1] - 1),
                                min((r_index) * self.step_size, self.img_shape[0] - 1))
                    patch.clear()
                    cleared = True

                    if len(self.patch_array) == r_index + 1 and len(row) == c_index + 1:
                        tmp = Point(self.img_shape[1] - 1, self.img_shape[0] - 1)
                    elif len(row) == c_index + 1 and not c_index * self.step_size == self.img_shape[0] - 1:
                        tmp = Point(self.img_shape[1] - 1, r_index * self.step_size)
                    elif len(self.patch_array) == r_index + 1 and not r_index * self.step_size == self.img_shape[1] - 1:
                        tmp = Point(c_index * self.step_size, self.img_shape[0] - 1)
                    patch.append(tmp)

                xl = c_index * self.step_size
                xr = (c_index + 1) * self.step_size
                yu = self.img_shape[0] - (r_index * self.step_size)
                yb = self.img_shape[0] - ((r_index + 1) * self.step_size)
                for line in self.lines:
                    if (yb <= line.get_y(xl) <= yu) or (yb <= line.get_y(xr) <= yu) or (xl <= line.get_x(yb) <= xr) or (
                            xl <= line.get_x(yu) <= xr):
                        if not cleared:
                            patch.clear()
                            cleared = True
                        r = random.randrange(self.step_size)
                        x = (c_index * self.step_size) + r
                        y = line.get_y(x)
                        patch.append(Point(x, int(self.img_shape[0] - y), line_eq=line))

                # else if not empty, pick random and remove others
                if length > 0 and not cleared:
                    r = random.randrange(length)
                    tmp = patch[r]
                    patch.clear()
                    patch.append(tmp)
                # else create random
                elif not cleared:
                    x = random.randrange(c_index * self.step_size,
                                         min(c_index * self.step_size + self.step_size, self.img_shape[1]))
                    y = random.randrange(r_index * self.step_size,
                                         min(r_index * self.step_size + self.step_size, self.img_shape[0]))
                    patch.append(Point(x, y))

    def get_as_img(self) -> np.ndarray:
        """
        Getter for image with points
        :return: black image with points
        """
        img = np.zeros(shape=self.img_shape, dtype=np.uint8)
        for row in self.patch_array:
            for patch in row:
                if len(patch) > 0:
                    img[patch[0][1], patch[0][0]] = 255
        return img

    def get_as_ndarray(self) -> np.ndarray:
        """
        getter for array of points
        :return: array of points
        """
        points = []
        for row in self.patch_array:
            for patch in row:
                if len(patch) > 0:
                    points.extend([tuple(n) for n in patch])
        return np.array(points)

    def get_as_Points(self) -> np.ndarray:
        """
        getter for array of points
        :return: array of points
        """
        points = []
        for row in self.patch_array:
            for patch in row:
                if len(patch) > 0:
                    points.extend(patch)
        return np.array(points)
