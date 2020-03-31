import cv2
import numpy as np
from typing import Tuple

import Utils
from SaliencyObject import SaliencyObject, ObjectParameter
from Point import PointType


class PointsClassifier:

    def __init__(self, points: np.ndarray, simplices: np.ndarray, saliency_map: np.ndarray, target_shape: Tuple):
        """
        Fills apropriate members
        :param points: Detected points in image
        :param simplices: Indices to mesh
        :param saliency_map: Binary importance map
        """
        self.shape = saliency_map.shape
        self.target_shape = target_shape
        self.corner_points = []
        self.border_points = []
        self.saliency_objects = []
        self.edge_points = []
        self.other_points = []
        self.all_points = []

        self.__save_all_points(points)
        self.__find_border_corner_points(points)
        self.__find_saliency_objects(points, simplices, saliency_map)
        self.__find_other_points(points)

        print(list(self.saliency_objects[0].triangles.intersection(self.border_points))[0])
        print("aaa")


    def __find_border_corner_points(self, points: np.ndarray) -> None:
        """
        Finds border and corner points and fills appropriate member variable
        :param points: Input points
        :return: None
        """
        for point in points:
            if point[0] == 0 or point[1] == 0 or point[0] == (self.shape[1] - 1) or point[1] == (self.shape[0] - 1):
                if (point[0] == 0 and point[1] == 0) or (point[1] == 0 and point[0] == self.shape[1] - 1) or (
                        point[0] == self.shape[1] - 1 and point[1] == self.shape[0] - 1) or (
                        point[1] == self.shape[0] - 1 and point[0] == 0):
                    point.type = PointType.CORNER
                    self.corner_points.append(point)
                else:
                    point.type = PointType.BORDER
                    if point.x == 0:
                        point.type |= PointType.LEFT_B
                    elif point.x == self.shape[1] - 1:
                        point.type |= PointType.RIGHT_B
                    elif point.y == 0:
                        point.type |= PointType.UP_B
                    elif point.y == self.shape[0] - 1:
                        point.type |= PointType.BOTTOM_B
                    self.border_points.append(point)

    def __find_saliency_objects(self, points: np.ndarray, simplices: np.ndarray, saliency_map: np.ndarray) -> None:
        """
        Finds points containting saliency objects and fills appropriate member variable
        :param points: Input points
        :param simplices: Indices of mesh
        :param saliency_map: Binary importance map
        :return: None
        """
        for triangle in simplices:
            triangle_vertices = np.array([tuple(points[triangle[0]]), tuple(points[triangle[1]]), tuple(points[triangle[2]])],
                                         dtype=np.int32)
            triangle_mask = np.zeros_like(saliency_map, dtype=np.uint8)
            cv2.fillPoly(triangle_mask, [triangle_vertices], (255))
            triangle_vertices = [points[triangle[0]], points[triangle[1]], points[triangle[2]]]
            triangle_mask = cv2.bitwise_and(saliency_map, saliency_map, mask=triangle_mask)
            if np.sum(triangle_mask) > 0:
                val = Utils.find_if(self.saliency_objects,
                                    lambda salinecy_object, test=triangle_vertices: salinecy_object.append_if_same(
                                        test))
                if val is None:
                    for point in triangle_vertices:
                        point.type |= PointType.SALIENCY
                    self.saliency_objects.append(SaliencyObject(triangle_vertices))

        condition = True
        while condition:
            condition = False
            index = 0
            while index < len(self.saliency_objects):
                marked_for_del = []
                for to_check_index, to_check in enumerate(self.saliency_objects):
                    if to_check_index <= index:
                        continue
                    if self.saliency_objects[index].append_if_same(to_check):
                        marked_for_del.append(to_check_index)
                        condition = True
                for del_index in sorted(marked_for_del, reverse=True):
                    del self.saliency_objects[del_index]
                index += 1

        x_lines = []
        y_lines = []
        for saliency_object in self.saliency_objects:
            ex = saliency_object.get_extremes()
            x_lines.append(ex[0])
            y_lines.append(ex[1])
        x_length = Utils.lines_length(np.array(x_lines))
        y_length = Utils.lines_length(np.array(y_lines))

        x_scale = min((self.target_shape[1] - 1) / x_length, 1.0)
        y_scale = min((self.target_shape[0] - 1) / y_length, 1.0)

        scale = min(x_scale, y_scale)

        for index, saliency_object in enumerate(self.saliency_objects):
            sum = np.array([list(p) for p in saliency_object.triangles]).sum(axis=0)
            center_point = sum / len(saliency_object.triangles)
            for point in saliency_object.triangles:
                pol = Utils.cart2pol(point.x, point.y, center_point)
                point.object_parameter = ObjectParameter(index, pol[0], pol[1], scale)
                print("")

    def __find_other_points(self, points: np.ndarray) -> None:
        """
        Fill until now non-classified points.
        :param points: Input points
        :return: None
        """
        for point in points:
            flag = False
            if (point in self.corner_points) or (point in self.border_points):
                continue
            for saliency_object in self.saliency_objects:
                flag = point in saliency_object.triangles
                if flag:
                    break
            if flag:
                continue
            point.type |= PointType.OTHER
            self.other_points.append(point)

    def get_point_type_array(self) -> np.ndarray:
        types = []
        for point in self.all_points:
            types.append(point.type)
        return np.array(types)

    def __save_all_points(self, points) -> None:
        for point in points:
            self.all_points.append(point)
