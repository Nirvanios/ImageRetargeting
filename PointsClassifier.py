import cv2
import numpy as np

import Utils
from SaliencyObject import SaliencyObject


class PointsClassifier:

    def __init__(self, points: np.ndarray, simplices: np.ndarray, saliency_map: np.ndarray):
        """
        Fills apropriate members
        :param points: Detected points in image
        :param simplices: Indices to mesh
        :param saliency_map: Binary importance map
        """
        self.shape = saliency_map.shape
        self.corner_points = []
        self.border_points = []
        self.saliency_objects = []
        self.edge_points = []
        self.other_points = []

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
                    self.corner_points.append(point)
                else:
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
            self.other_points.append(point)
