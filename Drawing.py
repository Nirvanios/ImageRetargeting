import cv2
import numpy as np
from typing import Tuple


def drawPoints(img: np.ndarray, points: np.ndarray, color: Tuple = (0,255,255)):
    """
    Draws points into image
    :param img: Image to draw into
    :param points: Points to draw
    :return: None
    """
    for point in points:
        cv2.circle(img, tuple(point), 3, color, -1)


def drawMesh(img: np.ndarray, points: np.ndarray, simplices: np.ndarray):
    """
    Draws mesh into image
    :param img: Image to draw into
    :param points: Vertices of mesh
    :param simplices: Indices of mesh
    :return: None
    """
    for poly in simplices:
        cv2.line(img, tuple(points[poly[0]]), tuple(points[poly[1]]), (0, 0, 255))
        cv2.line(img, tuple(points[poly[1]]), tuple(points[poly[2]]), (0, 0, 255))
        cv2.line(img, tuple(points[poly[2]]), tuple(points[poly[0]]), (0, 0, 255))
