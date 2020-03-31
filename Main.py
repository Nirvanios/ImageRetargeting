import argparse
import logging
import os.path

import cv2
import numpy as np
import scipy.spatial

import Drawing
import Saliency
from PatchArray import PatchArray
from PointsClassifier import PointsClassifier
import Constraints


def border_keypoints(img: np.ndarray, distance: int = 20):
    shape = img.shape
    img[0, 0] = img[shape[0] - 1, 0] = img[0, shape[1] - 1] = img[shape[0] - 1, shape[1] - 1] = 255
    for x in range(0, shape[1], distance):
        img[0, x] = 255
        img[shape[0] - 1, x] = 255

    for y in range(0, shape[0], distance):
        img[y, 0] = 255
        img[y, shape[1] - 1] = 255


def main(args):
    # Load image
    if not os.path.exists(args.src_img):
        raise FileNotFoundError("File \"" + args.src_img + "\" not found.")
    src_img = cv2.imread(args.src_img)
    src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    args_shape = (args.target_height, args.target_width)

    # Do Canny edge detection
    edges = cv2.Canny(src_img_gray, 100, 200)

    # Filter and get points
    p = PatchArray(30, edges)
    p.filter_points()
    points_tuples = p.get_as_ndarray()
    points_list = np.array([list(n) for n in points_tuples])
    points = p.get_as_Points()

    # Delaunay triangulation
    tri = scipy.spatial.Delaunay(points_tuples)
    simplices = tri.simplices.copy()

    # Draw mesh
    mesh_img = src_img.copy()
    Drawing.drawPoints(mesh_img, points)
    Drawing.drawMesh(mesh_img, points, simplices)

    # Get saliency map
    saliency_map = Saliency.get_saliency_map(src_img)
    saliency_map = np.zeros_like(saliency_map)
    cv2.circle(saliency_map, (100, 100), 75, (255), -1)

    points_debug = np.array([[1, 2], [5, 8], [8, 6], [9, 5]])
    simplices_debug = np.array([[0, 1, 2], [1, 2, 3]])

    classified_points = PointsClassifier(points, simplices, saliency_map, args_shape)

    points_list = points_list.reshape(len(points_list) * 2)

    ret = Constraints.boundary_constraint_fun(points_list, classified_points.get_point_type_array(), args_shape)
    ret = Constraints.saliency_constraint_fun(points_list, classified_points.all_points)
    points_list = points_list.reshape((-1, 2))

    saliency_map = np.zeros_like(src_img)
    cv2.circle(saliency_map, (100, 100), 75, (255,255,255), -1)
    Drawing.drawPoints(saliency_map, points)
    Drawing.drawMesh(saliency_map, points, simplices)
    Drawing.drawPoints(saliency_map, classified_points.saliency_objects[0].triangles, (255,0,0))

    cv2.imshow("mesh", mesh_img)
    cv2.imshow("saliency", saliency_map)

    cv2.waitKey()


parser = argparse.ArgumentParser(description="Image Retargeting using mesh parametrization")
parser.add_argument('-in',
                    action='store',
                    dest='src_img',
                    help='Input image',
                    required=True,
                    type=str)
parser.add_argument('-W',
                    action='store',
                    dest='target_width',
                    help='Output image width',
                    required=True,
                    type=int)
parser.add_argument('-H',
                    action='store',
                    dest='target_height',
                    help='Output image height',
                    required=True,
                    type=int)

if __name__ == "__main__":
    # setup logger
    logger = logging.getLogger('simple logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Call main function
    try:
        main(parser.parse_args())
    except Exception as e:
        logger.exception(e)
