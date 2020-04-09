import argparse
import logging
import os.path
from typing import Tuple

import cv2
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.spatial

import Constraints
import Drawing
import Saliency
import Utils
from PatchArray import PatchArray
from PointsClassifier import PointsClassifier


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
    src_shape = src_img_gray.shape
    args_shape = np.array((args.target_height, args.target_width))

    # Do Canny edge detection
    canny_edges = cv2.Canny(src_img_gray, 100, 200)

    # Filter and get points
    p = PatchArray(30, canny_edges)
    p.filter_points()
    points_tuples = p.get_as_ndarray()
    points_list = np.array([list(n) for n in points_tuples])
    points = p.get_as_Points()

    # Delaunay triangulation
    delaunay = scipy.spatial.Delaunay(points_tuples)
    simplices = delaunay.simplices.copy()
    mesh_edge_indices = Utils.get_edges(delaunay)

    # Draw mesh
    mesh_img = src_img.copy()
    Drawing.drawPoints(mesh_img, points)
    Drawing.drawMesh(mesh_img, points, simplices)

    # Get saliency map
    saliency_map = Saliency.get_saliency_map(src_img)

    # DEBUG
    # saliency_map = np.zeros_like(saliency_map)
    # cv2.circle(saliency_map, (300, 175), 125, (255), -1)
    # a = src_img.copy()
    # cv2.imshow("mask", cv2.bitwise_and(a, a, mask=saliency_map))
    # # cv2.waitKey()
    # saliency_map = np.zeros_like(saliency_map)

    # Classify points
    classified_points = PointsClassifier(points, simplices, saliency_map, args_shape)

    # estimation is naive resize
    estimation = points_list * ((args_shape / src_shape)[::-1])
    estimation = estimation.reshape(len(estimation) * 2)

    # Fill attributes for minimization function
    attributes = Constraints.ConstraintAttributes(classified_points.all_points, classified_points.get_scales(),
                                                  src_shape, args_shape, len(classified_points.saliency_objects),
                                                  classified_points.saliency_objects, mesh_edge_indices,
                                                  classified_points.get_saliency_object_indices(),
                                                  classified_points.get_saliency_object_relative_pos(),
                                                  classified_points.get_border_point_indices())

    epsilon_matrix = np.identity(estimation.size) * np.sqrt(np.finfo(float).eps)
    def boundary_jacobian(points: np.ndarray, target_shape: np.ndarray, border_points_indices: np.ndarray, ):
        j = Constraints.boundary_constraint_jac(points, target_shape, border_points_indices, epsilon_matrix)
        j -= Constraints.boundary_constraint_fun(points, target_shape, border_points_indices)
        return j / np.sqrt(np.finfo(float).eps)

    def saliency_jacobian(points: np.ndarray, saliency_objects_indices: np.ndarray, saliency_objects_relative_pos: np.ndarray, obj_count: int):
        j = Constraints.saliency_constraint_jac(points, saliency_objects_indices, saliency_objects_relative_pos, obj_count, epsilon_matrix)
        j -= Constraints.saliency_constraint_fun(points, saliency_objects_indices, saliency_objects_relative_pos, obj_count)
        return j / np.sqrt(np.finfo(float).eps)

    def length_jacobian(points: np.ndarray, edges: np.ndarray, scales: np.ndarray):
        j = Constraints.length_constraint_energy_jac(points, edges, scales, epsilon_matrix)
        j -= Constraints.length_constraint_energy_fun(points, edges, scales)
        return j / np.sqrt(np.finfo(float).eps)

    # Create constraints functions
    constraints = []
    c1 = {'type': 'eq', 'fun': Constraints.boundary_constraint_fun, 'args': (attributes.target_shape, attributes.border_points_indices), 'jac': boundary_jacobian}
    c2 = {'type': 'eq', 'fun': Constraints.saliency_constraint_fun, 'args': (attributes.saliency_objects_indices, attributes.saliency_objects_relative_pos, attributes.obj_count), 'jac': saliency_jacobian}
    constraints.append(c1)
    constraints.append(c2)

    # Minimization options
    options = {'disp': True, 'maxiter': 2000}
    res = scipy.optimize.minimize(Constraints.length_constraint_energy_fun, estimation, args=(attributes.edges, attributes.point_scales),
                                  method='SLSQP', options=options, constraints=constraints, jac=length_jacobian)

    # DEBUG
    ret_b = Constraints.boundary_constraint_fun(res.x, attributes.target_shape, attributes.border_points_indices)
    ret_s = Constraints.saliency_constraint_fun(res.x, attributes.saliency_objects_indices, attributes.saliency_objects_relative_pos, attributes.obj_count)
    ret_l = Constraints.length_constraint_energy_fun(res.x, attributes.edges, attributes.point_scales)
    print("Boundary constraint: {}".format(ret_b))
    print("Saliency constraint: {}".format(ret_s))

    # Reshape points back
    result_points = res.x.reshape((-1, 2))
    estimation = estimation.reshape((-1, 2))

    size = args_shape - 1
    step = args_shape * 1j

    # Map image from src points to copmuted points
    grid_x, grid_y = np.mgrid[0:size[0]:step[0], 0:size[1]:step[1]]
    grid_z = scipy.interpolate.griddata(result_points[:, ::-1], points_list[:, ::-1], (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(args_shape)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(args_shape)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    warped_image = cv2.remap(src_img, map_x_32, map_y_32, cv2.INTER_CUBIC)

    cv2.imshow("src", src_img)
    cv2.imshow("mapped", warped_image)
    # # # cv2.imshow("mesh", mesh_img)
    # # # cv2.imshow("saliency", saliency_map)
    #
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
