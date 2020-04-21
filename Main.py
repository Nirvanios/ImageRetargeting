import argparse
import logging
import math
import os.path
from typing import Any

import cv2
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.spatial

import Constraints
import Drawing
import Utils
from Line import Line
from PatchArray import PatchArray
from PointsClassifier import PointsClassifier
from Saliency import Saliency

counter: int = 0

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

    def select_lines(event: int, x: int, y: int, flags: int, param: Any):
        if event == cv2.EVENT_LBUTTONUP:
            lines, line_img = param
            lines.append((x, src_shape[0] - y))
            if len(lines) % 2 == 0:
                cv2.line(line_img, (lines[-1][0], src_shape[0] - lines[-1][1]),
                         (lines[-2][0], src_shape[0] - lines[-2][1]), (0, 255, 0), 2)
                cv2.imshow("Select lines", line_img)

    line_img = np.copy(src_img)
    cv2.imshow("Select lines", line_img)
    point_lines = []
    cv2.setMouseCallback("Select lines", select_lines, [point_lines, line_img])
    key = cv2.waitKey()
    while key != 13:
        key = cv2.waitKey()
    cv2.destroyWindow("Select lines")

    lines = []
    for i in range(0, len(point_lines), 2):
        lines.append(Line(point_lines[i], point_lines[i + 1]))

    # Filter and get points
    p = PatchArray(math.ceil(np.sqrt((src_shape[0] * src_shape[1]) / 300)), canny_edges, lines)
    p.filter_points()
    points_tuples = p.get_as_ndarray()
    points_list = np.array([list(n) for n in points_tuples])
    points = p.get_as_Points()

    # Delaunay triangulation
    delaunay = scipy.spatial.Delaunay(points_tuples)
    simplices = delaunay.simplices.copy()
    mesh_edge_indices = Utils.get_edges(delaunay)
    mesh_edge_neighbour_indices = Utils.get_edge_neighbours(delaunay, mesh_edge_indices, points_list)
    original_orientations = Utils.compute_orientations(points_list, mesh_edge_neighbour_indices).reshape((-1, 2))

    # Get saliency map
    saliency_map = Saliency(src_img, False).get_saliency_map()

    # Classify points
    classified_points = PointsClassifier(points, simplices, saliency_map, args_shape)

    # Line indices and coefficients
    line_coefficients = np.array([[line.a, line.b] for line in lines])
    coefficient_indices = np.array([lines.index(l_point.line_eq) for l_point in classified_points.edge_points])
    line_point_indices = np.array(
        [classified_points.all_points.index(l_point) for l_point in classified_points.edge_points])

    # Draw mesh
    mesh_img = src_img.copy()
    Drawing.drawPoints(mesh_img, points)
    Drawing.drawMesh(mesh_img, points, simplices)
    cv2.imshow("mesh", mesh_img)
    cv2.waitKey()

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

    def saliency_jacobian(points: np.ndarray, saliency_objects_indices: np.ndarray,
                          saliency_objects_relative_pos: np.ndarray, obj_count: int):
        j = Constraints.saliency_constraint_jac(points, saliency_objects_indices, saliency_objects_relative_pos,
                                                obj_count, epsilon_matrix)
        j -= Constraints.saliency_constraint_fun(points, saliency_objects_indices, saliency_objects_relative_pos,
                                                 obj_count)
        return j / np.sqrt(np.finfo(float).eps)

    def length_jacobian(points: np.ndarray, edges: np.ndarray, scales: np.ndarray, original_orientations: np.ndarray,
                        mesh_edge_neighbour_indices: np.ndarray, line_coefficients: np.ndarray,
                        coefficient_indices: np.ndarray, line_point_indices: np.ndarray, src_shape: np.ndarray):
        j = Constraints.length_structure_energy_wraper_jac(points, edges, scales, original_orientations,
                                                           mesh_edge_neighbour_indices, line_coefficients,
                                                           coefficient_indices, line_point_indices, src_shape, epsilon_matrix)
        j -= Constraints.length_structure_energy_wraper(points, edges, scales, original_orientations,
                                                        mesh_edge_neighbour_indices, line_coefficients,
                                                        coefficient_indices, line_point_indices, src_shape)
        return j / np.sqrt(np.finfo(float).eps)

    def counter_callback(xk: np.ndarray) -> bool:
        global counter
        counter += 1
        print("{}%".format(round((counter*100)/2000, 1)))
        return False

    # Create constraints functions
    constraints = []
    c1 = {'type': 'eq', 'fun': Constraints.boundary_constraint_fun,
          'args': (attributes.target_shape, attributes.border_points_indices), 'jac': boundary_jacobian}
    c2 = {'type': 'eq', 'fun': Constraints.saliency_constraint_fun,
          'args': (attributes.saliency_objects_indices, attributes.saliency_objects_relative_pos, attributes.obj_count),
          'jac': saliency_jacobian}
    constraints.append(c1)
    constraints.append(c2)

    # Minimization options
    options = {'disp': True, 'maxiter': 2000}
    minimize = True
    ttt = 1
    while minimize:
        res = scipy.optimize.minimize(Constraints.length_structure_energy_wraper, estimation, args=(
            attributes.edges, attributes.point_scales, original_orientations, mesh_edge_neighbour_indices,
            line_coefficients, coefficient_indices, line_point_indices, np.array(src_shape)), method='SLSQP', options=options,
                                      constraints=constraints, jac=length_jacobian, callback=counter_callback)

        # DEBUG
        ret_b = Constraints.boundary_constraint_fun(res.x, attributes.target_shape, attributes.border_points_indices)
        ret_s = Constraints.saliency_constraint_fun(res.x, attributes.saliency_objects_indices,
                                                    attributes.saliency_objects_relative_pos, attributes.obj_count)
        ret_l = Constraints.length_constraint_energy_fun(res.x, attributes.edges, attributes.point_scales,
                                                         original_orientations, mesh_edge_neighbour_indices)
        print("Boundary constraint: {}".format(ret_b))
        print("Saliency constraint: {}".format(ret_s))

        # Reshape points back
        result_points = res.x.reshape((-1, 2))
        estimation = estimation.reshape((-1, 2))

        size = args_shape - 1
        step = args_shape * 1j

        # Map image from src points to copmuted points
        grid_x, grid_y = np.mgrid[0:size[0]:step[0], 0:size[1]:step[1]]
        grid_z = scipy.interpolate.griddata(result_points[:, ::-1], points_list[:, ::-1], (grid_x, grid_y),
                                            method='linear')
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(args_shape)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(args_shape)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')
        warped_image = cv2.remap(src_img, map_x_32, map_y_32, cv2.INTER_CUBIC)

        cv2.imshow("src", src_img)
        cv2.imshow("mapped", warped_image)
        cv2.imshow("mesh", mesh_img)
        # # # cv2.imshow("saliency", saliency_map)
        #
        key = cv2.waitKey(2)
        key = 1
        if key == 27 or ttt == 1:  # ESC TODO
            minimize = False
        else:
            print("Next n iteraions...")
            minimize = True
            estimation = result_points
            ttt += 1
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
