import argparse
import logging
import os.path

import cv2
import numpy as np
import scipy.spatial

from PatchArray import PatchArray
import Drawing
import Saliency

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
    #Load image
    if not os.path.exists(args.src_img):
        raise FileNotFoundError("File \"" + args.src_img + "\" not found.")
    src_img = cv2.imread(args.src_img)
    src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    #Do Canny edge detection
    edges = cv2.Canny(src_img_gray, 100, 200)

    #Filter and get points
    p = PatchArray(30, edges)
    p.filter_points()
    points = p.get_as_ndarray()

    #Delaunay triangulation
    tri = scipy.spatial.Delaunay(points)
    simplices = tri.simplices.copy()

    #Draw mesh
    Drawing.drawPoints(src_img, points)
    Drawing.drawMesh(src_img, points, simplices)

    #Get saliency map
    saliency_map = Saliency.get_saliency_map(src_img)

    cv2.imshow("points", src_img)

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
    #setup logger
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

