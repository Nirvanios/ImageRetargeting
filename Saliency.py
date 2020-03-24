import cv2
import numpy as np


def get_saliency_map(img: np.ndarray) -> np.ndarray:
    """
    :param img: source image
    :return: binary saliency map
    """
    s = cv2.saliency.StaticSaliencyFineGrained_create()
    (result, saliency_map) = s.computeSaliency(img)
    saliency_map = (saliency_map * 255).astype("uint8")
    return cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
