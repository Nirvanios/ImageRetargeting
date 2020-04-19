from typing import Any

import cv2
import numpy as np

class Saliency:

    __PLUS_KEY = 43
    __MINUS_KEY = 45
    __ENTER_KEY = 13

    def __init__(self, img: np.ndarray, auto: bool = True):
        self.saliency_map = np.zeros_like(img)
        self.img = np.copy(img)
        self.auto = auto
        self.mouse_down = False
        self.circle_radius = 3
        self.compute_saliency()

    def compute_saliency(self):
        if not self.auto:
            cv2.imshow("Select important object", cv2.addWeighted(self.saliency_map, 0.4, self.img, 1 - 0.4, 0))
            cv2.setMouseCallback("Select important object", self.__select_region, self)
            key = cv2.waitKey()
            while key != self.__ENTER_KEY:
                if key == self.__MINUS_KEY:
                    self.circle_radius = max(1, self.circle_radius - 1)
                elif key == self.__PLUS_KEY:
                    self.circle_radius = min(25, self.circle_radius + 1)
                key = cv2.waitKey()
        else:
            s = cv2.saliency.StaticSaliencyFineGrained_create()
            (result, saliency_map) = s.computeSaliency(self.img)
            saliency_map = (saliency_map * 255).astype("uint8")
            saliency_map = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            kernel = np.ones((7, 7), np.uint8)
            self.saliency_map = cv2.morphologyEx(saliency_map, cv2.MORPH_OPEN, kernel)

    def get_saliency_map(self) -> np.ndarray:
        return self.saliency_map[:,:,2]

    @staticmethod
    def __select_region(event: int, x: int, y: int, flags: int, param: Any):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.mouse_down = True
            cv2.circle(param.saliency_map, (x,y), param.circle_radius, (0,0,255), -1)
            cv2.imshow("Select important object", cv2.addWeighted(param.saliency_map, 0.4, param.img, 1 - 0.4, 0))
        elif event == cv2.EVENT_LBUTTONUP:
            param.mouse_down = False
        elif event == cv2.EVENT_MOUSEMOVE and param.mouse_down:
            cv2.circle(param.saliency_map, (x,y), param.circle_radius, (0,0,255), -1)
            cv2.imshow("Select important object", cv2.addWeighted(param.saliency_map, 0.4, param.img, 1 - 0.4, 0))
        elif event == cv2.EVENT_MOUSEWHEEL:
            a =1+1
            pass


