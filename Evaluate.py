import copy

import cv2
import numpy as np


class Evaluate:

    def evaluate(self, path):
        merge = cv2.imread(path + '/merge.jpg', 0)
        h, w = np.shape(merge)
        contours, hierarchy = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoursList = list(copy.copy(contours))
        areas = []
        lengths = []
        for contour in contoursList:
            areas.append(int(cv2.contourArea(contour)))
            lengths.append(int(cv2.arcLength(contour, True) / 2))

        areaTotal = sum(areas)
        areaTotalRitio = round(areaTotal / (h * w), 2) * 100

        maxArea = max(areas)
        maxAreaRitio = round(maxArea / (h * w), 2) * 100

        maxLength = max(lengths)
        maxLengthRitio = round(maxLength / ((h + w) * 6), 2) * 100

        grade = 100

        return [areaTotalRitio, maxAreaRitio, maxLengthRitio, grade]
