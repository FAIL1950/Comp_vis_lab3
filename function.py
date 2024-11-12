from pickletools import uint8

import cv2
import numpy as np
from collections import namedtuple
import random

def convert_to_grayscale(image_path: str)-> np.ndarray:
    image: np.ndarray = cv2.imread(image_path)

    if image is None:
        raise ValueError("Помилка: не вийшло завантажити зображення.")

    res_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return res_image

def checkThreshold(threshold: int):
    if threshold < 0 or threshold > 255:
        raise ValueError("Помилка: значення порогу виходить за межі (0 <= T <=255).")

def checkIsImageGray(img: np.ndarray):
    if len(img.shape) == 3:
        raise ValueError("Помилка: вхідне зображення має бути у відтінках сірого.")

def binarizationWithThresholdRaw(img: np.ndarray, threshold: int) -> np.ndarray:
    res = img.copy()
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            if pixel < threshold:
                res[y, x] = 0
            else:
                res[y, x] = 255
    return res


def binarizationWithThreshold(img: np.ndarray, threshold: int) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        checkThreshold(threshold)
        return binarizationWithThresholdRaw(img, threshold)
    except ValueError as e:
        print(e)
        return None

def getHistogramRaw(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            hist[pixel] += 1
    return hist


def getHistogram(img: np.ndarray) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        return getHistogramRaw(img)
    except ValueError as e:
        print(e)
        return None

def getOmegaRaw(img: np.ndarray):
    return img.shape[0] * img.shape[1]


def getRelativeHistogramRaw(img: np.ndarray) -> np.ndarray:
    hist = getHistogramRaw(img)
    omega = getOmegaRaw(img)
    relHist = np.zeros(256)
    for i in range(256):
        relHist[i] = hist[i] / omega
    return relHist


def getRelativeHistogram(img: np.ndarray) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        return getRelativeHistogramRaw(img)
    except ValueError as e:
        print(e)
        return None


def OtsuBinarization(img: np.ndarray, t0: int) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        if t0 < 0 or t0 > 255:
            raise ValueError("Помилка: значення приросту виходить за межі (0 <= T <=255).")
        hist = getHistogramRaw(img)
        relHist = getRelativeHistogramRaw(img)
        gmax = 255
        u = t0
        t = u
        smax = 0
        while u < gmax:
            ci = 0
            tmpSum1 = 0
            tmpSum2 = 0
            for v in range(u+1):
                ci += relHist[v]
                tmpSum1 += v*hist[v]
                tmpSum2 += hist[v]
            m1 = tmpSum1 / tmpSum2
            tmpSum1 = 0
            tmpSum2 = 0
            for v in range(u + 1, gmax+1):
                tmpSum1 += v * hist[v]
                tmpSum2 += hist[v]
            m2 = tmpSum1 / tmpSum2
            disp = ci * (1-ci)*(m1 - m2)**2
            if disp > smax:
                smax = disp
                t = u
            u += t0
        return binarizationWithThresholdRaw(img, t)
    except ValueError as e:
        print(e)
        return None

def XDoGRaw(img: np.ndarray, sigma1, sigma2, k, epsilon, phi) -> np.ndarray :
    g1 = cv2.GaussianBlur(img, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma2)

    dog = g1 - k * g2
    xdog = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
    xdog = np.clip(xdog * 255, 0, 255).astype(np.uint8)
    return xdog

def XDoG(img: np.ndarray, sigma1=0.5, sigma2=1.5, k=1.2, epsilon=50, phi=0.01) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        return XDoGRaw(img, sigma1, sigma2, k, epsilon, phi)
    except ValueError as e:
        print(e)
        return None

def XDoGBinarization(img: np.ndarray, sigma1=0.5, sigma2=1.5, k=1.2, epsilon=50, phi=0.01) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        xdog = XDoGRaw(img, sigma1, sigma2, k, epsilon, phi)
        res = np.where(xdog < 128, 0, 255).astype(np.uint8)
        return res
    except ValueError as e:
        print(e)
        return None

def getInterval(pixel):
    intervals = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 255)]
    for interval in intervals:
        if pixel >= interval[0] and pixel <= interval[1]:
            return interval


def generate_unique_color(existing_colors):
    while True:
        color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

        if not any(np.array_equal(color, existing_color) for existing_color in existing_colors):
            existing_colors.append(color)
            return color

def SRG(img: np.ndarray) -> np.ndarray | None:
    try:
        checkIsImageGray(img)
        height, width = img.shape
        control_array = np.zeros((height, width), dtype=bool)
        segment = namedtuple("Segment", ["interval", "points"])
        segments = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        activeInterval = None
        tmpPoints = []
        stack = []
        for y in range(height):
            for x in range(width):
                if not control_array[y, x]:
                    tmpPoints = []
                    control_array[y, x] = True
                    stack.append((y, x))
                    tmpPoints.append((y, x))
                    activeInterval = getInterval(img[y, x])
                    while stack:
                        py, px = stack.pop()
                        for dx, dy in directions:
                            nx, ny = px + dx, py + dy

                            if 0 <= ny < height and 0 <= nx < width:
                                pixel = img[ny, nx]
                                if pixel >= activeInterval[0] and pixel <= activeInterval[1]:
                                    if not control_array[ny, nx]:
                                        control_array[ny, nx] = True
                                        stack.append((ny, nx))
                                        tmpPoints.append((ny, nx))
                    segments.append(segment(interval=activeInterval, points=tmpPoints))

        image_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        generated_colors = []
        for seg in segments:
            color = generate_unique_color(generated_colors)
            for point in seg.points:
                y, x = point
                image_color[y, x] = color  # Color the segment points in the output image

        return image_color
    except ValueError as e:
        print(e)
        return None






