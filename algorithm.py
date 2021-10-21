from typing import List, Callable
from functools import wraps
import numpy as np
from numpy import ndarray


def split_channel(func: Callable[[ndarray], ndarray]):
    # check the input image
    # assume decorated functions only accept gray-scale image
    # if the input has several channels,
    # separately call decorated function on each channel
    # and merge the result at the end.

    @wraps(func)
    def wrapper(*args):
        img: ndarray = args[0]
        if img.ndim == 2:    # gray-scale
            return func(img)
        elif img.ndim == 3:  # rgb
            channels = [func(img[:, :, i]) for i in range(3)]
            return np.dstack(channels)
        else:
            raise ValueError('invalid argument ndim:', img.ndim)

    return wrapper


def plane(img: ndarray) -> ndarray:
    def kth_plane(img: ndarray, k: int) -> ndarray:
        return np.where(img & (1 << k), 1, 0)

    # prepare each bit-plane (9th is empty)
    _planes: List[ndarray] = [kth_plane(img, i) for i in reversed(range(8))]
    _planes.append(np.ones(img.shape))

    # horizontally stack (group by 3)
    _stacked: List[ndarray] = [np.hstack(_planes[i*3:i*3+3]) for i in range(3)]
    return np.vstack(_stacked)


@split_channel
def equalize(img: ndarray) -> ndarray:
    G = 256

    # image as float -> uint8
    if img.dtype in [np.float16, np.float32, np.float64]:
        img = (img * (G - 1)).astype(np.uint8)

    # count each gray level [0, G)
    gray: ndarray = np.bincount(img.flatten())
    if gray.shape[0] < G:
        gray = np.pad(gray, (0, G - gray.shape[0]))

    # calculate new gray level
    gray = np.cumsum(gray / np.sum(gray))
    return gray[img]


@split_channel
def denoise(img: ndarray) -> ndarray:
    def median_filter_slow(img: ndarray, k: int) -> ndarray:
        if k % 2 == 0:
            k += 1
        p = k // 2
        ret = img.copy()
        h, w = img.shape
        for i in range(p, w - p):
            for j in range(p, h - p):
                ret[j, i] = np.median(img[j - p:j + p, i - p:i + p])
        return ret

    def median_filter_fast(img: ndarray, k: int) -> ndarray:
        if k % 2 == 0:
            k += 1
        p = k // 2
        h, w = img.shape

        # extend the image to handle margin cases
        row_ext = img.copy()
        for i in range(p):
            row_ext = np.insert(row_ext, 0, img[i + 1, :], axis=0)
        ext = row_ext.copy()
        for i in range(p):
            ext = np.insert(ext, 0, row_ext[:, i + 1], axis=1)

        # trick: put k*k neighbors of a pixel into the array
        tmp = np.dstack([np.roll(ext, -i, axis=1) for i in range(k)])
        ret = np.dstack([np.roll(tmp, -i, axis=0) for i in range(k)])
        ret = np.median(ret, axis=2)[:h, :w]
        return ret

    if img.dtype == np.uint8:
        img = img / 255
    return median_filter_fast(img, 5)


@split_channel
def interpolate(img: ndarray) -> ndarray:
    def bilinear(img: ndarray, kw: float, kh: float) -> ndarray:
        # prepare the param matrix (shape [h, w, 4])
        r10 = np.roll(img, -1, axis=1)
        r01 = np.roll(img, -1, axis=0)
        r11 = np.roll(r01, -1, axis=1)
        # f(1, 0) - f(0, 0)
        p0 = r10 - img
        # f(0, 1) - f(0, 0)
        p1 = r01 - img
        # f(1, 1) + f(0, 0) - f(0, 1) - f(1, 0)
        p2 = r11 + img - r01 - r10
        # f(x, y) = p0 * x + p1 * y + p2 * x * y + f(0, 0)
        params = np.dstack([p0, p1, p2, img])  # p3 = f(0, 0)

        # get coordinates mapping to the origin image
        h, w = img.shape
        hh, ww = round(h * kh), round(w * kw)
        xs = np.repeat(np.array([range(ww)]), hh, axis=0) / kw
        ys = np.repeat(np.array([range(hh)]), ww, axis=0).transpose() / kh
        xy: ndarray = np.dstack([xs, ys])
        pivots = xy.astype(np.uint32)
        xy -= pivots

        # get each coordinate's params using pivots (shape [h, w, 2])
        selected_params = params[pivots[:, :, 1], pivots[:, :, 0]]

        # calculate coordinate for multiply
        # (shape [h, w, 1, 4] @ shape [h, w, 4, 1] => shape[h, w, 1, 1])
        xyxy: ndarray = np.dstack(
            [xy[:, :, 0], xy[:, :, 1], xy[:, :, 0] * xy[:, :, 1], np.ones((hh, ww))])
        ret = np.matmul(selected_params.reshape(hh, ww, 1, 4),
                        xyxy.reshape(hh, ww, 4, 1))
        ret = ret.reshape(hh, ww)
        return ret

    if img.dtype == np.uint8:
        img = img / 255
    return bilinear(img, 2, 2)


def dft(img: ndarray) -> ndarray:
    return None


def butterworth(img: ndarray) -> ndarray:
    return None


def canny(img: ndarray) -> ndarray:
    return None


def morphology(img: ndarray) -> ndarray:
    return None
