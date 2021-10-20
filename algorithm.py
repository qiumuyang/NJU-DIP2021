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


def denoise(img: ndarray) -> ndarray:
    return None


def interpolate(img: ndarray) -> ndarray:
    return None


def dft(img: ndarray) -> ndarray:
    return None


def butterworth(img: ndarray) -> ndarray:
    return None


def canny(img: ndarray) -> ndarray:
    return None


def morphology(img: ndarray) -> ndarray:
    return None
