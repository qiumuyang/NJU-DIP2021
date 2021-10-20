from typing import List
import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    def kth_plane(img: ndarray, k: int) -> ndarray:
        return np.where(img & (1 << k), 1, 0)

    # prepare each bit-plane (9th is empty)
    _planes: List[ndarray] = [kth_plane(img, i) for i in reversed(range(8))]
    _planes.append(np.ones(img.shape))

    # horizontally stack (group by 3)
    _stacked: List[ndarray] = [np.hstack(_planes[i*3:i*3+3]) for i in range(3)]
    return np.vstack(_stacked)


def equalize(img: ndarray) -> ndarray:
    G = 256

    # split RGB channels
    channels: List[ndarray]
    if img.ndim == 2:
        channels = [img]
    else:
        channels = [img[:, :, i] for i in range(3)]

    for i, channel in enumerate(channels):
        # image as float -> uint8
        if channel.dtype in [np.float16, np.float32, np.float64]:
            channel = (channel * (G - 1)).astype(np.uint8)

        # count each gray level [0, 256)
        gray: ndarray = np.bincount(channel.flatten())
        if gray.shape[0] < G:
            gray = np.pad(gray, (0, G - gray.shape[0]))

        # calculate new gray level
        gray = np.cumsum(gray / np.sum(gray))
        channels[i] = gray[channel]
    return np.dstack(channels) if img.ndim == 3 else channels[0]


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
