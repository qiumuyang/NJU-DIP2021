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
    G = 255
    # split RGB channels
    channels: List[ndarray]
    if img.ndim == 2:
        channels = [img]
    else:
        channels = [img[:, :, i] for i in range(3)]

    for i, ch in enumerate(channels):
        ch = ch.astype(np.int32)
        gray = np.bincount(ch.flatten())
        gray = (np.cumsum(gray / np.sum(gray)) * G).astype(np.int32)
        channels[i] = gray[ch]
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
