from typing import List, Callable
from functools import wraps
import numpy as np
from numpy import ndarray
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def split_channel(func: Callable[[ndarray], ndarray]):
    ''' Split channels of the input image.

    Assume the decorated function only accept gray-scale image.

    If the input has more than one channels, separately call decorated function on each channel and merge the result at the end.
    '''

    @wraps(func)
    def wrapper(*args):
        img: ndarray = args[0]
        if img.ndim == 2:    # gray-scale
            return func(img)
        elif img.ndim == 3:  # rgb
            channels = [func(img[:, :, i]) for i in range(3)]
            return np.dstack(channels)
        else:
            raise ValueError('invalid image ndim:', img.ndim)

    return wrapper


def position_index(arr: ndarray, idx: ndarray) -> ndarray:
    ''' arr: shape=(L, d)
        idx: shape=(L,) idx in [0, d)
        return: arr[idx] shape=(L,)
    '''
    onehot = np.eye(arr.shape[1], dtype='int')[idx]
    return (arr * onehot).sum(axis=1)


class Neighbors:
    ''' Get k*k neighbors for each item in the input array.

    '''

    def __init__(self, arr: ndarray, k: int) -> None:
        assert(k % 2 == 1)
        assert(arr.ndim == 2)

        self.k = k

        p = k // 2
        h, w = arr.shape
        xs = np.repeat(np.arange(w), h).reshape(w, h).transpose()
        ys = np.repeat(np.arange(h), w).reshape(h, w)

        tmp = []
        for y in range(-p, p + 1):
            for x in range(-p, p + 1):
                xs_offset = xs + np.ones(arr.shape, dtype='int') * x
                ys_offset = ys + np.ones(arr.shape, dtype='int') * y
                xs_offset[xs_offset < 0] = 0
                xs_offset[xs_offset >= w] = w - 1
                ys_offset[ys_offset < 0] = 0
                ys_offset[ys_offset >= h] = h - 1
                tmp.append(arr[ys_offset, xs_offset])

        self.neighbors: ndarray = np.dstack(tmp)

    def at(self, x: int, y: int) -> ndarray:
        y += self.k // 2
        x += self.k // 2
        return self.neighbors[:, :, y * self.k + x]


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
    level = 256
    img = (img * (level - 1)).astype(np.uint)

    hist, bins = np.histogram(img, level, (0, level))

    # calculate new gray level
    gray = np.cumsum(hist) / np.sum(hist)

    return gray[img]


@split_channel
def denoise(img: ndarray) -> ndarray:
    def median_filter(img: ndarray, k: int) -> ndarray:
        return np.median(Neighbors(img, k).neighbors, axis=2)

    def adaptive_median_filter(img: ndarray, k1: int, k2: int) -> ndarray:
        h, w = img.shape
        mult_neighbor = [Neighbors(img, i).neighbors
                         for i in range(k1, k2 + 1, 2)]

        # median / max / min: shape=((k2 - k1) // 2, h * w)
        medians: ndarray = np.array([np.median(n, axis=2).flatten()
                                    for n in mult_neighbor])
        maxs: ndarray = np.array([np.max(n, axis=2).flatten()
                                  for n in mult_neighbor])
        mins: ndarray = np.array([np.min(n, axis=2).flatten()
                                  for n in mult_neighbor])

        # Calculate suitable window size foreach pixel, default largest size
        index = np.ones(h * w, 'int') * (len(medians) - 1)

        for i in range(len(medians) - 1):
            index = np.where(
                # median is not noisy
                (medians[i] < maxs[i]) & (medians[i] > mins[i])
                # need update
                & (i < index),
                i, index)

        img_flatten = img.flatten()  # h * w
        max_selected = position_index(maxs.transpose(), index)
        min_selected = position_index(mins.transpose(), index)
        med_selected = position_index(medians.transpose(), index)

        ret = np.where(
            # this pixel is not noisy
            (img_flatten < max_selected) & (img_flatten > min_selected),
            img_flatten,  # keep
            med_selected  # replaced by median
        )
        return ret.reshape(h, w)

    if img.dtype == np.uint8:
        img = img / 255
    return median_filter(img, 3)

    # return adaptive_median_filter(img, 3, 9) # seems not working


@split_channel
def interpolate(img: ndarray) -> ndarray:
    def bilinear(img: ndarray, kw: float, kh: float) -> ndarray:
        # prepare the param matrix (shape [h, w, 4])
        neighbors = Neighbors(img, 3)
        # p0 = f(1, 0) - f(0, 0)
        p0 = neighbors.at(1, 0) - img
        # p1 = f(0, 1) - f(0, 0)
        p1 = neighbors.at(0, 1) - img
        # p2 = f(1, 1) + f(0, 0) - f(0, 1) - f(1, 0)
        p2 = neighbors.at(1, 1) + img - neighbors.at(0, 1) - neighbors.at(1, 0)
        # p3 = f(0, 0)
        p3 = img
        # f(x, y) = p0 * x + p1 * y + p2 * x * y + f(0, 0)
        #         = [p0, p1, p2, p3] \mul [x, y, xy, 1]
        #         x \in [0, 1), y \in [0, 1)
        params = np.dstack([p0, p1, p2, p3])

        # map the pixels' position of the target image to the origin
        h, w = img.shape
        hh, ww = round(h * kh), round(w * kw)
        xs = np.repeat(np.array([range(ww)]), hh, axis=0) / kw
        ys = np.repeat(np.array([range(hh)]), ww, axis=0).transpose() / kh
        xs_int = xs.astype('int')
        ys_int = ys.astype('int')
        xs -= xs_int
        ys -= ys_int

        # prepare the matrix
        target_mat_1: ndarray = params[ys_int, xs_int].reshape(hh, ww, 1, 4)
        target_mat_2: ndarray = np.dstack((xs, ys, xs * ys, np.ones((hh, ww)))) \
                                  .reshape(hh, ww, 4, 1)

        # shape [h, w, 4, 1] \mul shape [h, w, 4, 1] => shape[h, w, 1, 1]
        ret = np.matmul(target_mat_1, target_mat_2).reshape(hh, ww)
        return ret

    if img.dtype == np.uint8:
        img = img / 255
    return bilinear(img, 2, 2)


def dft(img: ndarray) -> ndarray:
    f = fftshift(fft2(img))
    f = np.log(np.abs(f))
    return f


def butterworth(img: ndarray) -> ndarray:
    def _butterworth(D_0: float, k: int = 1):
        freq = fftshift(fft2(img))
        h, w = freq.shape
        xs = np.repeat(np.array([range(w)]), h, axis=0)
        ys = np.repeat(np.array([range(h)]), w, axis=0).transpose()
        xs = np.power((xs - w / 2), 2)
        ys = np.power((ys - h / 2), 2)
        D_uv = np.power(xs + ys, 0.5)
        H_uv = 1 / (1 + np.power(D_uv / D_0, 2 * k))
        freq *= H_uv
        return np.abs(ifft2(ifftshift(freq)))

    return _butterworth(70, 1)


def canny(img: ndarray) -> ndarray:
    # Gauss Filter
    gauss = np.array([2, 4, 5, 4, 2,
                      4, 9, 12, 9, 4,
                      5, 12, 15, 12, 5,
                      4, 9, 12, 9, 4,
                      2, 4, 5, 4, 2]) / 159
    img_neighbors = Neighbors(img, 5)
    img = np.matmul(img_neighbors.neighbors, gauss)

    # Calculate gradient
    img_neighbors = Neighbors(img, 3)
    grad_x = (img_neighbors.at(1, 0) - img_neighbors.at(-1, 0)) / 2
    grad_y = (img_neighbors.at(0, 1) - img_neighbors.at(0, -1)) / 2
    grad: ndarray = np.power(grad_x * grad_x + grad_y * grad_y, 0.5)

    # Non-Maximum Suppression
    # here we get gradient by interpolation (instead of approximation)
    epsilon = np.finfo(np.float64).eps
    grad_x_fix = np.where(grad_x == 0, epsilon, grad_x)
    grad_y_fix = np.where(grad_y == 0, epsilon, grad_y)
    grad_k: ndarray = grad_y / grad_x_fix
    grad_k_abs = np.abs(grad_k)
    grad_t_abs = np.abs(grad_x / grad_y_fix)

    grad_neighbors = Neighbors(grad, 3)

    descriminate = [
        (
            grad_k >= 1, grad_t_abs,
            grad_neighbors.at(1, 1), grad_neighbors.at(0, 1),
            grad_neighbors.at(-1, -1), grad_neighbors.at(0, -1),
        ),
        (
            grad_k <= -1, grad_t_abs,
            grad_neighbors.at(-1, 1), grad_neighbors.at(0, 1),
            grad_neighbors.at(1, -1), grad_neighbors.at(0, -1),
        ),
        (
            (grad_k < 1) & (grad_k >= 0), grad_k_abs,
            grad_neighbors.at(1, 1), grad_neighbors.at(1, 0),
            grad_neighbors.at(-1, -1), grad_neighbors.at(-1, 0),
        ),
        (
            (grad_k > -1) & (grad_k < 0), grad_k_abs,
            grad_neighbors.at(-1, 1), grad_neighbors.at(-1, 0),
            grad_neighbors.at(1, -1), grad_neighbors.at(1, 0),
        )
    ]

    d1 = np.zeros(img.shape)
    d2 = np.zeros(img.shape)
    for cond, coef, p0, p1, p2, p3 in descriminate:
        d1 = np.where(cond, coef * p0 + (np.ones(coef.shape) - coef) * p1, d1)
        d2 = np.where(cond, coef * p2 + (np.ones(coef.shape) - coef) * p3, d2)
    grad_suppress = np.where((grad >= d1) & (grad >= d2), grad, 0)

    # Double thresholding
    high, low = 0.75, 0.4

    sorted_grad = np.sort(grad.reshape(-1))
    thresh_low = sorted_grad[int(len(sorted_grad) * low)]
    thresh_high = sorted_grad[int(len(sorted_grad) * high)]

    neighbors = Neighbors(grad_suppress, 3).neighbors

    return np.where((grad_suppress > thresh_low) &
                    (np.max(neighbors, axis=2) > thresh_high), 1, 0)


def morphology(img: ndarray) -> ndarray:
    def dilate(img: ndarray, k: int) -> ndarray:
        return np.max(Neighbors(img, k).neighbors, axis=2)

    def erode(img: ndarray, k: int) -> ndarray:
        return np.min(Neighbors(img, k).neighbors, axis=2)

    k = 3
    img = erode(img, k)
    img = dilate(img, k)
    return img
