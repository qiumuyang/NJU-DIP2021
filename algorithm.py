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
    return median_filter_fast(img, 3)


@split_channel
def interpolate(img: ndarray) -> ndarray:
    def bilinear(img: ndarray, kw: float, kh: float) -> ndarray:
        # prepare the param matrix (shape [h, w, 4])
        r10 = np.roll(img, -1, axis=1)
        r10[:, -1] = img[:, -1]  # margin
        r01 = np.roll(img, -1, axis=0)
        r01[-1, :] = img[-1, :]  # margin
        r11 = np.roll(r01, -1, axis=1)
        r11[:, -1] = r01[:, -1]  # margin
        # p0 = f(1, 0) - f(0, 0)
        p0 = r10 - img
        # p1 = f(0, 1) - f(0, 0)
        p1 = r01 - img
        # p2 = f(1, 1) + f(0, 0) - f(0, 1) - f(1, 0)
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
        xy_values: ndarray = np.dstack(
            [xy[:, :, 0], xy[:, :, 1], xy[:, :, 0] * xy[:, :, 1], np.ones((hh, ww))])
        ret = np.matmul(selected_params.reshape(hh, ww, 1, 4),
                        xy_values.reshape(hh, ww, 4, 1))
        ret = ret.reshape(hh, ww)
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
        # D_uv = sqrt((u - M / 2) ^ 2 + (v - N / 2) ^ 2)
        xs = np.power((xs - w / 2), 2)
        ys = np.power((ys - h / 2), 2)
        D_uv = np.power(xs + ys, 0.5)
        H_uv = 1 / (1 + np.power(D_uv / D_0, 2 * k))
        # G_uv = H_uv * F_uv
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
    high, low = 0.8, 0.4

    sorted_grad = np.sort(grad.reshape(-1))
    thresh_low = sorted_grad[int(len(sorted_grad) * low)]
    thresh_high = sorted_grad[int(len(sorted_grad) * high)]

    neighbors = Neighbors(grad_suppress, 3).neighbors

    return np.where((grad_suppress > thresh_low) &
                    (np.max(neighbors, axis=2) > thresh_high), 1, 0)


def morphology(img: ndarray) -> ndarray:
    return None
