import importlib
import sys
import time
from typing import Callable, Optional
from numpy import ndarray
from skimage import io, color
import skimage


def load(path: str) -> ndarray:
    raw = io.imread(path)
    if raw.ndim == 3 and raw.shape[-1] == 4:
        raw = color.rgba2rgb(raw)
    return raw


path = [f'images/{x}' for x in ['office.jpg',
                                'chelsea.png',
                                'cameraman.tif',
                                'gantrycrane.png',
                                'chessboard.png',
                                'coins.png']]
images = [load(p) for p in path]
images.append(skimage.data.page())
images.append(skimage.data.text())


def count_time(func: Callable):
    def wrapper(*args):
        start = time.time()
        func(*args)
        end = time.time()
        print(round(end - start, 2))
    return wrapper


@count_time
def run_single(image_func: Callable):
    for img in images:
        result = image_func(img)


def parse_func(mod: str) -> Optional[Callable]:
    tokens = mod.lstrip('.\\').split('.')
    if len(tokens) >= 2:
        module = importlib.import_module(tokens[0])
        return vars(module).get(tokens[1], None)


def to_gray_scale() -> None:
    global images
    for i, img in enumerate(images):
        if img.ndim == 3:
            images[i] = color.rgb2gray(img)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        if arg == '-g':
            to_gray_scale()
            continue
        func = parse_func(arg)
        if func is not None:
            print(f'{func.__module__}.{func.__name__}')
            run_single(func)
        else:
            print(f'cannot load module {arg}', file=sys.stderr)
