from typing import Callable, List, Tuple
import numpy as np
from numpy import ndarray
from skimage import io, color, img_as_float
from skimage.util.dtype import img_as_ubyte
import os
import argparse
from sys import argv
from importlib import import_module


INPUT = 'images'
OUTPUT = 'output/'


suffix = ['.tif', '.png', '.jpg', '.jpeg', '.bmp', '.jfif']


def load_files(path: str) -> List[Tuple[str, ndarray]]:
    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(suf) for suf in suffix):
                img = io.imread(os.path.join(root, file))
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = color.rgba2rgb(img)
                img = img_as_float(img)

                name, ext = os.path.splitext(file)
                ret.append((name, img))
    return ret


def save_file(img: ndarray, method: Callable, iname: str, oname: str, flatten: bool) -> None:
    method_name = method.__name__
    if flatten:
        output = os.path.join(oname, iname + '_' + method_name + '.png')
    else:
        output = os.path.join(oname, method_name, iname + '.png')
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    if img.dtype == np.float64:
        io.imsave(output, (img*255).astype(np.uint8))
    else:
        io.imsave(output, img)


def run(args: argparse.Namespace):
    input_dir = INPUT if not args.input else args.input
    output_dir = OUTPUT if not args.output else args.output
    flatten = args.f
    alg = import_module(args.algorithm.rstrip('.py').lstrip('.\\'))
    for name, img in load_files(input_dir):
        print(name)
        if args.suffix:
            name += '_' + args.suffix
        methods = [alg.plane, alg.equalize, alg.denoise, alg.interpolate,
                   alg.dft, alg.butterworth, alg.canny, alg.morphology]
        for func in methods:
            if func == alg.plane:
                _img = img_as_ubyte(img) if img.ndim == 2 \
                    else img_as_ubyte(color.rgb2gray(img))
            elif func in [alg.dft, alg.butterworth, alg.canny, alg.morphology]:
                _img = img.copy() if img.ndim == 2 \
                    else color.rgb2gray(img)
            else:
                _img = img.copy()
            result = func(_img)
            if result is not None:
                save_file(result, func, name, output_dir, flatten)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str,
                        help='path for algorithm.py')
    parser.add_argument(
        '-i', '--input', type=str, help='input image dir')
    parser.add_argument(
        '-o', '--output', type=str, help='output image dir')
    parser.add_argument(
        '-s', '--suffix', type=str, help='suffix append after file name (not file extension)'
    )
    parser.add_argument('-f', action='store_true',
                        help="do not save the result in different dirs")
    args = parser.parse_args(argv[1:])
    run(args)
