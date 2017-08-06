#!/usr/bin/env python3
"""Crops all images in a directory of face images so that faces are of the
appropriate size and location, and in greyscale."""

import argparse
import os

from scipy.misc import imread, imsave, imresize

from fidentify.detector import FacePipe
from fidentify.utils import get_paths, rgb2grey

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help='directory containing input images')
parser.add_argument('output_dir', help='directory to put processed images in')
parser.add_argument(
    '--size', default=128, type=int, help='output width/height')

if __name__ == '__main__':
    args = parser.parse_args()
    print('Reading images from %s, writing images to %s' %
          (args.input_dir, args.output_dir))
    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        pass
    input_dir = os.path.abspath(args.input_dir)
    image_paths = get_paths(input_dir)
    pipe = FacePipe()
    for num, input_path in enumerate(image_paths, start=1):
        filename = os.path.basename(input_path)
        input_path_dirname = os.path.dirname(input_path)
        subdir = input_path_dirname[len(input_dir):]
        subdir = subdir.lstrip(os.path.sep)
        output_subdir = os.path.join(args.output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, filename)
        print('\n%d. Processing "%s", storing in "%s"' %
              (num, input_path, output_path))
        im = imread(input_path)
        faces = list(pipe.enumerate_faces(im, args.size))
        if len(faces) == 0:
            print("Didn't detect any faces (!); not cropping at all")
            # this will stretch faces sometimes
            out_im = imresize(im, (args.size, args.size))
        else:
            if len(faces) > 1:
                print('Got %d faces; using the first only' % len(faces))
            _, out_im = faces[0]
        out_im = rgb2grey(out_im)
        imsave(output_path, out_im)
