#!/usr/bin/env python3
"""Obtain a mean image vector and collection of covariance eigenvectors for
Eigenfaces. Running demo.py will use pretrained vectors."""

import argparse
import sys

import numpy as np

from fidentify.eigenfaces import make_image_mat, get_eigenvectors
from fidentify.utils import get_paths

parser = argparse.ArgumentParser()
parser.add_argument('training_dir', help='directory holding training images')
parser.add_argument(
    '--dest', default='data/eigenfaces.npz', help='where to store eigenfaces')
parser.add_argument(
    '--max-faces',
    default=None,
    type=int,
    help='maximum number of eigenfaces to store')

if __name__ == '__main__':
    args = parser.parse_args()
    image_paths = get_paths(args.training_dir)
    print('Processing %d images from %s' % (len(image_paths),
                                            args.training_dir))
    if len(image_paths) == 0:
        print(
            'No files to process in %s; exiting' % args.training_dir,
            file=sys.stderr)
        sys.exit(1)
    image_mat = make_image_mat(image_paths)
    eigmat, mu = get_eigenvectors(image_mat, args.max_faces)
    print('Saving to %s' % args.dest)
    np.savez(args.dest, eigmat=eigmat, mu=mu)
