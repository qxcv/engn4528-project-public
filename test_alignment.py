#!/usr/bin/env python3
"""Runs face detector on image to show cropping performance. Useful tool for
working on alignment stages of pipeline."""

import argparse

import numpy as np
from scipy.misc import imread

from fidentify.detector import FacePipe

import matplotlib.pyplot as plt
from matplotlib import patches


parser = argparse.ArgumentParser()
parser.add_argument('group_photo', help='path to group photo')

if __name__ == '__main__':
    args = parser.parse_args()
    # we will show everyone in a big group photo
    print('Loading group photo')
    group_im = imread(args.group_photo)
    plt.figure()
    plt.imshow(group_im)

    print('Running face detector')
    fp = FacePipe()
    positions_images = list(fp.enumerate_faces(group_im))
    for x, y, w, h in [p for p, i in positions_images]:
        patch = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="green", linewidth=2)
        plt.gca().add_patch(patch)
    plt.figure()
    face_num = 0
    square_width = int(np.ceil(np.sqrt(len(positions_images))))
    for face_pos, face_im in positions_images:
        # also make a separate plot for each detected individual
        face_num += 1
        plt.subplot(square_width, square_width, face_num)
        plt.title('Person %d' % face_num)
        plt.imshow(face_im)
    plt.suptitle('Cropped faces')
    plt.show()
