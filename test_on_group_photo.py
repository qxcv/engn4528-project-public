#!/usr/bin/env python3
"""Script to run complete, trained face recognition pipeline. It is easiest to
run this through demo.py."""

import argparse
from time import time
from os import path, makedirs

import numpy as np
from scipy.misc import imread, imsave

from keras.models import load_model

from fidentify.detector import FacePipe
from fidentify.eigenfaces import EigenfaceMatcher
from fidentify.vae import VAEMatcher
from fidentify.utils import clamp01, im2double, grey2rgb, crop

import matplotlib.pyplot as plt
from matplotlib import patches


def sane_imshow(im, save_dir=None, save_name=None):
    """Like sane_imshow, but (1) makes sure that image is clamped correctly so
    that MPL doesn't apply a weird colour scheme to it, and (2) takes out ugly
    axes."""
    # Convert everything to float, clamp within right range, then check that
    # it's RGB. I came upon this process after a lot of trial-and-error.
    if im.ndim == 3 and im.shape[2] == 1:
        # drop last dimension
        im = im[..., 0]
    im = clamp01(grey2rgb(im2double(im)))
    plt.imshow(im)
    plt.axis('equal')
    plt.axis('off')

    if save_dir is not None:
        # save to some path if save_dir exists
        assert save_name is not None, 'need save dir AND save name'
        save_path = path.join(save_dir, save_name)
        makedirs(save_dir, exist_ok=True)
        print('Saving image to %s' % save_path)
        imsave(save_path, im)


parser = argparse.ArgumentParser()
parser.add_argument('identities', help='path to .ini file defining identities')
parser.add_argument('group_photo', help='path to group photo')
parser.add_argument(
    '--method',
    choices=['eigen', 'vae', 'idvae', 'plainvae'],
    default='eigen',
    help='method to use for face recognition')
parser.add_argument(
    '--eigenfaces',
    default='data/eigenfaces.npz',
    help='path to learnt eigenfaces (only used by --method eigen)')
parser.add_argument(
    '--save-to',
    default=None,
    help='save produced images to a directory, in addition to showing them '
    'on-screen')
parser.add_argument(
    '--vae-models',
    nargs=2,
    default=[
        'data/plain-vae-grey/encoder.h5', 'data/plain-vae-grey/decoder.h5'
    ],
    help='path to learnt encoder and decoder (only used by --method vae)')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.method == 'eigen':
        print('Loading eigenfaces')
        fp = np.load(args.eigenfaces)
        db = EigenfaceMatcher(fp['eigmat'], fp['mu'])
    else:
        print('Loading VAE')
        if args.method == 'idvae':
            print('Identity VAE, ignoring --vae-models')
            encoder_path = 'data/ident-vae-grey-112/encoder.h5'
            decoder_path = 'data/ident-vae-grey-112/decoder.h5'
        elif args.method == 'plainvae':
            print('Plain VAE, ignoring --vae-models')
            encoder_path = 'data/plain-vae-grey/encoder.h5'
            decoder_path = 'data/plain-vae-grey/decoder.h5'
        else:
            print('Arbitrary VAE, using --vae-models if supplied')
            encoder_path, decoder_path = args.vae_models
        encoder = load_model(encoder_path)
        decoder = load_model(decoder_path)
        db = VAEMatcher(encoder, decoder)
    print('Faceprinting people in test set')
    start = time()
    db.load_id_file(args.identities)
    print('Took %fs to faceprint everyone' % (time() - start))

    # we will show everyone in a big group photo
    print('Loading group photo')
    group_im = imread(args.group_photo)
    main_fig = plt.figure('Group photo', figsize=(12.80, 7.20))
    sane_imshow(group_im)

    print('Running face detector')
    start = time()
    fp = FacePipe()
    positions_images = list(fp.enumerate_faces(group_im))
    people_present = set()
    for idx, pi in enumerate(positions_images):
        box, person_pic = pi
        x, y, w, h = box
        patch = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="green", linewidth=2, picker=1)
        # so that we can use it from picker callback
        patch.person_idx = idx
        plt.gca().add_patch(patch)

        # recognise this person
        match, = db.match_people(person_pic, 1)
        # TODO: can avoid false positives by using similarity threshold
        people_present.add(match[0])

    people_absent = set(db.people.keys()) - people_present
    print('=' * 78)
    print('Detected faces: %d' % len(positions_images))
    print('Distinct identities: %d' % len(people_present))
    print('Subjects present: ' + ', '.join(sorted(people_present)) + '\n')
    print('Subjects absent: ' + ', '.join(sorted(people_absent)))
    print('Took %fs to do matching and produce report' % (time() - start))
    print('=' * 78)

    # will display result of each stage of pipeline applied to crop
    person_fig = plt.figure('Pipeline detail', figsize=(7.20, 4.00))
    plt.title('Select a person')
    plt.axis('off')

    # will display nearest identities
    nearest_fig = plt.figure('Nearest faces', figsize=(12.8, 7.20))
    match_grid_size = 4
    plt.title('Select a person')
    plt.axis('off')

    def on_click_box(event):
        person_idx = event.artist.person_idx
        print('Detection #%d selected' % person_idx)
        face_pos, face_im = positions_images[person_idx]

        if args.save_to is not None:
            photo_basename = path.basename(args.group_photo)
            save_dir = path.join(args.save_to, 'row-%s-%d-%d-%d-%d' % (
                (photo_basename, ) + tuple(map(int, face_pos))))
        else:
            save_dir = None

        plt.figure(person_fig.number)
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.title('Original crop')
        orig_crop = crop(group_im, *face_pos)
        sane_imshow(orig_crop, save_dir, 'original.jpg')

        plt.subplot(1, 3, 2)
        plt.title('Preprocessed')
        sane_imshow(face_im, save_dir, 'preprocessed.jpg')

        # plot reproduced faces
        plt.subplot(1, 3, 3)
        plt.title('From descriptor')
        face_descriptor = db.face_descriptor(face_im)
        reconst_face = db.reconstruct_descriptor(face_descriptor)
        sane_imshow(reconst_face, save_dir, 'reconstruction.jpg')

        plt.suptitle('Detected person #%d' % person_idx)

        person_fig.canvas.draw()

        plt.figure(nearest_fig.number)
        matches = db.match_people(face_im, match_grid_size**2)
        for subplot, match in enumerate(matches):
            person, similarity = match
            # show original picture of other person
            person_im = db.get_pictures(person)[0]
            plt.subplot(match_grid_size, 2 * match_grid_size, 2 * subplot + 1)
            sane_imshow(person_im)
            plt.title('%s (d=%g)' % (person, similarity), fontsize=10)
            sane_imshow(person_im, save_dir, 'match-%d.jpg' % subplot)

            # show picture run through reconstruction method
            # shows to the right of original
            plt.subplot(match_grid_size, 2 * match_grid_size, 2 * subplot + 2)
            person_descriptor = db.face_descriptor(person_im)
            reconst_person_im = db.reconstruct_descriptor(person_descriptor)
            sane_imshow(reconst_person_im, save_dir, 'match-recon-%d.jpg' %
                        subplot)

        plt.suptitle('Nearest faces for detection %d' % person_idx)
        nearest_fig.canvas.draw()

    main_fig.canvas.mpl_connect('pick_event', on_click_box)
    plt.show()
