#!/usr/bin/env python3
"""Evaluates accuracy of various face recognition strategies on LFW. Reproduces
accuracy table from our report."""

import argparse

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from fidentify.facedb import RandomMatcher
from fidentify.datasets import LFW
from fidentify.eigenfaces import EigenfaceMatcher
from fidentify.vae import VAEMatcher, ImageHandler
from fidentify.eigenfaces import make_image_mat, get_eigenvectors

# maximum number of images to use for eigenface training
EIGENFACE_MAX_IMAGES = 4000

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lfw-path',
    default='data/lfw-deepfunneled',
    help='path to Labelled Faces in the Wild')
parser.add_argument(
    '--method',
    choices=['eigen', 'vae', 'rand'],
    default='vae',
    help='method to use for face recognition (eigenfaces get re-trained)')
parser.add_argument(
    '--vae-models',
    nargs=2,
    default=['data/plain-vae-rgb/encoder.h5', 'data/plain-vae-rgb/decoder.h5'],
    help='path to learnt encoder and decoder (only used by --method vae)')
parser.add_argument(
    '--ident-vars',
    type=int,
    default=None,
    help='truncate VAE descriptor to this many characters; implies IDVAE')
parser.add_argument(
    '--colour-space',
    default='rgb',
    choices=['grey', 'rgb'],
    help='colour space to use for image loading')
parser.add_argument(
    '--max-faces', default=16, type=int, help='maximum number of eigenfaces')

if __name__ == '__main__':
    # this ensures that our register/retrieve split is deterministic, among
    # other things
    np.random.seed(42)

    args = parser.parse_args()

    image_handler = ImageHandler(args.colour_space, 128)
    lfw = LFW(args.lfw_path, image_handler)

    if args.method == 'eigen':
        print('Using Eigenfaces, will have to re-train')
        path_lists = lfw.person_pictures(is_train=True).values()
        # just pick a random path in each case
        image_paths = [np.random.choice(paths) for paths in path_lists]
        # cut it off at 1000 paths (too hard to get eigenvectors otherwise)
        path_inds = np.random.permutation(len(image_paths))
        path_inds_trunc = path_inds[:EIGENFACE_MAX_IMAGES]
        image_paths_trunc = list(np.asarray(image_paths)[path_inds_trunc])
        image_mat = make_image_mat(image_paths_trunc)
        eigmat, mean = get_eigenvectors(image_mat, args.max_faces)
        print('Eigenvector matrix size: %s' % (eigmat.shape, ))
        matcher = EigenfaceMatcher(eigmat, mean)
    elif args.method == 'vae':
        print('Loading VAE')
        encoder_path, decoder_path = args.vae_models
        encoder = load_model(encoder_path)
        decoder = load_model(decoder_path)
        idvae = args.ident_vars is not None
        matcher = VAEMatcher(
            encoder, decoder, truncate_to=args.ident_vars, use_kld=idvae)
        if matcher.truncate_to is not None:
            assert matcher.truncate_to >= 1, "Need at least descriptor element"
            print("Truncating VAE output to %d elements" % matcher.truncate_to)
        if matcher.use_kld:
            print("VAE will use KLD to measure descriptor similarity")
        else:
            print("VAE will use L2 norm to measure descriptor similarity")
    else:
        print('Using stupid random baseline')
        matcher = RandomMatcher()

    # add entire LFW test set to classifier DB
    print('Adding fingerprints')
    test_people = lfw.person_pictures(is_train=False)
    held_out = {}
    skipped = 0
    not_skipped = 0
    for person, pic_ids in tqdm(sorted(test_people.items()), unit='people'):
        # give the matcher fingerprints for half the pictures, then use the
        # remaining half to try to retrieve the identities of the people in the
        # test set
        if len(pic_ids) < 2:
            skipped += 1
            continue
        not_skipped += 1
        pic_ids = np.asarray(pic_ids)
        n = len(pic_ids)
        to_add_mask = np.arange(n) == np.random.permutation(n)[n // 2]

        # we will allow the database to associate with this subset of pics
        for pic_id in tqdm(pic_ids[to_add_mask], unit='pictures'):
            matcher.add_picture(person, lfw.get_picture(pic_id))

        # we will try to find matches for the other subset of pics
        for pic_id in pic_ids[~to_add_mask]:
            held_out.setdefault(person, []).append(pic_id)

    print('\nUsed %d people, after skipping %d people who had too few pictures'
          % (not_skipped, skipped))

    # now we go through again getting top matches for everything in held_out
    print('Matching against held-out people')
    sorted_names = sorted(test_people.keys())
    int_ids = {p: i for i, p in enumerate(sorted_names)}
    true_labels = []
    predicted_labels = []
    samples = 0
    top = {5: 0, 10: 0, 25: 0}
    if top:
        k = max(top)
    else:
        k = 1
    for person, pic_ids in tqdm(held_out.items(), unit='people'):
        true_label = int_ids[person]
        for pic_id in tqdm(pic_ids, unit='pictures'):
            picture = lfw.get_picture(pic_id)
            matches = matcher.match_people(picture, k)
            match_ids = [int_ids[p] for p, _ in matches]

            # record top-K stats
            samples += 1
            for tk in top:
                if true_label in match_ids[:tk]:
                    top[tk] += 1

            # record top match for sklearn metrics
            true_labels.append(true_label)
            predicted_labels.append(match_ids[0])

    dest_path = 'classification-report-%s.txt' % args.method
    print("Done! Printing report and writing it to %s, too" % dest_path)
    report = classification_report(true_labels, predicted_labels).split('\n')
    report.append('Total accuracy: %f' %
                  accuracy_score(true_labels, predicted_labels))
    for tk in top:
        acc = top[tk] / float(samples)
        report.append('Total top-%d accuracy: %f' % (tk, acc))
    with open(dest_path, 'w') as fp:
        for line in report:
            print(line)
            print(line, file=fp)
