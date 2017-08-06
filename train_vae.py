#!/usr/bin/env python3
"""Train a vanilla VAE on Labelled Faces in the Wild. Use ``fetch_data.sh`` (in
this directory) to download dataset, and ``python train_vae.py --help`` for
usage options. This script is not necessary when using demo.py"""

import argparse
from os import path, makedirs

import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from fidentify.vae import BasicVAE, IdentityVAE, ImageHandler, \
    OtherModelCheckpoint, ScaleRamper, VAESampleCallback
from fidentify.datasets import LFW


def batch_generator(face_dataset,
                    aug_gen,
                    is_train,
                    batch_size=16,
                    pics_per_person=None):
    """Yields batches of images from face_dataset augmented with aug_gen. If
    pics_per_person is None, then it will choose pictures uniformly at random;
    otherwise, it will choose people at random according to the number of
    pictures they have, then add at least pics_per_person augmented pictures to
    the batch for each chosen person."""
    if pics_per_person is None:
        pic_lists = face_dataset.person_pictures(is_train).values()
        all_paths = []
        for sublist in pic_lists:
            all_paths.extend(sublist)
    else:
        # for each batch we will draw people_per_batch people with
        # probabilities corresponding to the number of unique pictures they
        # have (so batches are going to be very heavy on GWB…)
        people_per_batch = max(batch_size // pics_per_person, 1)
        people = face_dataset.person_pictures(is_train)
        people_arr = sorted(people.keys())
        people_arr_probs = np.array(
            [len(people[k]) for k in people_arr], dtype='float32')
        people_arr_probs /= np.sum(people_arr_probs)

    while True:
        if pics_per_person is None:
            # choose images uniformly at random
            chosen_paths = np.random.choice(all_paths, size=(batch_size, ))
            images = map(face_dataset.get_picture, chosen_paths)
            images_aug = map(aug_gen.random_transform, images)
            images_aug_stack = np.stack(images_aug, axis=0)
            # targets don't really matter with the current loss setup, so I'll
            # just give it the same image tensor
            batch_data = (images_aug_stack, images_aug_stack)

        else:
            # need to choose some people and then choose pictures corresponding
            # to those people
            replace_people = people_per_batch > len(people_arr)
            chosen_people = np.random.choice(
                people_arr,
                size=(people_per_batch, ),
                replace=replace_people,
                p=people_arr_probs)
            images = []
            ident_list = []

            for pidx, person in enumerate(chosen_people):
                # pick out some images for this person
                if pidx == len(chosen_people) - 1:
                    to_pick = batch_size - len(images)
                else:
                    to_pick = pics_per_person

                assert to_pick > 0
                # we'll need to replace images if we ask for too many
                replace = to_pick > len(people[person])
                paths = np.random.choice(
                    people[person], size=(to_pick, ), replace=replace)

                # the last person might need some extra pics
                images.extend(map(face_dataset.get_picture, paths))
                ident_list.extend([pidx] * to_pick)

            ident_vec = np.array(ident_list, dtype='int')
            images_aug = map(aug_gen.random_transform, images)
            images_aug_stack = np.stack(images_aug, axis=0)
            in_dict = {'images': images_aug_stack, 'identities': ident_vec}
            batch_data = (in_dict, images_aug_stack)

        yield batch_data


parser = argparse.ArgumentParser(description='Train a simple VAE on LFW')
parser.add_argument(
    '--work-dir',
    default='data/working/vae/',
    help='where to save checkpoints, image mean, etc.')
parser.add_argument(
    '--colour-space',
    choices=['rgb', 'grey'],
    default='rgb',
    help='colour (or image intensity) representation to use')
parser.add_argument(
    '--checkpoints', nargs=2, default=None, help='encoder and decoder to load')
parser.add_argument(
    '--batch-size', type=int, default=16, help='size of train/test batches')
parser.add_argument(
    '--num-epochs',
    type=int,
    default=4000,
    help='number of training epochs to go through')
parser.add_argument(
    '--epoch-size', type=int, default=512, help='batches per epoch')
parser.add_argument(
    '--tb-hist-freq',
    type=int,
    default=1,
    help='frequency of TensorBoard histogram writes')
parser.add_argument(
    '--lfw-dir',
    default='data/lfw-deepfunneled',
    help='path to Labelled Faces in the Wild (LFW)')
parser.add_argument(
    '--vae-channels',
    type=int,
    default=8,
    help='number of channels for 4x4 VAE representation')
parser.add_argument(
    '--app-vars',
    type=int,
    default=None,
    help='number of appearance latents (unsupplied = use vanilla VAE)')

if __name__ == '__main__':
    args = parser.parse_args()

    assert K.backend() == 'tensorflow', \
        "This script only works with KERAS_BACKEND=tensorflow"

    print('WARNING: This script loads all data into memory at once.')
    print('If you get allocation errors then you need a beefier machine :-)')

    image_handler = ImageHandler(args.colour_space, 128)

    print('Indexing training data')
    lfw = LFW(args.lfw_dir, image_handler)

    # make working dir
    subdirs = ['checkpoints', 'pretty', 'augmented']
    for subdir in subdirs:
        makedirs(path.join(args.work_dir, subdir), exist_ok=True)

    # generates augmented data, saves mean pixel
    print('Fitting data generator')
    aug_gen = ImageDataGenerator(
        # featurewise_center=True,
        horizontal_flip=True,
        rotation_range=7.5,
        width_shift_range=0.025,
        height_shift_range=0.025,
        channel_shift_range=0.01,
        zoom_range=0.05)
    if args.app_vars is not None:
        pics_per_person = min(args.batch_size, max(2, args.batch_size // 4))
    else:
        pics_per_person = None
    big_train_batch = next(
        batch_generator(
            lfw, aug_gen, True, 2048, pics_per_person=pics_per_person))
    big_test_batch = next(
        batch_generator(
            lfw, aug_gen, False, 2048, pics_per_person=pics_per_person))
    if args.app_vars is not None:
        big_train_images = big_train_batch[0]['images']
        big_test_images = big_test_batch[0]['images']
    else:
        big_train_images = big_train_batch[0]
        big_test_images = big_test_batch[0]
    aug_gen.fit(big_train_images)
    np.savez(path.join(args.work_dir, 'norm_data.npz'), mean=aug_gen.mean)

    print('Building models')
    encoder = decoder = None
    if args.checkpoints is not None:
        # TODO: add ability to reload ALL arguments/parameters from past
        # experiment (to make resumable training easier)
        enc_path, dec_path = args.checkpoints
        print('Using encoder and decoder at %s and %s, respectively' %
              (enc_path, dec_path))
        encoder = load_model(enc_path)
        decoder = load_model(dec_path)
    total_latents = 4 * 4 * args.vae_channels
    if args.app_vars is not None:
        ident_vars = total_latents - args.app_vars
        print(('Creating identity VAE with %d identity variables and %d '
               'appearance variables') % (ident_vars, args.app_vars))
        vae_cls = IdentityVAE(ident_vars, args.app_vars, image_handler)
    else:
        print('Creating VAE with %d latents' % total_latents)
        vae_cls = BasicVAE(args.vae_channels, image_handler)
    vae, encoder, decoder = vae_cls.make_vae(encoder, decoder)

    print('Training, go!')
    encoder_save_path = path.join(args.work_dir, 'checkpoints',
                                  'encoder_checkpoint_{epoch:03d}.h5')
    decoder_save_path = path.join(args.work_dir, 'checkpoints',
                                  'decoder_checkpoint_{epoch:03d}.h5')
    log_path = path.join(args.work_dir, 'logs.csv')
    callbacks = [
        # yapf: disable
        # save best models AND save every 20 epochs
        OtherModelCheckpoint(encoder, encoder_save_path, save_best_only=True),
        OtherModelCheckpoint(decoder, decoder_save_path, save_best_only=True),
        OtherModelCheckpoint(encoder, encoder_save_path, period=20),
        OtherModelCheckpoint(decoder, decoder_save_path, period=20),
        # this probably isn't helping much because of the way we ramp up loss
        # over time
        ReduceLROnPlateau(patience=60),
        CSVLogger(log_path),
        VAESampleCallback(big_train_images, big_test_images, decoder, aug_gen,
                          path.join(args.work_dir, 'pretty'), image_handler,
                          args.app_vars is not None),
        # 100 seems to be the goldilocks zone for the current (2017-05-21)
        # configuration. It also means that the loss is a real lower bound on
        # NLL (although not as tight as it could be).
        ScaleRamper(schedule=[1e-10, 0.0001, 0.01, 1, 10, 100], wait_time=15)
        # yapf: enable
    ]
    print('Enabling TensorBoard')
    tb_dir = path.join(args.work_dir, 'tensor board')
    # TODO: how do I use the embedding functionality? Could make some cool
    # plots :)
    try:
        callbacks.append(
            TensorBoard(
                tb_dir,
                histogram_freq=args.tb_hist_freq,
                # This uses too much memory (e.g. doesn't work with batch
                # size of eight!). Unfortunate, since grads and acts are
                # far more informative than weights.
                # write_grads=True,
                batch_size=args.batch_size, ))
    except TypeError as e:
        print("Couldn't enable TensorBoard (may be using older version "
              "of Keras). Error:\n%s" % e)
    real_train_gen = batch_generator(
        lfw, aug_gen, False, args.batch_size, pics_per_person=pics_per_person)
    vae.fit_generator(
        real_train_gen,
        args.epoch_size,
        epochs=args.num_epochs,
        validation_data=big_test_batch,
        callbacks=callbacks)
