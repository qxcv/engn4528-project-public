#!/usr/bin/env python3
"""Explore the latent space of VAE and IDVAE. Used to produce Appendix C of our
report."""

import argparse

import numpy as np
from keras.models import load_model

from fidentify.datasets import LFW
from fidentify.vae import ImageHandler
from fidentify.utils import clamp01, im2double, grey2rgb

import matplotlib.pyplot as plt


def sane_imshow(im):
    im = clamp01(grey2rgb(im2double(im)))
    rv = plt.imshow(im)
    plt.axis('equal')
    plt.axis('off')
    return rv


parser = argparse.ArgumentParser()
parser.add_argument(
    '--lfw-path',
    default='data/lfw-deepfunneled',
    help='path to Labelled Faces in the Wild')
parser.add_argument(
    '--plain-vae-models',
    nargs=2,
    default=['data/plain-vae-rgb/encoder.h5', 'data/plain-vae-rgb/decoder.h5'],
    help='path to learnt encoder and decoder for plain VAE')
parser.add_argument(
    '--id-vae-models',
    nargs=2,
    default=[
        'data/ident-vae-rgb-112/encoder.h5',
        'data/ident-vae-rgb-112/decoder.h5'
    ],
    help='path to learnt encoder and decoder for ID VAE')
parser.add_argument(
    '--ident-vars',
    type=int,
    default=112,
    help='treat this many variables as belonging to identity group')

# parser.add_argument(
#     '--colour-space',
#     default='rgb',
#     choices=['grey', 'rgb'],
#     help='colour space to use for image loading')


def get_sampled_reconstructions(picture, n, encoder, decoder):
    """Pass a picture through the encoder/decoder pair several times, sampling
    different latents each time. Returns 4D stack of images."""
    assert picture.ndim == 3, "Need 3D picture array, got shape %s" \
        % (picture.shape,)
    means, vars = encoder.predict_on_batch(picture[None, ...])
    mean, std = means[0], np.sqrt(vars[0])
    # now sample a bunch of latents
    latents = np.random.randn(n, mean.size) * std + mean
    image_means, _ = decoder.predict(latents, batch_size=4)
    # don't bother sampling the images, it will just make them noisier
    return image_means


def get_tweaked_reconstructions(picture, n, encoder, decoder, id_num):
    """As above, except we try to see what arbitrary choices of appearance
    variables do. Keep first id_num latent distributions fixed, but replace
    last few with samples from N(0, I)."""
    assert picture.ndim == 3, "Need 3D picture array, got shape %s" \
        % (picture.shape,)
    means, vars = encoder.predict_on_batch(picture[None, ...])
    mean_id, std_id = means[0, :id_num], np.sqrt(vars[0, :id_num])
    id_latents = np.random.randn(n, mean_id.size) * std_id + mean_id
    app_num = means.size - id_num
    app_latents = np.random.randn(n, app_num)
    # join them together so that appearance latents come second
    latents = np.concatenate([id_latents, app_latents], axis=1)
    image_means, _ = decoder.predict(latents, batch_size=4)
    return image_means


if __name__ == '__main__':
    args = parser.parse_args()

    image_handler = ImageHandler('rgb', 128)
    lfw = LFW(args.lfw_path, image_handler)

    print('Loading plain VAE')
    plain_encoder_path, plain_decoder_path = args.plain_vae_models
    plain_encoder = load_model(plain_encoder_path)
    plain_decoder = load_model(plain_decoder_path)

    print('Loading identity VAE')
    id_encoder_path, id_decoder_path = args.id_vae_models
    id_encoder = load_model(id_encoder_path)
    id_decoder = load_model(id_decoder_path)

    people = lfw.person_pictures(is_train=True)
    # display grid of this many * this many people that you can CLICK ON! WOO!
    pgrid_size = 4
    pics = sorted(set().union(*people.values()))
    chosen_pics = np.random.choice(
        pics, size=pgrid_size * pgrid_size, replace=False)
    person_click_fig = plt.figure()
    for subplot, chosen_pic in enumerate(chosen_pics, start=1):
        pic = lfw.get_picture(chosen_pic)
        plt.subplot(pgrid_size, pgrid_size, subplot)
        pic_handle = sane_imshow(pic)
        pic_handle.pic_id = chosen_pic
        pic_handle.set_picker(True)
    plt.suptitle('Select a person')

    display_fig = plt.figure()
    plt.title('Select a person in other figure')
    plt.axis('off')

    def on_click_pic(event):
        pic_id = event.artist.pic_id
        pic = lfw.get_picture(pic_id)
        ncols = 5

        plt.figure(display_fig.number)
        plt.clf()
        # draw original picture
        plt.subplot(5, ncols, 1)
        sane_imshow(pic)
        plt.title('Original', fontsize=8)

        plain_recon = get_sampled_reconstructions(pic, ncols, plain_encoder,
                                                  plain_decoder)
        plain_tweak = get_tweaked_reconstructions(
            pic, ncols, plain_encoder, plain_decoder, args.ident_vars)
        id_recon = get_sampled_reconstructions(pic, ncols, id_encoder,
                                               id_decoder)
        id_tweak = get_tweaked_reconstructions(pic, ncols, id_encoder,
                                               id_decoder, args.ident_vars)

        for i in range(1, ncols + 1):
            # plain VAE
            plt.subplot(5, ncols, ncols + i)
            sane_imshow(plain_recon[i - 1])
            if i == 1:
                plt.title('VAE decode', fontsize=8)

            plt.subplot(5, ncols, 2 * ncols + i)
            sane_imshow(plain_tweak[i - 1])
            if i == 1:
                plt.title('VAE tweak', fontsize=8)

            # identity VAE
            plt.subplot(5, ncols, 3 * ncols + i)
            sane_imshow(id_recon[i - 1])
            if i == 1:
                plt.title('IDVAE decode', fontsize=8)

            plt.subplot(5, ncols, 4 * ncols + i)
            sane_imshow(id_tweak[i - 1])
            if i == 1:
                plt.title('IDVAE tweak', fontsize=8)

        plt.suptitle('VAE on %s' % pic_id, fontsize=8)
        display_fig.canvas.draw()

    person_click_fig.canvas.mpl_connect('pick_event', on_click_pic)
    plt.show()
