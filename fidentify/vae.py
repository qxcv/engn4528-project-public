"""Variational autoencoder support code."""

from os import path
from multiprocessing import Pool

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Input, Activation, \
    Lambda, Flatten, Reshape, Dense, Layer, UpSampling2D, Concatenate
from keras.optimizers import RMSprop
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
from skimage.color import rgb2gray, gray2rgb
import tensorflow as tf

from fidentify.facedb import Recogniser
from fidentify.preprocessing import Floating, RGB, Greyscale, \
    ResizedSquare
from fidentify.utils import clamp01, gaussian_kld_pseudometric

INIT_LR = 0.0001

# TODO list:
# - Try decomposing latents into appearance and identity in a manner similar to
#   DC-IGN, but without bizarre backprop hacks. You came up with an elegant
#   implementation strategy for this while you were reading the paper (well, it
#   was elegant modulo the hacking you'll have to do to ensure that the data
#   generator serves up batches with several copies of the same face). If you
#   run into trouble, you might want to make appearance latents shallower &
#   better structured (don't flatten them) so that they actually start to
#   capture things like background, instead of being used to reproduce face.
#   Remember to hide capability to use structured latents behind a flag so that
#   you can toggle it off at evaluation time.


def load_paths_as_tensor(image_paths, image_handler):
    """Load N image paths into a N*H*W*3 tensor."""
    loader = image_handler.load_as_colspace
    with Pool() as pool:
        lab_iter = pool.imap(loader, image_paths)
        return np.stack(lab_iter, axis=0)


def rgb2colspace(rgb, colspace):
    """Convert an RGB image into desired colour space."""
    if colspace == 'rgb':
        return rgb
    if colspace == 'grey':
        luminance = rgb2gray(rgb)
        # add back in third channel
        return luminance[..., None]
    raise ValueError('Unknown colour space "%s"' % colspace)


def colspace2rgb(other, colspace):
    """Convert image in given colourspace to nearest RGB representation."""
    if colspace == 'rgb':
        return other
    if colspace == 'grey':
        assert other.ndim == 3 and other.shape[-1] == 1, \
            "Need single-channel 3D tensor, got shape %s" % (other.shape,)
        return gray2rgb(other[..., 0])
    raise ValueError('Unknown colour space "%s"' % colspace)


def unit_gaussian_kl(mean, var):
    """KL divergence between Gaussian defined by mean/std tensors and a
    unit Gaussian."""
    mean_flat, var_flat = K.batch_flatten(mean), K.batch_flatten(var)
    noise_dim = K.cast(K.prod(K.shape(mean_flat)[1:]), K.floatx())
    var_sum = K.sum(var_flat, axis=-1)
    log_var_sum = K.sum(K.log(var_flat), axis=-1)
    mean_squares = K.sum(K.square(mean_flat), axis=-1)
    return 0.5 * (var_sum + mean_squares - log_var_sum - noise_dim)


def pairwise_gaussian_kl(mean1, var1, mean2, var2):
    """Full KL divergence between two arbitrary Gaussians."""
    mean1, var1 = K.batch_flatten(mean1), K.batch_flatten(var1)
    mean2, var2 = K.batch_flatten(mean2), K.batch_flatten(var2)
    noise_dim = K.cast(K.prod(K.shape(mean1)[1:]), K.floatx())
    sq_mean_diff = K.square(mean1 - mean2)
    first_tens = (var1 + sq_mean_diff) / var2
    log_var1 = K.log(var1)
    log_var2 = K.log(var2)
    out_sum = K.sum(first_tens + log_var2 - log_var1, axis=-1)
    return 0.5 * (out_sum - noise_dim)


def gaussian_nll(mean, var, samples):
    """Negative log likelihood of given samples under diagonal multivariate
    Gaussian defined by mean and std."""
    mean_flat = K.batch_flatten(mean)
    var_flat = K.batch_flatten(var) + 1e-5
    samples_flat = K.batch_flatten(samples)
    noise_dim = K.cast(K.prod(K.shape(mean_flat)[1:]), K.floatx())
    l2pi = np.log(2 * np.pi)
    scale_diff = K.square((mean_flat - samples_flat)) / var_flat
    scale_diff_sum = K.sum(scale_diff, axis=-1)
    log_var_sum = K.sum(K.log(var_flat), axis=-1)
    return 0.5 * (noise_dim * l2pi + log_var_sum + scale_diff_sum)


class PureLossLayer(Layer):
    """Layer which passes its inputs straight through after adding a loss based
    on those inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        loss = self.make_loss(*inputs)
        self.add_loss(K.mean(loss), inputs)
        return inputs

    def make_loss(self, *inputs):
        raise NotImplementedError


class PureGaussNLL(PureLossLayer):
    """As above, but computes negative log likelihood of some samples under a
    Gaussian."""

    def make_loss(self, mean, var, samples):
        return gaussian_nll(mean, var, samples)


class PureKLBase(PureLossLayer):
    """Handles setting of KL coefficient (see PureUnitKL)."""

    def __init__(self, kl_coeff=1.0, **kwargs):
        super().__init__(**kwargs)
        self._kl_coeff = K.variable(kl_coeff)

    @property
    def kl_coeff(self):
        return K.get_value(self._kl_coeff)

    @kl_coeff.setter
    def kl_coeff(self, value):
        K.set_value(self._kl_coeff, value)

    def get_config(self):
        config = {'kl_coeff': self.kl_coeff}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PureUnitKL(PureKLBase):
    """Passes tensors of means and variances through unchanged, but adds loss
    term based on their KL divergence from unit Gaussian.

    Has ability to multiply KL divergence by a coefficient which can be turned
    up or down over time."""

    def make_loss(self, mean, var):
        return self._kl_coeff * unit_gaussian_kl(mean, var)


class PurePairKL(PureKLBase):
    """Passes tensors of means and variances through unchanged, but adds loss
    term based on their KL divergence from one another.

    Like PureUnitKL, can set coefficient as needed."""

    def make_loss(self, mean1, var1, mean2, var2):
        return self._kl_coeff * pairwise_gaussian_kl(mean1, var1, mean2, var2)


class ScaleRamper(Callback):
    """Callback to gradually ramp up coefficient on KL divergence. Helps the
    network to learn an accurate representation early, then improve its
    generalisation ability later."""

    def __init__(self, schedule, wait_time):
        self.schedule = list(schedule)
        self.wait_time = wait_time

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch % self.wait_time) == 0:
            sched_ind = epoch // self.wait_time
            if sched_ind >= len(self.schedule):
                print('Exhausted %d coefficients on epoch %d' %
                      (len(self.schedule), epoch))
                return
            next_scale = self.schedule[sched_ind]
            n = 0
            for layer in self.get_target_layers():
                n += 1
                layer.kl_coeff = next_scale
            print('Ramped KL coefficient up to %g on %d layers' %
                  (next_scale, n))

    def get_target_layers(self):
        for layer in self.model.layers:
            if isinstance(layer, PureKLBase):
                yield layer


class OtherModelCheckpoint(ModelCheckpoint):
    """Like ModelCheckpoint callback, but saves a *different* model to the one
    being trained."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().set_model(model)

    def set_model(self, model):
        # do nothing because we already have our model
        # this only exists to stop same method being called from superclass
        pass


class VAESampleCallback(Callback):
    """Makes some pretty pictures from VAE so I can see how well it's
    training."""

    def __init__(self,
                 train_data,
                 test_data,
                 decoder,
                 aug_gen,
                 save_dir,
                 image_handler,
                 is_ident=False):
        self.train_data = train_data
        self.test_data = test_data
        self.aug_gen = aug_gen
        self.save_dir = save_dir
        self.decoder = decoder
        self.image_handler = image_handler
        self.is_ident = is_ident

    def _enc_montage(self, data):
        """Select some random images from data tensor and run them through
        encoder a few times. Return big image showing results."""
        to_sel = 10
        num_samp = 9
        sel_inds = np.random.permutation(len(data))[:to_sel]
        batch_ims = data[sel_inds]
        batch_ims = self.aug_gen.standardize(batch_ims)
        if self.is_ident:
            batch = {
                'images': batch_ims,
                'identities': np.arange(len(batch_ims))
            }
        else:
            batch = batch_ims
        # we will have one selected image per row
        out_image = np.concatenate(batch_ims, axis=0)
        for i in range(num_samp):
            # we will also have a different batch of samples in each column
            results = self.model.predict_on_batch(batch)
            new_col = np.concatenate(results, axis=0)
            out_image = np.concatenate([out_image, new_col], axis=1)
        return self._postproc(out_image)

    def _postproc(self, out_image):
        # add back in mean, shift back into range, convert to bytes
        # TODO: need to handle other forms of standardisation that aug_gen can
        # do, if I use them.
        if self.aug_gen.mean:
            out_shift = out_image + self.aug_gen.mean.reshape((1, 1, -1))
        else:
            out_shift = np.copy(out_image)
        out_shift[out_shift < 0] = 0
        out_shift[out_shift > 1] = 1
        out_rgb = self.image_handler.to_rgb(out_shift)
        out_bytes = (out_rgb * 255).astype('uint8')
        return out_bytes

    def _dec_montage(self):
        rows = 10
        cols = 10
        total = rows * cols
        noise = np.random.randn(total, np.prod(self.decoder.input_shape[1:]))
        outputs, _ = self.decoder.predict(noise, batch_size=4)
        outputs = outputs.reshape((rows, cols) + outputs.shape[1:])
        out_image = np.concatenate(outputs, axis=1)
        out_image = np.concatenate(out_image, axis=1)
        assert out_image.ndim == 3, out_image.shape
        return self._postproc(out_image)

    def on_epoch_end(self, epoch, logs={}):
        save_pfx = path.join(self.save_dir, 'samples_%d' % epoch)

        # make a big spread of pictures from training data
        train_save_path = save_pfx + '_train.jpg'
        train_montage = self._enc_montage(self.train_data)
        imsave(train_save_path, train_montage)

        # now for test data
        test_save_path = save_pfx + '_test.jpg'
        test_montage = self._enc_montage(self.test_data)
        imsave(test_save_path, test_montage)

        # pass some random noise through the network and see what it comes up
        # with
        gen_path = save_pfx + '_random.jpg'
        gen_montage = self._dec_montage()
        imsave(gen_path, gen_montage)


class BasicVAE:
    """Class for a simple convolutional VAE which maps image down into
    low-dimensional latent representation, then maps back up to full scale.
    Does not attempt to distinguish between 'identity' and 'appearance'
    latents."""

    def __init__(self, latent_dim, image_handler):
        """Stores vars required for construction.

        Args:
            latent_dim: Number of channels in latent representation. More
                specifically, latents occupy a h*w*c tensor, and latent_dim
                gives the c.
        """
        self.latent_dim = latent_dim
        self.image_size = 128
        # activations before latents will be latent_size*latent_size*latent_dim
        # (like an image), then get flattened down
        self.latent_size = 4
        # how many /2 downsampling steps do we need?
        # self.downsample_steps = int(
        #     np.log(self.image_size / self.latent_size, 2))
        self.colspace_dim = image_handler.colspace_dim

    def _make_decoder(self):
        # input (4x4 latents)
        in_dim = self.latent_size**2 * self.latent_dim
        x = in_layer = Input(shape=(in_dim, ), name='dec_in')
        # Dropout will hopefully lead to more noise-tolerant latent
        # distribution. TODO: Not sure what the probabilistic interpretation
        # is. Might actually make sense to incorporate it in KL term or
        # something.
        # x = Dropout(0.2)(x)
        # FC linear map and reshape
        x = Dense(in_dim)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Reshape((self.latent_size, self.latent_size, self.latent_dim))(x)
        # now start doing transposed convolutions
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # upsample step (goes to 8x8)
        # x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # upsample step (goes to 16x16)
        # x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # upsample step (goes to 32x32)
        # x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # upsample step (goes to 64x64)
        # x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # upsample step (goes to 128x128)
        # x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        dec_mean = Conv2D(
            self.colspace_dim, (3, 3),
            padding='same',
            name='dec_mean',
            activation='sigmoid')(x)
        dec_var = Conv2D(
            self.colspace_dim, (3, 3),
            padding='same',
            name='dec_var',
            activation='softplus')(x)

        decoder = Model(
            inputs=[in_layer], outputs=[dec_mean, dec_var], name='decoder')

        return decoder

    def _make_encoder(self):
        # input (128x128 image)
        x = in_layer = Input(
            shape=(self.image_size, self.image_size, self.colspace_dim),
            name='enc_in')
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # downsample step (goes to 64x64)
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # downsample step (goes to 32x32)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # downsample step (goes to 16x16)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # downsample step (goes to 8x8)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        # downsample step (goes to 4x4)
        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Activation('relu')(x)

        # flatten, will hopefully make it harder for VAE to learn to produce
        # spatial correlations in latent distribution
        x = Flatten()(x)
        last_enc = BatchNormalization()(x)
        # latents!
        out_size = self.latent_size**2 * self.latent_dim
        mean = Dense(
            out_size, input_shape=(out_size, ), name='enc_mean')(last_enc)
        var = Dense(
            out_size,
            input_shape=(out_size, ),
            activation='softplus',
            name='enc_var')(last_enc)

        encoder = Model(inputs=[in_layer], outputs=[mean, var], name='encoder')

        return encoder

    def make_vae(self, encoder=None, decoder=None):
        if encoder is None:
            encoder = self._make_encoder()
        if decoder is None:
            decoder = self._make_decoder()

        encoder_in = Input(shape=(self.image_size, self.image_size,
                                  self.colspace_dim))
        # should be much more stable to produce variance than to produce
        # standard deviation, since we don't get a tiny quantity when we
        # square!
        mean, var = encoder(encoder_in)
        std = Lambda(lambda t: K.sqrt(t))(var)
        mean, var = PureUnitKL()([mean, var])

        def make_noise(layers):  # noqa
            # reparamterisation trick to sample latnets
            mu, sigma = layers
            noise = K.random_normal(shape=K.shape(sigma), mean=0., stddev=1.)
            return noise * sigma + mu

        latent = Lambda(
            make_noise,
            name='make_noise',
            output_shape=(self.latent_size**2 * self.latent_dim, ))(
                # must pass in STANDARD DEVIATION, NOT variance
                [mean, std])

        # now get an image back from our decoder
        decoder_mean, decoder_var = decoder(latent)
        decoder_mean, decoder_var, _ = PureGaussNLL()(
            [decoder_mean, decoder_var, encoder_in])

        vae = Model(inputs=[encoder_in], outputs=[decoder_mean], name='vae')

        # "Official" loss is empty; real losses come from Pure* layers!
        def null_loss(x, x_hat):
            # can't use K.zeros because it tries to pass something to Numpy :(
            return K.sum(K.zeros_like(K.batch_flatten(x)), axis=-1)

        # now we make some fake losses to see KL and Gaussian NLL
        # basically pulling loss tensor out of VAE itself (super hacky, should
        # patch Keras to let you show intermediate loss as metric!)
        def kl_loss(*args):
            assert len(vae.losses) == 2
            return vae.losses[0]

        def nll_loss(*args):
            assert len(vae.losses) == 2
            return vae.losses[1]

        # We compile full VAE in this function---instead of giving it to the
        # parent---so that our losses don't have to be passed back out. Encoder
        # and decoder can be compiled explicitly by user (but don't need to be
        # unless they're directly used for training).
        vae_opt = RMSprop(lr=INIT_LR)
        vae.compile(vae_opt, loss=null_loss, metrics=[kl_loss, nll_loss])

        return vae, encoder, decoder


class IdentityVAE(BasicVAE):
    def __init__(self, ident_dim, app_dim, *args, **kwargs):
        """ident_dim is dimension of identity variables, app_dim is dimension
        of appearance variables. Complete latent dim is ident_dim + app_dim."""
        total_dim = ident_dim + app_dim
        # we need to fit T-dimensional latents into a T = 4*4*D volume because
        # of way architecture works
        # TODO: make this less ugly by figuring out number of channels for last
        # conv layer automatically, then using a dense layer to flatten down
        # into desired shape
        assert (total_dim % 16) == 0, "Need channel count divisible by 4*4"
        latent_channels = total_dim // 16
        super().__init__(latent_channels, *args, **kwargs)
        self.ident_dim = ident_dim
        self.app_dim = app_dim

    def make_vae(self, encoder=None, decoder=None):
        assert K.backend() == 'tensorflow', \
            "Customer loss needs KERAS_BACKEND=tensorflow"

        if encoder is None:
            encoder = self._make_encoder()
        if decoder is None:
            decoder = self._make_decoder()

        ident_in = Input(shape=(1, ), dtype=tf.int32, name='identities')
        ident_flat = Lambda(K.flatten)(ident_in)

        encoder_in = Input(
            shape=(self.image_size, self.image_size, self.colspace_dim),
            name='images')
        # get all latent distributions from encoder
        all_z_mean, all_z_var = encoder(encoder_in)

        # now we can make actual loss
        # first few dims are identity, rest are appearance
        ident_dim = self.ident_dim
        ident_z_mean = Lambda(lambda v: v[:, :ident_dim])(all_z_mean)
        ident_z_var = Lambda(lambda v: v[:, :ident_dim])(all_z_var)
        app_z_mean = Lambda(lambda v: v[:, ident_dim:])(all_z_mean)
        app_z_var = Lambda(lambda v: v[:, ident_dim:])(all_z_var)

        def merge_idents(layers):
            """Averages means and variances for each person, then replaces
            those individual means/variances with their averaged equivalents.
            Needs to be called once for mean and once for variance."""
            id_params, idents = layers

            idents = K.cast(idents, tf.int32)
            seg_params = tf.segment_mean(id_params, idents)
            comb_params = K.gather(seg_params, idents)

            return comb_params

        # appearance gets unit KL
        app_z_mean, app_z_var = PureUnitKL()([app_z_mean, app_z_var])

        # identity gets KL divergence from mean distribution
        comb_ident_z_mean = Lambda(
            merge_idents,
            output_shape=(ident_dim, ))([ident_z_mean, ident_flat])
        comb_ident_z_var = Lambda(
            merge_idents,
            output_shape=(ident_dim, ))([ident_z_var, ident_flat])
        ident_z_mean, ident_z_var, comb_ident_z_mean, comb_ident_z_var \
            = PurePairKL()([ident_z_mean,      ident_z_var,
                            comb_ident_z_mean, comb_ident_z_var])

        # put the variables back together again, then pass to sampler
        # this ensures they're not orphans in the Keras graph
        reconst_z_mean = Concatenate()([comb_ident_z_mean, app_z_mean])
        reconst_z_var = Concatenate()([comb_ident_z_var, app_z_var])

        # sample latent variables
        reconst_z_std = Lambda(lambda t: K.sqrt(t))(reconst_z_var)

        def make_noise(layers):  # noqa
            """Applies reparamterisation trick to sample latents"""
            mu, sigma = layers
            noise = K.random_normal(shape=K.shape(sigma), mean=0., stddev=1.)
            return noise * sigma + mu

        latent = Lambda(
            make_noise,
            name='make_noise',
            output_shape=(self.latent_size**2 * self.latent_dim, ))(
                # must pass in STANDARD DEVIATION, NOT variance
                [reconst_z_mean, reconst_z_std])

        # finally get an image back from our decoder
        decoder_mean, decoder_var = decoder(latent)
        decoder_mean, decoder_var, _ = PureGaussNLL()(
            [decoder_mean, decoder_var, encoder_in])

        vae = Model(
            inputs=[encoder_in, ident_in], outputs=[decoder_mean], name='vae')

        def null_loss(x, x_hat):
            return K.sum(K.zeros_like(K.batch_flatten(x)), axis=-1)

        def app_kl(*args):
            assert len(vae.losses) == 3
            return vae.losses[0]

        def ident_kl(*args):
            assert len(vae.losses) == 3
            return vae.losses[1]

        def nll(*args):
            assert len(vae.losses) == 3
            return vae.losses[2]

        vae_opt = RMSprop(lr=INIT_LR)
        vae.compile(vae_opt, loss=null_loss, metrics=[app_kl, ident_kl, nll])

        return vae, encoder, decoder


class VAEMatcher(Recogniser):
    def __init__(self, encoder, decoder, use_kld=False, truncate_to=None):
        _, h, w, c = encoder.input_shape
        assert h == w, \
            "Square input assumed, got %dx%d (wxh)" % (w, h)
        pipeline = [ResizedSquare(w), Floating()]
        if c == 1:
            pipeline.append(Greyscale(trailing_dim=True))
        else:
            pipeline.append(RGB())
        super().__init__(pipeline=pipeline)
        self.encoder = encoder
        self.decoder = decoder
        self.use_kld = use_kld
        self.truncate_to = truncate_to

    def preproc_face_descriptor(self, preproc_image):
        preproc_image = np.asarray(preproc_image)
        means, vars = self.encoder.predict(
            preproc_image[None, ...], batch_size=1)
        mean, = means
        var, = vars
        if self.truncate_to is not None:
            mean = mean[:self.truncate_to]
            var = var[:self.truncate_to]
        return (mean, var)

    def descriptor_similarity(self, desc1, desc2):
        mean1, var1 = desc1
        mean2, var2 = desc2
        # On LFW, Gaussian KLD does fractionally worse than mean-mean distance.
        # Interesting.
        if self.use_kld:
            return gaussian_kld_pseudometric(mean1, var1, mean2, var2)
        return np.linalg.norm(mean1 - mean2)

    def reconstruct_descriptor(self, desc):
        """Turn a latent representation back into a face."""
        mean, var = map(np.asarray, desc)
        output = self.decoder.predict(mean[None, ...])[0][0]
        return clamp01(output)


class ImageHandler:
    """Convenience class to convert from RGB to chosen colour space and back
    again. Also handles resizing."""

    def __init__(self, colspace, size):
        self.colspace = colspace
        self.size = size
        if colspace == 'rgb':
            self.colspace_dim = 3
        elif colspace == 'grey':
            self.colspace_dim = 1
        else:
            raise ValueError('Unknown colour space "%s"' % colspace)

    def load_as_rgb(self, path):
        img = load_img(
            path, grayscale=False, target_size=(self.size, self.size))
        # move into [0, 1]
        rgb = img_to_array(img).astype(K.floatx()) / 255
        return rgb

    def load_as_colspace(self, path):
        rgb = self.load_as_rgb(path)
        return self.to_colspace(rgb)

    def to_colspace(self, rgb_img):
        if self.colspace == 'rgb':
            return rgb_img
        if self.colspace == 'grey':
            luminance = rgb2gray(rgb_img)
            # add back in third channel
            return luminance[..., None]
        assert False, "should never reach here"

    def to_rgb(self, colspace_img):
        if self.colspace == 'rgb':
            return colspace_img
        if self.colspace == 'grey':
            assert colspace_img.ndim == 3 and colspace_img.shape[-1] == 1, \
                "Need single-channel 3D tensor, got shape %s" \
                % (colspace_img.shape,)
            return gray2rgb(colspace_img[..., 0])
        assert False, "should never reach here"
