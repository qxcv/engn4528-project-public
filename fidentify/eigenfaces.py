"""Implementation of Turk & Pentland's Eigenfaces technique."""

import argparse

from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

from fidentify.preprocessing import Floating, Greyscale, ResizedSquare
from fidentify.utils import get_paths, clamp01, im2double, rgb2grey, \
    imresize_opt
from fidentify.facedb import Recogniser


def img2vec(img):
    """Convert a 2D image into a flat vector."""
    assert 2 <= img.ndim <= 3, "Image must be 2D grey or RGB"
    assert img.shape[0] == img.shape[1], "Image must be square"
    img = rgb2grey(im2double(img))
    assert img.ndim == 2, \
        "Need grey image; got array with shape {}".format(img.shape)
    return img.flatten()


def vec2img(vec):
    """Roughly does the inverse of img2vec (can't get back your colour channels
    though---sorry!)"""
    assert vec.ndim == 1, "Can only convert vectors"
    dim = int(np.sqrt(vec.size))
    assert dim**2 == vec.size, "Must be vector for square image"
    # we'll just return floating point vector (who cares?)
    return vec.reshape((dim, dim))


def load_img_as_vec(image_path, new_size=None):
    img = imread(image_path)
    img_right_size = imresize_opt(img, new_size)
    return img2vec(img_right_size)


def make_image_mat(image_paths, new_size=None):
    """Convert a list of N paths to image files into an N*(W*H) matrix, where
    (W, H) denotes width and height of the images (must be constant across all
    images)."""
    num_images = len(image_paths)

    # load first image to get representation dimensionality
    first_path = image_paths[0]
    rest = image_paths[1:]
    first_vec = load_img_as_vec(first_path, new_size)
    vec_dim = first_vec.size
    return_mat = np.empty((num_images, vec_dim))
    return_mat[0, :] = first_vec

    for idx, other_path in enumerate(rest, start=1):
        return_mat[idx, :] = load_img_as_vec(other_path, new_size)

    return return_mat


def get_eigenvectors(X, max_vecs=None):
    """Given an image matrix created by make_image_mat (for instance), return a
    matrix containing up to max_vecs eigenvectors of the covariance matrix (one
    per column), and a mean to subtract out when projecting onto the covariance
    matrix."""
    # mean subtraction first
    mean = np.mean(X, axis=0)
    U = X - mean[None, ...]

    # we want eigenvectors of (1/n)UU^T, but can get eigenvectors of U^T U
    # and premultiply by U instead (1/n dropped because magnitude irrelevant).
    eigvals, eigmat_ld = np.linalg.eigh(np.inner(U, U))

    # return all eigenvectors with nonzero eigenvalues, up to max_vecs
    num_nonzero = np.sum(np.abs(eigvals) > 1e-8)
    if max_vecs is None or num_nonzero < max_vecs:
        max_vecs = num_nonzero

    # eigvals is sorted ascending, and vectors in eigmat_ld correspond to those
    # ascending values. This trims to "best" :max_vecs eigenvectors.
    eigmat_ld = eigmat_ld[:, ::-1][:, :max_vecs]

    # now we can get eigenvectors in the original space, normalise, and check
    # dims
    eigmat = np.dot(U.T, eigmat_ld)
    norms = np.linalg.norm(eigmat, axis=0, keepdims=True)
    assert np.all(np.abs(norms) > 1e-8), "Some eigenvectors are near zero (?!)"
    eigmat /= norms
    assert eigmat.shape == (X.shape[1], max_vecs)
    assert mean.size == eigmat.shape[0]

    return eigmat, mean


def face_descriptor(eigmat, mean, img_vecs):
    """Convert img_vec into low-dimensional representation by projecting onto
    columns of eigmat."""
    assert mean.ndim == 1, "Need 1D mean"
    assert 1 <= img_vecs.ndim <= 2
    added_leader = False
    if img_vecs.ndim == 1:
        added_leader = True
        # insert extra dimension to make it 2D
        img_vecs = img_vecs[None, ...]

    projected = np.dot(img_vecs - mean[None, ...], eigmat)

    if added_leader:
        # take out extra dimension
        assert projected.shape[0] == 1, \
            "should still have a singleton leading dim (?)"
        projected = projected[0]

    return projected


def reconstruct_face_vec(eigmat, mean, descriptor):
    assert descriptor.shape == (eigmat.shape[1], )
    assert mean.ndim == 1
    actual_vec = np.dot(eigmat, descriptor[:, None]) + mean[:, None]
    return actual_vec.flatten()


class EigenfaceMatcher(Recogniser):
    """Wrapper class to store eigenfaces and projected weights for known
    individuals."""

    def __init__(self, eigenfaces, mean, dim=128):
        """Initialise class from a trained eigenface matrix (``eigenfaces``)
        and image vector mean."""
        super().__init__(pipeline=[
            Floating(), Greyscale(), ResizedSquare(dim)
        ])
        self.eigenfaces = eigenfaces
        self.mean = mean
        self.dim = 128

    def preproc_face_descriptor(self, preproc_picture):
        pic_vec = img2vec(preproc_picture)
        return face_descriptor(self.eigenfaces, self.mean, pic_vec)

    def descriptor_similarity(self, desc1, desc2):
        return np.linalg.norm(desc1 - desc2)

    def reconstruct_descriptor(self, desc):
        face_vec = reconstruct_face_vec(self.eigenfaces, self.mean, desc)
        return vec2img(face_vec)


# Here's a bit of demo code so that you can see what eigenfaces look like. To
# run it, cd into the root directory of this reposistory (the one containing
# the data/, fidentify/, etc. directories) and do "python -m
# fidentify.eigenfaces <image dir>" (where image_dir is a path to a bunch of
# pre-cropped face images).

_parser = argparse.ArgumentParser()
_parser.add_argument('image_dir', help='directory to look for .pngs in')

if __name__ == '__main__':
    args = _parser.parse_args()
    image_paths = get_paths(args.image_dir)
    image_mat = make_image_mat(image_paths, 128)
    eigmat, mean = get_eigenvectors(image_mat)

    # Plot some of the eigenfaces
    grid_width = 3
    num_vecs = eigmat.shape[1]
    plt.figure()
    for i in range(min(grid_width * grid_width, num_vecs)):
        plt.subplot(grid_width, grid_width, i + 1)
        vec = eigmat[:, i]
        img = vec2img(vec)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle('Some creepy eigenfaces')
    plt.show(block=False)

    # Now pick a face at random and project it into FACE SPACE with variable
    # number of faces.
    img_vec = image_mat[np.random.randint(image_mat.shape[0])]
    img = vec2img(img_vec)
    face_progression = [1, 3, 7, 20, num_vecs]
    face_progression = [n for n in face_progression if n <= num_vecs]
    n_progs = len(face_progression)
    plt.figure()
    plt.subplot(1, n_progs + 1, 1)
    plt.imshow(clamp01(img), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    for i, nfaces in enumerate(face_progression, start=2):
        plt.subplot(1, n_progs + 1, i)
        plt.title('%d eigenfaces' % nfaces)
        trunc_eigmat = eigmat[:, :nfaces]
        descriptor = face_descriptor(trunc_eigmat, mean, img_vec)
        reconstructed = reconstruct_face_vec(eigmat[:, :nfaces], mean,
                                             descriptor)
        recon_img = vec2img(reconstructed)
        plt.imshow(clamp01(recon_img), cmap='gray')
        plt.axis('off')
    plt.suptitle('Reconstruction with varying numbers of eigenfaces')
    plt.show()
