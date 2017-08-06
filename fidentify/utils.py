"""Miscellaneous code, not specific to any one component."""

import os

import numpy as np
from scipy.misc import imresize


def im2double(img):
    """Converts integer types, etc., to floats."""
    img = np.asarray(img)
    old_dt = img.dtype
    if np.issubdtype(old_dt, np.integer):
        # convert to [0, 1]
        iinfo = np.iinfo(old_dt)
        denom = float(iinfo.max - iinfo.min)
        img = (img.astype('float32') - iinfo.min) / denom
    else:
        assert np.issubdtype(old_dt, np.float), "Input must be float or int"
    return img


def rgb2grey(img):
    """Converts H*W*3 RGB image to H*W greyscale one."""
    assert 2 <= img.ndim <= 3, "Image must be 2D (grey) or 3D (RGB)"
    if img.ndim == 2:
        return img
    # convert from RGB to grey (luminance) using ITU BT.709 formula
    rgb = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    # we might have accidentally converted to float, so we convert back
    return rgb.astype(img.dtype)


def imresize_opt(img, new_size=None):
    if new_size is None:
        return img
    assert img.shape[0] == img.shape[1], \
        "Can only resize squares, got shape %s" % (img.shape,)
    return imresize(img, (new_size, new_size))


def grey2rgb(img):
    """Converts H*W greyscale image to H*W*3 RGB one."""
    assert 2 <= img.ndim <= 3, "Image must be 2D (grey) or 3D (RGB)"
    if img.ndim == 3:
        return img
    # just duplicate channels
    return np.broadcast_to(img[..., None], img.shape[:2] + (3, ))


def gaussian_kld(mean1, var1, mean2, var2):
    """KL divergence between two Gaussians (Numpy version)."""
    noise_dim = mean1.size
    sq_mean_diff = (mean1 - mean2)**2
    first_vec = (var1 + sq_mean_diff) / var2
    out_vec = first_vec + np.log(var2) - np.log(var1)
    return 0.5 * (out_vec.sum() - noise_dim)


def gaussian_kld_pseudometric(mean1, var1, mean2, var2):
    """Approximate distance between two Gaussians. Obtained by taking KL
    divergence in each direction and averaging. Not sure whether it's actually
    a metric in the mathematical sense."""
    kld1 = gaussian_kld(mean1, var1, mean2, var2)
    kld2 = gaussian_kld(mean2, var2, mean1, var1)
    return (kld1 + kld2) / 2.0


def get_paths(root_dir, allowed_exts={'.png', '.jpg', '.gif'}):
    """Recursively walk root_dir to find paths to all files with an allowed
    extension (e.g. to find all images in a directory)."""
    return_paths = []
    for root, dirs, files in os.walk(root_dir):
        valid_files = [
            f for f in files if os.path.splitext(f)[1].lower() in allowed_exts
        ]
        return_paths.extend([os.path.join(root, f) for f in valid_files])
    return return_paths


def clamp01(arr):
    """Clamp values to [0, 1]"""
    return np.minimum(np.maximum(arr, 0), 1)


def crop(image, x, y, w, h, pad_mode='edge'):
    """Crop image to box, padding with some sane method"""
    image = np.asarray(image)
    assert 2 <= image.ndim <= 3, "Image must have HxW dims or HxWxC dims"
    assert h > 0 and w > 0, "Can't crop down to nothing"

    # figure out which part of image we need to keep
    im_h, im_w = image.shape[:2]
    x_start = max(x, 0)
    # "stop" means "final included position, plus one"
    x_stop = min(x + w, im_w)
    y_start = max(y, 0)
    y_stop = min(y + h, im_h)

    # check if out of bounds (will result in crop of size zero)
    oob_y = y_stop <= 0 or y_start >= im_h
    oob_x = x_stop <= 0 or x_start >= im_w
    if oob_y or oob_x:
        # return mean pixel if it is out of bounds
        mean_px = np.mean(image, axis=0).mean(axis=0)
        rv = np.broadcast_to(
            mean_px[None, None, ...], (h, w) + image.shape[2:], subok=True)
    else:
        crop = image[y_start:y_stop, x_start:x_stop]
        # pad the array in each direction
        pre_y = y_start - y
        post_y = y + h - y_stop
        pre_x = x_start - x
        post_x = x + w - x_stop
        pads = [(pre_y, post_y), (pre_x, post_x)]
        if image.ndim > 2:
            pads.append((0, 0))
        rv = np.pad(crop, pads, pad_mode)

    # basic shape sanity checks
    assert rv.ndim == image.ndim, rv.shape
    assert rv.shape[:2] == (h, w), rv.shape
    assert rv.shape[2:] == image.shape[2:], rv.shape

    return rv


def path_from_root(root, relpath):
    """Resolve a path relative to root without going above it (avoids directory
    traversal)."""
    real_root = os.path.abspath(root) + os.path.sep
    out_path = os.path.abspath(os.path.join(real_root, relpath))
    if not out_path.startswith(real_root):
        raise ValueError('Path "%s" is above root "%s"!' % (relpath, root))
    return out_path


def data_path(filename=None):
    """Return path to data directory (it's one level above the fidentify/
    directory that this file is stored in). Can append a relative filename to
    it for convenience (e.g. data_path() -> '/path/to/data', data_path('foo')
    -> '/path/to/data/foo')."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(this_dir, '../data/'))
    if filename is None:
        return data_dir
    return path_from_root(data_dir, filename)
