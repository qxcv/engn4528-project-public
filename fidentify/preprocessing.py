"""Code for preprocessing images before they're passed to various detectors,
etc."""

from fidentify.utils import im2double, rgb2grey, imresize_opt, grey2rgb


def apply_pipeline(pipeline, image):
    for stage in pipeline:
        image = stage(image)
    return image


class Floating:
    """Convert byte data to floats in [0, 1]"""
    def __call__(self, image):
        return im2double(image)


class Greyscale:
    """Convert RGB or greyscale image to greyscale"""
    def __init__(self, trailing_dim=False):
        # If self.trailing_dim, then we make sure to return a 3D image instead
        # of a 2D one. This is useful for DNN libraries which expect the last
        # dimension in a batch to select a channel.
        self.trailing_dim = trailing_dim

    def __call__(self, image):
        assert 2 <= image.ndim <= 3, \
            "Expecting 2D or 3D image, got shape %s" % (image.shape,)
        if image.ndim == 2:
            rv = rgb2grey(image)
        else:
            rv = image
        if self.trailing_dim and rv.ndim < 3:
            # add a singleton dimension on the end
            rv = rv[..., None]
        return rv


class RGB:
    """Convert greyscale or RGB image to RGB"""
    def __call__(self, image):
        assert 2 <= image.ndim <= 3, \
            "Expecting 2D or 3D image, got shape %s" % (image.shape,)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        return grey2rgb(image)


class ResizedSquare:
    def __init__(self, size):
        # need single int giving length of one side of square
        assert isinstance(size, int)
        self.size = size

    def __call__(self, image):
        return imresize_opt(image, self.size)
