import numpy as np
import math
from scipy import ndimage


def _convolve(img, kernel):
    '''Convenience method around ndimage.convolve.

    This calls ndimage.convolve with the boundary setting set to 'nearest'.  It
    also performs some checks on the input image.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    kernel : numpy.ndarray
        filter kernel

    Returns
    -------
    numpy.ndarray
        filter image

    Raises
    ------
    ValueError
        if the image is not greyscale
    TypeError
        if the image or filter kernel are not a floating point type
    '''
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    if img.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Image must be floating point.')

    if kernel.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Filter kernel must be floating point.')

    return ndimage.convolve(img, kernel, mode='nearest')


def moving_average(img, width):

    if img.dtype == np.ubyte:
        raise TypeError('Image must be floating point.')
    if width <= 0 or width % 2 == 0:
        raise ValueError("The width is even, zero or negative.")
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    return _convolve(img, np.ones((width, width), dtype=float) / width**2)


def gaussian(img, sigma):

    if sigma <= 0:
        raise ValueError("The sigma value is negative.")
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    sigma = float(sigma)

    if sigma > 0 and 6 * sigma >= 1:
        N = math.ceil(6 * sigma)
        N = max(N, 3)
        if N % 2 == 0:
            N = N + 1

        tmp_one = 1 / (math.sqrt(2 * math.pi * (sigma**2)))
        tmp_two = np.ones((N, N), dtype=float)

        for i in range(N):
            e = ((i - (0.5 * (N - 1)))**2) / (2 * (sigma**2))
            tmp_two[i] = tmp_one * math.exp(-e)

        tmp_two = np.array(np.transpose(tmp_two) * tmp_two, dtype=float)
        tmp_two = tmp_two / np.sum(tmp_two)

        return _convolve(img, tmp_two)
    else:
        raise ValueError("Sigma is a negative.")


def laplacian(img):

    if img.dtype == np.ubyte:
        raise TypeError('Image must be floating point.')
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    return _convolve(img, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
