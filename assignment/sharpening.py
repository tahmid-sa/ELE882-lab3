from . import filters
import numpy as np


def unsharp_masking(img, gain, sigma):

    if gain < 0:
        raise ValueError("The gain is negative.")

    return np.array((img + gain * (img - (filters.gaussian(img, sigma) * img))).clip(0, 1))


def laplacian(img, gain):

    if gain < 0:
        raise ValueError("The gain is negative.")

    return np.array((img + gain * filters.laplacian(img)).clip(0, 1))
