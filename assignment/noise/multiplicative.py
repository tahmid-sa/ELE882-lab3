from skimage.color import rgb2gray
from assignment import filters, sharpening


def denoise(img):

    if img.ndim == 3:
        img = rgb2gray(img)

    return sharpening.unsharp_masking(filters.gaussian(img, 0.67), 1.2, 0.67)
