from skimage.color import rgb2gray
from assignment import filters, sharpening


def denoise(img):

    if img.ndim == 3:
        img = rgb2gray(img)

    return sharpening.laplacian(filters.moving_average(img, 5), 1.2)
