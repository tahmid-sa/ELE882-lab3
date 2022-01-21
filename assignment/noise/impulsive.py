import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import median
from assignment import sharpening


def denoise(img):

    if img.ndim == 3:
        img = rgb2gray(img)

    denoised = np.zeros(np.shape(img))
    N, M = img.shape

    for n in range(N):
        for m in range(M):
            if (n == 0 or m == 0 or n == N or m == M):
                denoised[n][m] = img[n][m]
            else:
                denoised[n][m] = median(img[n-1: n+2, m-1:m+2])

    return sharpening.laplacian(np.array(denoised, dtype=float), 0.5)
