import imageio # for image input/output
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter

# Imageio provides a range of [example images](https://imageio.readthedocs.io/en/latest/standardimages.html).
img = imageio.imread('imageio:astronaut.png')
print(type(img), img.shape)

# imageio.core.util.Array can compute with numpy.ndarray
a = np.full((512, 512, 3), 0.5)
tint = (img * a).astype(np.uint8)

# write and image to the specific uri
# imageio.imwrite('./astronaut_tint.png', tint)

img = img[::128, ::128, 1] # sampling the image

d = squareform(pdist(img, 'euclidean')) # pdist computes the dist between all pairs of points
print(d) # d[i, j] is the Euclidean dist between img[i, :] and img[j, :]

filtered_img = gaussian_filter(img, sigma=1, mode='nearest')
print(img)
print(filtered_img)

# fft
img_fft = np.fft.fft2(img)
print(img_fft)
img_rvt = np.fft.ifft2(img_fft)
print(img_rvt)