__author__ = 'ank'

import numpy as np
from matplotlib import pyplot as plt
import PIL
from skimage import img_as_float, color
from skimage.filter import roberts, sobel

from skimage import filter, measure
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter

from skimage.morphology import medial_axis

from skimage.morphology import watershed
from skimage.feature import peak_local_max

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk

# TODO: try a closing, white_tophat from scipy.ndimage
# TODO: sliding saturated_sobel - skeleton
#       In order to implement a filtering that is sufficiently precise
# TODO: try reverse 2D fourrier transform

ImageRoot = "/home/ank/Documents/var"
gray = PIL.Image.open(ImageRoot + '/Test1.png')

# Load an image as a floating-point grayscale
image = color.rgb2gray(img_as_float(gray))

# image = gaussian_filter(image, 1)
image2=image

seed = np.copy(image2)
seed[1:-1, 1:-1] = image2.min()
mask = image2

dilated = reconstruction(seed, mask, method='dilation')

edge_sobel = sobel(image)

Saturated_Sobel = edge_sobel.copy()
Saturated_Sobel[edge_sobel < 0.25*np.max(edge_sobel)] = 0
Saturated_Sobel[edge_sobel > 0.25*np.max(edge_sobel)] = 1
# Saturated_Sobel[edge_sobel > 0.33*np.max(edge_sobel)]= 0.5*np.max(edge_sobel)

edges1 = filter.canny(image, sigma=1)
r_org = dilated
edges2 = filter.canny(r_org, sigma=0.5)


selem = disk(1)
closed = closing(Saturated_Sobel, selem)

skel, distance = medial_axis(Saturated_Sobel, return_distance=True)
dist_on_skel = distance * skel

skel2, distance2 = medial_axis(closed, return_distance=True)
dist_on_skel2 = distance2 * skel2


plt.figure()
plt.subplot(231)
plt.title('Original')
plt.imshow(image, cmap = 'gray')
plt.colorbar()


plt.subplot(232)
plt.title('Original+Canny')
plt.imshow(image, cmap = 'gray')
plt.imshow(edges1, cmap=plt.cm.spectral, interpolation='nearest', alpha=0.5)
plt.colorbar()

plt.subplot(233)
plt.title('Canny ')
plt.imshow(edges1, cmap = 'gray')
plt.colorbar()


plt.subplot(234)
plt.title('Saturated Sobel')
plt.imshow(Saturated_Sobel, cmap = 'gray')
plt.colorbar()

plt.subplot(235)
plt.title('Skeleton over Sobel')
plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
plt.colorbar()

plt.subplot(236)
plt.title('Image + skeleton')
plt.imshow(image, cmap = 'gray')
plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest', alpha=0.5)
plt.colorbar()

plt.show()