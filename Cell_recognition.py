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

ImageRoot = "/home/ank/Documents/var/"


def get_image(source):
    return color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot + source)))


def gaussian_blur(image, blur=1):
    return gaussian_filter(image, blur)


def get_dilatation(image):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    return dilated, image-dilated


def get_sobel(image):
    return sobel(image)


def get_saturated(image, threshold):
    image2 = image.copy()
    image2[image < threshold*np.max(image)] = 0
    image2[image > threshold*np.max(image)] = 1
    return image2


def get_canny_edges(image, sigma=1):
    return filter.canny(image,sigma=sigma)


def get_skeleton(image):
    skel, distance = medial_axis(image, return_distance = True)
    dist_on_skel = distance * skel
    return dist_on_skel


def get_chain_results(image, action_chain, optional_inputs):
    retlist = [image]
    for action, option in zip(action_chain,optional_inputs):
        if option == None:
            retlist.append(action(retlist[-1]))
        else:
            retlist.append(action(retlist[-1], option))
    return retlist, ['Original']+[f.__name__ for f in action_chain]


def get_superposition(image1, image2):
    return [image1, image2]


def render_chain(chain_res_images, chain_action_names, shape = 120):
    max_images = int(str(shape)[0])*int(str(shape)[1])
    if len(chain_res_images) > max_images or len(chain_action_names) > max_images:
        raise Exception("Too many images for the configuration")
    pos = shape
    plt.figure()
    for image, name in zip(chain_res_images,chain_action_names):
        pos+=1
        plt.subplot(pos)
        plt.title(name)
        plt.imshow(image, cmap='gray')
        # TODO: add a case for superposition later
        plt.colorbar()
    plt.show()


# Load an image as a floating-point grayscale
image = get_image('Test2.png')
act_chain=[get_canny_edges]

resimgs, resnames = get_chain_results(image,act_chain,[None])
render_chain(resimgs, resnames)

# seed = np.copy(image2)
# seed[1:-1, 1:-1] = image2.min()
# mask = image2
#
# dilated = reconstruction(seed, mask, method='dilation')

# edge_sobel = sobel(image)
#
# Saturated_Sobel = edge_sobel.copy()
# Saturated_Sobel[edge_sobel < 0.20*np.max(edge_sobel)] = 0
# Saturated_Sobel[edge_sobel > 0.20*np.max(edge_sobel)] = 1
# Saturated_Sobel[edge_sobel > 0.33*np.max(edge_sobel)]= 0.5*np.max(edge_sobel)

# edges1 = filter.canny(image, sigma=1)
# r_org = dilated
# edges2 = filter.canny(r_org, sigma=0.5)


# selem = disk(1)
# closed = closing(Saturated_Sobel, selem)
#
# skel, distance = medial_axis(Saturated_Sobel, return_distance=True)
# dist_on_skel = dis
# dist_on_skel2 = distance * skel
#
# skel2, distance2 = medial_axis(closed, return_distance=True)
# dist_on_skel2 = distance2 * skel2


plt.figure()
plt.subplot(231)
plt.title('Original')
plt.imshow(image, cmap = 'gray')
plt.colorbar()


plt.subplot(232)
plt.title('Original+Canny')
plt.imshow(edge_sobel, cmap = 'gray')
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
plt.imshow(dist_on_skel2, cmap=plt.cm.spectral, interpolation='nearest')
plt.colorbar()

plt.subplot(236)
plt.title('Image + skeleton')
plt.imshow(closed, cmap = 'gray')
plt.imshow(dist_on_skel2, cmap=plt.cm.spectral, interpolation='nearest', alpha=0.5)
plt.colorbar()

plt.show()