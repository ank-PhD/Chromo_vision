__author__ = 'ank'

import numpy as np
from matplotlib import pyplot as plt
import PIL
import itertools
from scipy import ndimage
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
from scipy import fftpack

ImageRoot = "/home/ank/Documents/var/"

# TODO: try a closing, white_tophat from scipy.ndimage
# TODO: sliding saturated_sobel - skeleton
#       In order to implement a filtering that is sufficiently precise
# TODO: try reverse 2D fourrier transform


# TODO: try to recover the center of the cell heterogeinties by closing a saturated sobel, then determine the distance
# Covered and delete the lines falling into the perimenter


# TODO: try to implement a lhassoing skeleton over Sobel, that keeps only the most early detected edges and then adds
# the newly detected edges only where it improves the path of the


# TODO: implement contour search

def get_image(source):
    return color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot + source)))


def get_gaussian_blur(image, blur=1):
    return gaussian_filter(image, blur)


def get_dilatation(image):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    return dilated, image-dilated


def get_sobel(image):
    return sobel(image)


def get_saturated(image, threshold=0.5):
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
            # TODO: add an option for superposition with an earlier result
            retlist.append(action(retlist[-1], option))
    return retlist, ['Original']+[f.__name__ for f in action_chain]


def get_superposition(image1, image2):
    return [image1, image2]


def get_reverted(image):
    return np.ones(image.shape)*np.max(image)-image


def get_dist_from_edges(image):
    return ndimage.distance_transform_edt(image)


def get_fine_dist_from_edges(image):
    return np.log10(ndimage.distance_transform_edt(image))


def get_watershed(image):
    distance = get_dist_from_edges(image)
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)),
                            labels = image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask = image)
    return labels

def get_fine_watershed(image):
    distance = get_fine_dist_from_edges(image)
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)),
                            labels = image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask = image)
    return labels

def get_connex(image):
    label_im, nb_labels = ndimage.label(image)
    print nb_labels
    return label_im

def get_double_roll(image):
    image2 = image.copy()
    image2 += np.roll(image, shift=1, axis=1)
    image2 += np.roll(image, shift=1, axis=0)
    image2[image2 > 0.5] = 1.0
    return image2

def get_quadruple_roll(image):
    image2 = image.copy()
    image2 += np.roll(image, shift=1, axis=1)
    image2 += np.roll(image, shift=1, axis=0)
    image2 += np.roll(image, shift=-1, axis=1)
    image2 += np.roll(image, shift=-1, axis=0)
    image2[image2 > 0.5] = 1.0
    return image2

def get_segmented_canny(image):
    # image is assumed to be canny
    return image*get_connex(get_double_roll(image))

def check_edge_unstability(image, wshed):
    re_wshed = wshed.copy()
    re_wshed[wshed < np.max(wshed)-0.001] = -1
    edge_contacts = get_quadruple_roll(image)
    re_im = edge_contacts*re_wshed

    if np.max(re_im)-np.min(re_im) > 0.99*(np.max(re_wshed)-np.min(re_wshed)):
        return True
    else:
        return False


def get_shuffle_unstable(image, wshed):
    """

    :param image: segmented canny
    :param wshed: non-median get watershed over inverted canny
    :return:
    :raise Exception:
    """
    image2 = image.copy()
    retimage = np.zeros(image.shape)
    while np.max(image2)>0.001:
        buff_im = np.zeros(image.shape)
        buff_im[image2 > (np.max(image2)-0.001) ] = 1
        if check_edge_unstability(buff_im, wshed):
            print 'True!'
            retimage += buff_im
        image2[image2 > (np.max(image2)-0.001) ] = 0
    return retimage

def get_non_median(image):
    image2 = image.copy()
    image2[image <> np.median(image)] = 1
    image2[image == np.median(image)] = 0
    return image2

def get_central(image, reduction = 10):
    l = image.shape
    idx1 = range(l[0]/2-l[0]/reduction,l[0]/2+l[0]/reduction)
    idx2 = range(l[1]/2-l[1]/reduction,l[1]/2+l[1]/reduction)
    res = image[idx1[0]:idx1[-1],idx2[0]:idx2[-1]]
    return res


def get_autocorr(image):
    return np.log10(np.abs(fftpack.fftshift(fftpack.fft2(image)))**2)


def get_blurred_canny(image):
    return get_double_roll(image)


def get_better_autocorr(image):
    return get_autocorr(get_blurred_canny(image))


def render_chain(chain_res_images, chain_action_names, shape = 120, cmaps = ['gray']):
    max_images = int(str(shape)[0])*int(str(shape)[1])
    if len(chain_res_images) > max_images or len(chain_action_names) > max_images:
        raise Exception("Too many images for the configuration")
    pos = shape
    plt.figure()
    for (image, name), clrmp  in zip(zip(chain_res_images,chain_action_names), itertools.cycle(cmaps)):
        pos+=1
        plt.subplot(pos)
        plt.title(name)
        plt.imshow(image, cmap=clrmp)
        # TODO: add a case for superposition options later on
        plt.colorbar()
    plt.show()


# Load an image as a floating-point grayscale
image = get_image('Test2.png')

act_chain = [get_canny_edges, get_reverted, get_watershed, get_non_median]
act_chain2 = [get_canny_edges, get_reverted, get_dist_from_edges]
act_chain3 = [get_canny_edges, get_autocorr, get_central]
act_chain4 = [get_canny_edges, get_reverted, get_fine_dist_from_edges]
act_chain5 = [get_canny_edges, get_segmented_canny]


colormap1 = ['gray', 'gray', 'gray', plt.cm.jet, plt.cm.jet]
colormap2 = ['gray', 'gray', 'gray', plt.cm.spectral]
colormap3 = ['gray', 'gray', plt.cm.jet, plt.cm.jet]
colormap4 = ['gray', 'gray', plt.cm.jet, plt.cm.jet]
colormap5 = ['gray', 'gray', plt.cm.jet, plt.cm.jet]


resimgs, resnames = get_chain_results(image, act_chain, [None, None, None, None])
render_chain(resimgs, resnames, 230, colormap1)
wshed = resimgs[-1]
canny = resimgs[1]


# plt.figure()
#
# plt.subplot(131)
# plt.title('Canny')
# plt.imshow(resimgs[2], cmap = 'gray')
# plt.colorbar()
#
# plt.subplot(132)
# plt.title('Segmentation')
# plt.imshow(resimgs[-1], cmap = plt.cm.jet)
# plt.colorbar()
#
# plt.subplot(133)
# plt.title('Seg+Canny')
# plt.imshow(resimgs[-1], cmap = plt.cm.jet)
# plt.imshow(resimgs[2], cmap = 'gray', alpha=0.5)
# plt.colorbar()
#
# plt.show()



# resimgs, resnames = get_chain_results(image, act_chain2, [None, None, None])
# render_chain(resimgs, resnames, 230, colormap2)

# resimgs, resnames = get_chain_results(image, act_chain3, [None, None, None, None])
# render_chain(resimgs, resnames, 230, colormap3)

# resimgs, resnames = get_chain_results(image, act_chain4, [None, None, None, None])
# render_chain(resimgs, resnames, 230, colormap4)

resimgs, resnames = get_chain_results(image, act_chain5, [None, None, None, None])
render_chain(resimgs, resnames, 230, colormap5)
seg_canny = resimgs[-1]


shuffle_unstable = get_shuffle_unstable(seg_canny, wshed)

plt.figure()

plt.subplot(231)
plt.title('Original')
plt.imshow(image, cmap = 'gray')
plt.colorbar()

plt.subplot(232)
plt.title('Canny')
plt.imshow(canny, cmap = plt.cm.jet)
plt.colorbar()

plt.subplot(233)
plt.title('Seg-Watershed')
plt.imshow(wshed, cmap = plt.cm.jet)
plt.colorbar()

plt.subplot(234)
plt.title('Seg-canny')
plt.imshow(seg_canny, cmap = plt.cm.jet)
plt.colorbar()

plt.subplot(235)
plt.title('Blurred shuffle-unstable')
plt.imshow(shuffle_unstable, cmap = 'gray')
plt.colorbar()

plt.subplot(236)
plt.title('Autocorr')
plt.imshow(get_better_autocorr(shuffle_unstable), cmap = plt.cm.jet)
plt.colorbar()

plt.show()

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
#
#
# plt.figure()
# plt.subplot(231)
# plt.title('Original')
# plt.imshow(image, cmap = 'gray')
# plt.colorbar()
#
#
# plt.subplot(232)
# plt.title('Original+Canny')
# plt.imshow(edge_sobel, cmap = 'gray')
# plt.colorbar()
#
# plt.subplot(233)
# plt.title('Canny ')
# plt.imshow(edges1, cmap = 'gray')
# plt.colorbar()
#
#
# plt.subplot(234)
# plt.title('Saturated Sobel')
# plt.imshow(Saturated_Sobel, cmap = 'gray')
# plt.colorbar()
#
# plt.subplot(235)
# plt.title('Skeleton over Sobel')
# plt.imshow(dist_on_skel2, cmap=plt.cm.spectral, interpolation='nearest')
# plt.colorbar()
#
# plt.subplot(236)
# plt.title('Image + skeleton')
# plt.imshow(closed, cmap = 'gray')
# plt.imshow(dist_on_skel2, cmap=plt.cm.spectral, interpolation='nearest', alpha=0.5)
# plt.colorbar()
#
# plt.show()