__author__ = 'ank'

import numpy as np
import matplotlib.pyplot as plt
import PIL
from time import time
import mdp
from itertools import product
from skimage.segmentation import random_walker, mark_boundaries


# TODO: try inducing a finer separator set
# TODO: try using the data for the edge detection by assigning a null label to all the edge detectors, so that the data is recovered by diffusion from a skeleton


def import_image():
    ImageRoot = '/home/ank/projects_files/2014/Image_recognition/'
    suffix = '/img_11.tif'
    # suffix = '/img_4jpg.jpeg'
    col = PIL.Image.open(ImageRoot + suffix)
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw = bw - np.min(bw)
    bw = bw.astype(np.float64)/float(np.max(bw))
    return bw


def gabor(bw_image, freq, scale, scale_distortion=1.):

    # gabor filter normalization with respect to the surface convolution
    def check_integral(gabor_filter):
        ones = np.ones(gabor_filter.shape)
        avg = np.average(ones*gabor_filter)
        return gabor_filter-avg

    quality = 16
    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    phis = [pi/2, pi]
    size = (10, 10)
    sgm = (5*scale, 3*scale*scale_distortion)

    nfilters = len(orientations)*len(phis)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, (alpha, phi) in enumerate(product(orientations, phis)):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        arr = check_integral(arr)
        gabors[i,:,:] = arr
    #     plt.subplot(6,6,i+1)
    #     plt.title('%s, %s, %s, %s'%('{0:.2f}'.format(alpha), '{0:.2f}'.format(phi), freq, sgm))
    #     plt.imshow(arr, cmap = 'gray', interpolation='nearest')
    # plt.show()
    node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill', fillvalue=0, output_2d=False)
    cim = node.execute(bw[np.newaxis, :, :])
    sum1 = np.zeros(cim[0, 0,:,:].shape)
    col1 = np.zeros((cim[0,:,:,:].shape[0]/2, cim[0,:,:,:].shape[1], cim[0,:,:,:].shape[2]))
    sum2 = np.zeros(cim[0, 0,:,:].shape)
    col2 = np.zeros((cim[0,:,:,:].shape[0]/2, cim[0,:,:,:].shape[1], cim[0,:,:,:].shape[2]))
    for i in range(0, nfilters):
        pr_cim = cim[0,i,:,:]
        if i%2 == 0:
            sum1 = sum1 + np.abs(pr_cim)
            col1[i/2,:,:] = pr_cim
        else:
            sum2 = sum2 - pr_cim
            col2[i/2,:,:] = np.abs(pr_cim)

    sum2[sum2>0] = sum2[sum2>0]/np.max(sum2)
    sum2[sum2<0] = -sum2[sum2<0]/np.min(sum2)

    return sum1/np.max(sum1), sum2, col1, col2, # two last ones just in case.


def cluster_by_diffusion(data):
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < -0.00] = 1
    markers[data > 0.03] = 2
    labels2 = random_walker(data, markers, beta=10, mode='bf')

    return labels2


if __name__ == "__main__":
    start = time()
    bw = import_image()
    sum1, sum2, _, _= gabor(bw, 1/4., 0.5)

    # The separator is acting here:
    sum10, sum20, _, _ = gabor(bw, 1/4., 0.25, 3)
    sum20[sum20>-0.3]=0
    sum2  = sum2 + sum20

    sum22 = np.copy(sum2)
    sum22[sum2<0] = 0
    d_c = cluster_by_diffusion(sum2)


    # align to same shape
    rebw = bw[4:, :][:,4:]

    plt.subplot(2,2,1)
    plt.title('Original image')
    plt.imshow(bw, cmap = 'gray', interpolation='nearest')

    plt.subplot(2,2,2)
    plt.title('Gabor - line detector')
    plt.imshow(sum2, cmap = 'gray', interpolation='nearest')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('Gabor - line detector, positive compound only')
    plt.imshow(sum22, cmap = 'gray', interpolation='nearest')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.title('Segmentation - total time %s'%"{0:.2f}".format(time()-start))
    plt.imshow(mark_boundaries(rebw, d_c))
    plt.show()
