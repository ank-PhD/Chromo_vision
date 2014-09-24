__author__ = 'ank'

import numpy as np
import matplotlib.pyplot as plt
import PIL
from time import time
import mdp
from skimage.segmentation import random_walker, mark_boundaries
from skimage.morphology import label, convex_hull_image
from skimage.filter import gaussian_filter
from skimage.measure import perimeter
from math import pow
from matplotlib import colors
from pylab import get_cmap
from itertools import product
from configs import image_directory, image_to_load, buffer_directory

def import_image():
    col = PIL.Image.open(image_directory + image_to_load)
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw = bw - np.min(bw)
    bw = bw.astype(np.float64)/float(np.max(bw))
    return bw


def import_edited():
    col = PIL.Image.open(buffer_directory+"EDIT_ME2.jpg")
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw<=120] = 0
    bw[bw>120] = 1
    return bw

def gabor(bw_image, freq, scale, scale_distortion=1., self_cross=False, field=10):

    # gabor filter normalization with respect to the surface convolution
    def check_integral(gabor_filter):
        ones = np.ones(gabor_filter.shape)
        avg = np.average(ones*gabor_filter)
        return gabor_filter-avg

    quality = 16
    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    phis = [pi/2, pi]
    size = (field, field)
    sgm = (5*scale, 3*scale*scale_distortion)

    nfilters = len(orientations)*len(phis)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, (alpha, phi) in enumerate(product(orientations, phis)):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        if self_cross:
            arr=np.minimum(arr,mdp.utils.gabor(size, alpha+pi/2, phi, freq, sgm))
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
    markers[data < -0.15] = 1
    markers[data > 0.15] = 2
    labels2 = random_walker(data, markers, beta=10, mode='bf')

    return labels2

def cluster_process(labels):
    # plt.imshow(labels, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    rbase = np.zeros(labels.shape)
    for i in range(1, int(np.max(labels))):
        base = np.zeros(labels.shape)
        base[labels==i] = 1
        # plt.imshow(base, cmap='gray', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        li = len(base.nonzero()[0])
        # print li
        if li>0:
            hull = convex_hull_image(base)
            lh =len(hull.nonzero()[0])
            cond = li>4000 and float(lh)/float(li)<1.07
            # print i, li, float(lh)/float(li), cond, pow(perimeter(base), 2.0)/li
            if cond:
                rbase = rbase + base
    return rbase


def repaint_culsters(clusterNo=100):
    prism_cmap = get_cmap('prism')
    prism_vals = prism_cmap(np.arange(clusterNo))
    prism_vals[0] = [0, 0, 0, 1]
    costum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', prism_vals)
    return costum_cmap


if __name__ == "__main__":
    start = time()
    bw = import_image()
    # plt.imshow(bw, cmap='gray', interpolation='nearest')
    # plt.show()
    # sum1, sum2, _, _= gabor(bw, 1/4., 0.5)
    sum1, sum2, _, _ = gabor(bw, 1/8., 1, self_cross=True, field=20)

    # The separator is acting here:
    sum10, sum20, _, _ = gabor(bw, 1/4., 0.5, field=20)
    # plt.imshow(sum20, cmap='gray', interpolation='nearest')
    # plt.show()
    sum20[sum20>-0.15] = 0
    sum2  = sum2 + sum20

    bw_blur = gaussian_filter(bw, 10)
    bwth = np.zeros(bw_blur.shape)
    bwth[bw_blur>0.3] = 1
    clsts = label(bwth)*bwth

    rbase = cluster_process(clsts)[9:,:][:,9:][:-10,:][:,:-10]

    plt.subplot(2,2,1)
    plt.title('Original image')
    plt.imshow(bw, cmap='gray', interpolation='nearest')

    plt.subplot(2,2,2)
    plt.title('blurred image')
    plt.imshow(bw_blur, cmap='gray', interpolation='nearest')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('blurred thresholded clusters')
    plt.imshow(clsts, cmap='spectral', interpolation='nearest')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.title('Clusters that look like cells')
    plt.imshow(rbase, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    sum22 = np.copy(sum2)
    sum22[sum2<0] = 0
    d_c = cluster_by_diffusion(sum2)
    rebw = bw[9:,:][:,9:][:-10,:][:,:-10]

    reim = PIL.Image.fromarray((rebw/np.max(rebw)*254).astype(np.uint8))
    reim.save(buffer_directory+"I_AM_THE_ORIGINAL.bmp")

    seg_dc = label(d_c, background=0)*d_c
    redd = set(seg_dc[rbase>0.01].tolist())

    # print redd
    for i in redd:
        seg_dc[seg_dc==i] = 0
    d_c = d_c*0
    d_c[seg_dc>0] = 1

    int_arr = np.asarray(np.dstack((d_c*254, d_c*254, d_c*0)), dtype=np.uint8)
    msk = PIL.Image.fromarray(int_arr)
    msk.save(buffer_directory+"EDIT_ME.bmp")
    raw_input("Please manually edit the mask image, save it. Once you are done, press enter to continue ")
    d_c = import_edited()

    seg_dc = (label(d_c)+1)*d_c
    colormap = repaint_culsters(int(np.max(seg_dc)))

    # plt.subplot(2,2,1)
    # plt.title('Original image')
    # plt.imshow(bw, cmap='gray', interpolation='nearest')
    #
    # plt.subplot(2,2,2)
    # plt.title('Gabor - line detector')
    # plt.imshow(sum2, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    #
    # plt.subplot(2,2,3)
    # plt.title('Gabor - line detector, positive compound only')
    # plt.imshow(sum22, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    #
    # plt.subplot(2,2,4)
    # plt.title('Segmentation - total time %s'%"{0:.2f}".format(time()-start))
    # plt.imshow(mark_boundaries(rebw, d_c))
    # plt.show()



    plt.subplot(2,2,1)
    plt.title('Original image')
    plt.imshow(bw, cmap='gray', interpolation='nearest')

    plt.subplot(2,2,2)
    plt.title('chromosome mask')
    plt.imshow(mark_boundaries(rebw, d_c))

    plt.subplot(2,2,3)
    plt.title('Segmentation - total time %s, clusters: %s'%("{0:.2f}".format(time()-start), len(set(seg_dc.flatten().tolist()))) )
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.title('Segmentation-image overlay')
    plt.imshow(mark_boundaries(rebw, d_c))
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest', alpha=0.3)
    plt.colorbar()

    plt.show()
