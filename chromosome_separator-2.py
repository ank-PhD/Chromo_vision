__author__ = 'ank'

import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
import math
from time import time
import random as rand
from pprint import PrettyPrinter
from copy import copy
import mdp
from itertools import product
from skimage.segmentation import felzenszwalb, slic, quickshift, random_walker
from skimage.segmentation import mark_boundaries
from skimage.data import lena
from skimage.util import img_as_float
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN

def import_image():
    ImageRoot = "/home/ank/Documents/var"
    col = PIL.Image.open(ImageRoot + '/n4.jpeg')
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw = bw - np.min(bw)
    bw = bw.astype(np.float64)/float(np.max(bw))
    return bw


def sub_sample_image(bw_image):
    density = 0.25
    msk = np.random.random(bw_image.shape)*bw_image>density
    xy = np.nonzero(msk).T
    # meh, we will loose information if we do this.
    # Let's try something else:
        # - spectral clustering
        # - mdp with gabor convolution


def check_integral(gabor_filter):
    ones = np.ones(gabor_filter.shape)
    avg = np.average(ones*gabor_filter)
    print avg
    return gabor_filter-avg


def gabor(bw_image):
    quality = 16
    scale = .5
    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    freq = 1./4
    phis = [pi/2, pi]
    size = (10, 10)
    sgm = (5*scale, 3*scale)

    nfilters = len(orientations)*len(phis)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, (alpha, phi) in enumerate(product(orientations, phis)):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        arr = check_integral(arr)
        gabors[i,:,:] = arr
        check_integral(arr)
    node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill', fillvalue=0, output_2d=False)
    cim = node.execute(bw[np.newaxis, :, :])
    sum1 = np.zeros(cim[0, 0,:,:].shape)
    col1 = np.zeros((cim[0,:,:,:].shape[0]/2, cim[0,:,:,:].shape[1], cim[0,:,:,:].shape[2]))
    sum2 = np.zeros(cim[0, 0,:,:].shape)
    col2 = np.zeros((cim[0,:,:,:].shape[0]/2, cim[0,:,:,:].shape[1], cim[0,:,:,:].shape[2]))
    for i in range(0, nfilters):
        pr_cim = cim[0, i,:,:]
        if i%2 == 0:
            sum1 = sum1 + np.abs(pr_cim)
            col1[i/2,:,:] = pr_cim
        else:
            pr_cim[pr_cim>0]=0
            sum2 = sum2 + np.abs(pr_cim)
            col2[i/2,:,:] = pr_cim

    return sum1, sum2, col1, col2


def core_cluster(core_domains):

    supersampling = 30.
    spread = 0.6

    core_domains = core_domains/np.max(core_domains)
    core_domains[core_domains < 1/256.0] = 0
    xy = []
    for x, y in np.array(core_domains.nonzero()).T.tolist():
        npts = int(core_domains[x,y]*supersampling)
        for i in range(0, npts):
            xy.append((x,y))
    xy = np.array(xy)
    print xy.shape
    xy = xy + np.random.normal(scale=spread, size=xy.shape)

    xy = mdp.numx.take(xy, mdp.numx_rand.permutation(xy.shape[0]), axis=0)

    # plt.imshow(core_domains, cmap = 'gray', interpolation='nearest')
    # plt.plot(xy.T[1], xy.T[0], 'ro', alpha=0.5)
    # plt.show()

    # NOPE, doesn't work either
    db = DBSCAN(eps=2, min_samples=50).fit(xy)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)


    plt.imshow(core_domains.T, cmap = 'gray', interpolation='nearest')
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'

        class_member_mask = (labels == k)

        x_y = xy[class_member_mask & core_samples_mask]
        plt.plot(x_y[:, 0], x_y[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        x_y = xy[class_member_mask & ~core_samples_mask]
        plt.plot(x_y[:, 0], x_y[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()




    # the following neural gaz approach failed miserably.

    # gng = mdp.nodes.GrowingNeuralGasNode(max_nodes = 500)
    # STEP = 5000
    # for i in range(0, xy.shape[0], STEP):
    #     gng.train(xy[i:i+STEP])
    #     # plt.imshow(core_domains.T, cmap = 'gray', interpolation='nearest')
    #     # plt.plot(gng.get_nodes_position(),' ro', alpha=0.5)
    #     # plt.show()
    #
    # plt.plot(xy.T[0],xy.T[1], 'ro')
    # plt.plot(gng.get_nodes_position(),' bo', alpha=0.5)
    # plt.show()


def diffuse_clusters(data, core_clusters):

    image = data.astype(np.int16)

    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)

    labels2 = random_walker(data2, core_clusters, beta=10, mode='bf')

    plt.imshow(mark_boundaries(re_img, labels))
    plt.show()

    plt.imshow(mark_boundaries(re_img, labels2))
    plt.show()


if __name__ == "__main__":
    bw = import_image()
    plt.imshow(bw, cmap = 'gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
    sum1, sum2, _, _ = gabor(bw)

    plt.imshow(sum1, cmap = 'gray', interpolation='nearest')
    plt.colorbar()

    plt.show()
    plt.imshow(sum2, cmap = 'gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    core_cluster(sum2)
    # sub_sample_image(bw)
