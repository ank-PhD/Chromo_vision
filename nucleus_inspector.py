__author__ = 'ank'

import sip
sip.setapi('QString', 2)


import os
import PIL
import mdp
import itertools
import numpy as np
import pickle
from time import time
from copy import copy
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from os.path import isfile
from skimage import img_as_float, color
from skimage import filter, measure
from collections import namedtuple, defaultdict
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering, DBSCAN
# from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from mayavi import mlab
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

# from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pylab import get_cmap
# from tvtk.util.ctf import ColorTransferFunction
# from tvtk.util.ctf import PiecewiseFunction

# Global_idea: recognise shapes, render them individually, then plot lines in


# ImageRoot = "/n/projects/ank/2014/Chromo_Motility/confocal-ank/2Apr_chopped_png"
ImageRoot = "L:/ank/supra from Linhao"
pickle_location = '3_D_dict.dump'
# titles = namedtuple('img',['t','z','c'])


def denseplot_3D(vls, colormap=None):
    """
    performs a density plot with nice colors in 3D

    :param vls:
    """
    vls = vls.T
    print vls.shape
    kde = stats.gaussian_kde(vls)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x, y, z = vls
    if colormap is None:
        density = kde(vls)
        colormap = density
    print type(colormap), colormap
    ax.scatter(x, y, z, c=colormap)
    plt.show()


def load_img_dict():
    """
   loads and classifies images from the ImageRoot directory

    :return:
    """
    name_dict = defaultdict(lambda : defaultdict(dict))
    for img in os.listdir(ImageRoot):
        coords = [wd[1:] for wd in img.split('.')[0].split('_')[2:]]
        print 1, coords
        if isfile(ImageRoot+'/'+img):
            print 2, os.path.join(ImageRoot, img)
            imge = color.rgb2gray(img_as_float(PIL.Image.open(os.path.join(ImageRoot, img))))
            name_dict[int(coords[0])][int(coords[1])][int(coords[2])] = imge


    z_stacks = max(name_dict[1])
    channel_shape = dict((chan_name, chan_value.shape) for timepoint in name_dict.iterkeys() for chan_name, chan_value in name_dict[timepoint][1].iteritems())

    return name_dict, z_stacks, channel_shape


def show_a_stack(name_dict):
    """
    Shows a series of z-stack for a randomly chosen time point

    :param name_dict:
    """
    z_stack = next(name_dict.itervalues())

    for depth, bi_image in z_stack.iteritems():

        plt.figure()

        plt.subplot(121)
        title = "Color 1, z: %s" % depth
        plt.title(title)
        plt.imshow(bi_image[1], cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplot(122)
        title = "Color 2, z: %s" % depth
        plt.title(title)
        plt.imshow(bi_image[2], cmap=plt.cm.jet)
        plt.colorbar()

        plt.show()

def defficient_spectral_clustring(name_dict, z_depth, shape_dict):
    # requires to re-implement the distance definition on sparse images.

    z_stack = next(name_dict.itervalues())

    _3D_chan1 = np.zeros((shape_dict[1][0], shape_dict[1][1], z_depth))
    _3D_chan2 = np.zeros((shape_dict[2][0], shape_dict[2][1], z_depth))

    for depth, bi_image in z_stack.iteritems():
        img1 = bi_image[1]
        img2 = bi_image[2]

        _3D_chan1[:, :, depth-1] = img1
        _3D_chan2[:, :, depth-1] = img2

    # mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan1), vmin=0.1)
    # mlab.show()

    _3D_chan2[_3D_chan2<0.04] = 0

    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan2))
    mlab.show()

    mask = _3D_chan2.astype(bool)
    img = _3D_chan2.astype(float)

    graph = image.img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())

    print graph.shape
    print len(graph.nonzero()[0])

    clusters = 4
    labels = spectral_clustering(graph, n_clusters = clusters, eigen_solver='arpack')
    label_im = -np.ones(mask.shape)
    label_im[mask] = labels

    for i in range(0, clusters):
        re_img = copy(_3D_chan2)
        re_img[label_im!=i] = 0
        mlab.pipeline.volume(mlab.pipeline.scalar_field(re_img))
        mlab.show()


# TODO: 4D clustering?
def mCh_hull_and_cluster(z_stack, z_depth, shape_dict):
    """
    Performs clustering of the mCh signal in 3D by DBSCAN method
    performs mean of cluster and convex hull computation

    :param name_dict: name_dictionary
    :param z_depth: +
    :param shape_dict:
    """

    _3D_chan1 = np.zeros((shape_dict[1][0], shape_dict[1][1], z_depth))
    _3D_chan2 = np.zeros((shape_dict[2][0], shape_dict[2][1], z_depth))

    for depth, bi_image in z_stack.iteritems():
        img1 = bi_image[1]
        img2 = bi_image[2]

        _3D_chan1[:, :, depth-1] = img1
        _3D_chan2[:, :, depth-1] = img2


    _3D_chan1[_3D_chan1<0.45] = 0
    _3D_chan2[_3D_chan2<0.04] = 0

    pts = np.zeros((len(_3D_chan2.nonzero()[1]), 3))
    pts[:, 0] = _3D_chan2.nonzero()[0]
    pts[:, 1] = _3D_chan2.nonzero()[1]
    pts[:, 2] = _3D_chan2.nonzero()[2]


    db = DBSCAN(eps=2, min_samples=10).fit(pts)
    # core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    _3D_labels =  - np.ones(_3D_chan2.shape)
    cluster_means = []
    hulls = []
    std = []
    counter = 0
    for label in set(labels):
        x_y_z = pts[labels == label, :].T.astype(int)
        x,y,z = x_y_z
        if x_y_z.shape[1] < 50 or label == -1:
            _3D_labels[x,y,z] = -1
        else:
            cluster_means.append(np.mean(x_y_z, axis=1))
            hulls.append(ConvexHull(x_y_z.T))
            std.append(np.sqrt(np.sum(np.power(np.std(x_y_z,axis=1),2))))
            _3D_labels[x,y,z] = counter
            counter += 1

    _3D_chan2[_3D_labels<0] = 0

    print('Estimated number of clusters: %d' % counter)

    return _3D_chan1, _3D_chan2, _3D_labels, counter, cluster_means, std, hulls


def cluster_show(_3D_chan1, _3D_chan2, _3D_labels, n_clusters_, means, std, hulls):
    """

    :param _3D_chan1:
    :param _3D_chan2:
    :param _3D_labels:
    """
    red = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.1)

    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan1), color=green)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan2), color=red)
    mlab.show()

    cm = get_cmap('gist_rainbow')

    for i in range(0, n_clusters_):
        re_img = np.zeros(_3D_chan2.shape)
        re_img[_3D_labels == i] = _3D_chan2[_3D_labels == i]
        mlab.pipeline.volume(mlab.pipeline.scalar_field(re_img), color=tuple(cm(1.*i/n_clusters_)[:-1]))

    # mean_arr = np.zeros((n_clusters_, 3))
    # std_arr = np.zeros((n_clusters_))
    # for i in range(0, n_clusters_):
    #     mean_arr[i, :] = means[i]
    #     std_arr[i] = std[i]
    # x,y,z = mean_arr.T
    # mlab.points3d(x,y,z, std_arr)

    for hull in hulls:
        x,y,z = hull.points.T
        triangles = hull.simplices
        mlab.triangular_mesh(x, y, z, triangles, representation='wireframe', color=(0, 0, 0))

    mlab.show()


def sample_operations(name_dict, z_depth, shape_dict):

    z_stack = next(name_dict.itervalues())

    _3D_chan1 = np.zeros((shape_dict[1][0], shape_dict[1][1], z_depth))
    _3D_chan2 = np.zeros((shape_dict[2][0], shape_dict[2][1], z_depth))

    for depth, bi_image in z_stack.iteritems():
        img1 = bi_image[1]
        img2 = bi_image[2]

        _3D_chan1[:, :, depth-1] = img1
        _3D_chan2[:, :, depth-1] = img2


    _3D_chan1[_3D_chan1<0.45] = 0
    _3D_chan2[_3D_chan2<0.04] = 0

    pts = np.zeros((len(_3D_chan2.nonzero()[1]), 3))
    pts[:, 0] = _3D_chan2.nonzero()[0]
    pts[:, 1] = _3D_chan2.nonzero()[1]
    pts[:, 2] = _3D_chan2.nonzero()[2]


    db = DBSCAN(eps=3, min_samples=10).fit(pts)
    # core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    _3D_labels =  np.zeros(_3D_chan2.shape)
    cluster_means = []
    hulls = []
    i = 0
    for label in set(labels):
        x_y_z = pts[labels == label, :].T.astype(int)
        x,y,z = x_y_z
        cluster_means.append(np.mean(x_y_z, axis=1))
        hulls.append(ConvexHull(x_y_z.T))
        _3D_labels[x,y,z] = label

    print cluster_means

    _3D_chan2[_3D_labels<0] = 0


    # <====================>Show the localization and clusters<====================>
    red = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.1)

    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan1), color=green)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan2), color=red)
    mlab.show()

    cm = get_cmap('gist_rainbow')

    for i in range(0, n_clusters_):
        re_img = np.zeros(_3D_chan2.shape)
        x,y,z = pts[labels == i, :].T.astype(int)
        re_img[x,y,z] = _3D_chan2[x,y,z]
        mlab.pipeline.volume(mlab.pipeline.scalar_field(re_img), color=tuple(cm(1.*i/n_clusters_)[:-1]))

    mlab.show()


if __name__ == "__main__":
    img_dict, z_depth, chan_shapes = load_img_dict()
    # show_a_stack(img_dict)
    # resdict = {}
    # for i, z_stack in img_dict.iteritems():
    #     resdict[i] = mCh_hull_and_cluster(z_stack, z_depth, chan_shapes)
    #     cluster_show(*resdict[i])
    # pickle.dump(resdict, open(pickle_location, 'w'))
    sample_operations(img_dict, z_depth, chan_shapes)


