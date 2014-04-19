__author__ = 'ank'

import os
import PIL
import itertools
import numpy as np
from time import time
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
from os.path import isfile
from skimage import img_as_float, color
from skimage import filter, measure
from collections import namedtuple, defaultdict
from sklearn.feature_extraction import image
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from mayavi import mlab

# Global_idea: recognise shapes, render them individually, then plot lines in


ImageRoot = "/n/projects/ank/2014/Chromo_Motility/confocal-ank/2Apr_chopped_png"
# titles = namedtuple('img',['t','z','c'])

def denseplot_3D(vls):
    """
    performs a density plot with nice colors in 3D

    :param vls:
    """
    vls = vls.T
    print vls.shape
    kde = stats.gaussian_kde(vls)
    density = kde(vls)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x, y, z = vls
    ax.scatter(x, y, z, c=density)
    plt.show()


def load_img_dict():
    """
   loads and classifies images from the ImageRoot directory

    :return:
    """
    name_dict = defaultdict(lambda : defaultdict(dict))
    for img in os.listdir(ImageRoot):
        if isfile(ImageRoot+'/'+img):
            imge = color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))
            coords = [wd[1:] for wd in img.split('.')[0].split('_')[2:]]
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


def sample_operations(name_dict, z_depth, shape_dict):


    z_stack = next(name_dict.itervalues())

    _3D_chan1 = np.zeros((shape_dict[1][0], shape_dict[1][1], z_depth))
    _3D_chan2 = np.zeros((shape_dict[2][0], shape_dict[2][1], z_depth))

    for depth, bi_image in z_stack.iteritems():
        img1 = bi_image[1]
        img2 = bi_image[2]

        _3D_chan1[:, :, depth-1] = img1
        _3D_chan2[:, :, depth-1] = img2

    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan1), vmin=0.2, vmax=0.8)
    mlab.show()

    mlab.pipeline.volume(mlab.pipeline.scalar_field(_3D_chan2), vmin=0.2, vmax=0.8)
    mlab.show()

if __name__ == "__main__":
    img_dict, z_stack, chan_shapes = load_img_dict()
    # show_a_stack(img_dict)
    sample_operations(img_dict, z_stack, chan_shapes)


    # def randrange(n, vmin, vmax):
    #     return (vmax-vmin)*np.random.rand(n) + vmin
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # n = 100
    # for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zl, zh)
    #     ax.scatter(xs, ys, zs, c=c, marker=m)
    #
    # plt.show()

    # mu=np.array([1,10,20])
    # sigma=np.matrix([[4,10,0],[10,25,0],[0,0,100]])
    # data=np.random.multivariate_normal(mu,sigma,1000)
    # values = data.T
    #
    # print type(values), values.shape
