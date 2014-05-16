__author__ = 'ank'


import os
import PIL
import itertools
import numpy as np
import pickle
from time import time
from copy import copy
from scipy import ndimage
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
from os.path import isfile
from skimage import img_as_float, color
from skimage import filter, measure
from collections import namedtuple, defaultdict
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering, DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from mayavi import mlab
from scipy.spatial import ConvexHull

from tvtk.util.ctf import load_ctfs, rescale_ctfs, save_ctfs

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pylab import get_cmap

ImageRoot = "/home/ank/Documents/projects_files/2014/supra_Philippe/ko fibers"
scaling_factor = (1.0, 1.0, 3.0)

def load_img_dict2(load_filter=0.4):
    """
    loads and classifies images from the ImageRoot directory, assuming that there are two channels

    :return:
    """

    name_dict = {}
    for img in os.listdir(ImageRoot):
        if '.png' in img:
            print '%s image was parsed' % img
            name_dict[tuple(int(element[1:]) for element in img[:-4].split('_')[1:])]=color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max([tpl[0] for tpl in name_dict.keys()])
    print 'focal planes detected: %s' % z_stack
    x, y = name_dict[(z_stack,1)].shape

    stack = [np.zeros((x,y,z_stack+1)), np.zeros((x,y,z_stack+1))]

    for z, plane in name_dict.iteritems():
        stack[z[1]-1][:,:,z[0]] = plane

    for sub_stack in stack:
        flter = sub_stack < sub_stack.max()*load_filter
        sub_stack[flter] = 0.0


    return stack[0], stack[1]



def load_img_dict(load_filter=0.4):
    """
   loads and classifies images from the ImageRoot directory, assuming that there is only one channel

    :return:
    """

    name_dict = {}
    for img in os.listdir(ImageRoot):
        print '%s image was parsed' % img
        name_dict[int(img[-6:-4])]=color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max(name_dict.keys())
    print 'focal planes detected: %s' % z_stack
    x, y = name_dict[z_stack].shape

    stack = np.zeros((x,y,z_stack*3+3))

    for z, plane in name_dict.iteritems():
        stack[:,:,z*3] = plane
        stack[:,:,z*3+1] = plane
        stack[:,:,z*3+2] = plane

    flter = stack < stack.max()*load_filter
    stack[flter] = 0.0

    return stack


def render_with_cut(chan1, chan2, v_min=0.6, cut = True):

    s1 = mlab.pipeline.scalar_field(chan1)
    s1.spacing = scaling_factor

    s2 = mlab.pipeline.scalar_field(chan2)
    s2.spacing = scaling_factor

    red = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.0)

    mlab.pipeline.volume(s1, color=red, vmin=v_min, name='RED')
    mlab.pipeline.volume(s2, color=green, vmin=v_min, name='GREEN')

    if cut:
        mlab.pipeline.image_plane_widget(s1, colormap = 'OrRd')
        mlab.pipeline.image_plane_widget(s2, colormap = 'Greens')



if __name__ == "__main__":

    stack1, stack2 = load_img_dict2(0.60)
    # stack1 = load_img_dict(0.0)

    # mlab.pipeline.volume(stack1, vmin=0.40)
    # mlab.pipeline.volume(stack2, vmin=0.40)
    render_with_cut(stack1, stack2, True)

    mlab.show()