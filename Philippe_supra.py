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

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from pylab import get_cmap

ImageRoot = "/home/ank/Documents/projects_files/2014/supra_Philippe/ko fibers"


def load_img_dict2():
    """
    loads and classifies images from the ImageRoot directory

    :return:
    """

    name_dict = {}
    for img in os.listdir(ImageRoot):
        if '.png' in img:
            print img
            name_dict[tuple(int(element[1:]) for element in img[:-4].split('_')[1:])]=color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max([tpl[0] for tpl in name_dict.keys()])
    print z_stack
    x, y = name_dict[(z_stack,1)].shape

    stack = [np.zeros((x,y,z_stack*3+3)), np.zeros((x,y,z_stack*3+3))]

    for z, plane in name_dict.iteritems():
        stack[z[1]-1][:,:,z[0]*3] = plane
        stack[z[1]-1][:,:,z[0]*3+1] = plane
        stack[z[1]-1][:,:,z[0]*3+2] = plane

    return stack[0], stack[1]



def load_img_dict():
    """
   loads and classifies images from the ImageRoot directory

    :return:
    """

    name_dict = {}
    for img in os.listdir(ImageRoot):
        print img
        name_dict[int(img[-6:-4])]=color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max(name_dict.keys())
    print z_stack
    x, y = name_dict[z_stack].shape

    stack = np.zeros((x,y,z_stack*3+3))

    for z, plane in name_dict.iteritems():
        stack[:,:,z*3] = plane
        stack[:,:,z*3+1] = plane
        stack[:,:,z*3+2] = plane

    return stack


if __name__ == "__main__":

    stack1, stack2 = load_img_dict2()

    print stack1

    mlab.pipeline.volume(mlab.pipeline.scalar_field(stack2), vmin=0.40,)
    mlab.show()