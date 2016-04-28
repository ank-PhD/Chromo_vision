__author__ = 'ank'


import os
import PIL
import itertools
import mdp
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
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from mayavi import mlab
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from tvtk.util.ctf import load_ctfs, rescale_ctfs, save_ctfs
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pylab import get_cmap
from chiffatools.wrappers import time_wrapper, debug_wrapper


debug = False
timing = False
plt.figure(figsize=(20.0, 15.0))

# ImageRoot = "/home/ank/Documents/projects_files/2014/supra_Philippe/ko fibers"
ImageRoot = "L:/ank/supra from Linhao"
rev_depth = 2
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
            print tuple(int(element[1:]) for element in img[:-4].split('_')[-rev_depth:])
            name_dict[tuple(int(element[1:]) for element in img[:-4].split('_')[-rev_depth:])]=color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max([tpl[0] for tpl in name_dict.keys()])
    print 'focal planes detected: %s' % z_stack
    x, y = name_dict[(z_stack, 1)].shape

    stack = [np.zeros((x,y,z_stack+1)), np.zeros((x,y,z_stack+1))]

    for z, plane in name_dict.iteritems():
        stack[z[1]-1][:, : ,z[0]] = plane

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
        name_dict[int(img[-6:-4])] = color.rgb2gray(img_as_float(PIL.Image.open(ImageRoot+'/'+img)))

    z_stack = max(name_dict.keys())
    print 'focal planes detected: %s' % z_stack
    x, y = name_dict[z_stack].shape

    stack = np.zeros((x,y,z_stack*3+3))

    for z, plane in name_dict.iteritems():
        stack[:, :, z*3] = plane
        stack[:, :, z*3+1] = plane
        stack[:, :, z*3+2] = plane

    flter = stack < stack.max() * load_filter
    stack[flter] = 0.0

    return stack


def render_with_cut(chan1, chan2, v_min=0.6, cut=True):

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


def _2D_Gabor(bw_image, freq=1/32., scale=2, scale_distortion=1., field=10, phi=np.pi, quality = 16):
    '''
    Size-stable Gabor transform of an image

    :param bw_image:
    :param freq:
    :param scale:
    :param scale_distortion:
    :param field:
    :param phi:
    :return:
    '''


    def check_integral(gabor_filter):
        ones = np.ones(gabor_filter.shape)
        avg = np.average(ones * gabor_filter)
        return gabor_filter - avg


    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    size = (field, field)
    sgm = (5*scale, 3*scale*scale_distortion)

    nfilters = len(orientations)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, alpha in enumerate(orientations):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        arr = check_integral(arr)
        gabors[i, :, :] = arr
        if debug:
            plt.subplot(6, 6, i+1)
            plt.title('%s, %s, %s, %s'%('{0:.2f}'.format(alpha), '{0:.2f}'.format(phi), freq, sgm))
            plt.imshow(arr, cmap = 'gray', interpolation='nearest')
    if debug:
        plt.show()
        # plt.clf()
    node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill', fillvalue=0, output_2d=False)

    cim = node.execute(bw_image[np.newaxis, :, :])[0, :, :, :]
    re_cim = np.zeros((cim.shape[0], cim.shape[1] + field - 1, cim.shape[2] + field - 1))
    re_cim[:, field/2-1:-field/2, field/2-1:-field/2] = cim
    cim = re_cim

    return cim

# @debug_wrapper
def sq_activation(activations_stack):
    sum2 = - np.sum(activations_stack, axis=0)
    sum2[sum2>0] /= np.max(sum2)
    sum2[sum2<0] /= -np.min(sum2)
    return sum2


@debug_wrapper
def optical_dominance(activation_stack):

    def funct_1d(_1D_arr):
        return np.argmax(_1D_arr)

    return np.apply_along_axis(funct_1d, 0, np.abs(activation_stack))


@debug_wrapper
def optical_dominance_strength(activation_stack):

    def funct_1d(_1D_arr):
        mx = np.max(_1D_arr)
        ref = np.percentile(_1D_arr, 25)
        if mx > 0.01:
            return mx / np.min((0.1*mx, ref*mx))
        else:
            return 1

    return np.apply_along_axis(funct_1d, 0, np.abs(activation_stack))


def activation_in_stack(channel):

    acc = []
    for item in np.split(channel, channel.shape[2], axis=2):
        acti_stack = _2D_Gabor(item.squeeze())
        acti_flat = np.abs(sq_activation(acti_stack))
        acc.append(acti_flat)
        optical_dominance(acti_stack)
        optical_dominance_strength(acti_stack)
    ret_arr = np.array(acc)
    print ret_arr.shape
    ret_arr = np.rollaxis(ret_arr, 2)
    print ret_arr.shape
    ret_arr = np.rollaxis(ret_arr, 2)
    print ret_arr.shape
    return ret_arr

if __name__ == "__main__":

    stack1, stack2 = load_img_dict2(0.050)
    stack1 = activation_in_stack(stack1)
    # stack2 = activation_in_stack(stack2)
    print 1, stack1.shape
    print 2, stack2.shape
    # stack1 = load_img_dict(0.0)

    # stack1 = filters.gaussian_filter(stack1, sigma = 1)
    # stack2 = filters.gaussian_filter(stack2, sigma = 2)

    # mlab.pipeline.volume(stack1, vmin=0.40)
    # mlab.pipeline.volume(stack2, vmin=0.40)

    render_with_cut(stack1, stack2, v_min=0.1, cut=False)

    mlab.show()