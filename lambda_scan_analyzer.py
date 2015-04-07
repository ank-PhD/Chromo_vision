__author__ = 'ank'

import os
import PIL
from PIL import Image
import numpy as np
from skimage import img_as_float, color
from colorpy.ciexyz import xyz_from_spectrum
from colorpy.colormodels import rgb_from_xyz, irgb_from_xyz
from matplotlib import pyplot as plt
from scipy.ndimage import filters
from scipy.interpolate import spline


directory = 'L:/ank/bfp_sarah/sliced10'
scaling_factor = (1.0, 1.0, 1.0)

def create_image_reader():
    stack_index = []
    stack = []
    for _i, im_name in enumerate(sorted(os.listdir(directory))):
        if '.tif' in im_name:
            wavelength = int(im_name.split('.')[0])
            print _i, wavelength
            image = color.rgb2gray(img_as_float(PIL.Image.open(os.path.join(directory, im_name))))
            stack.append(image)
            stack_index.append(wavelength)
    return np.array(stack_index), np.array(stack)


def rgb_from_spectrum(array_1D):
    retset = np.vstack([idx, array_1D]).T
    return rgb_from_xyz(xyz_from_spectrum(retset))*np.max(array_1D)


def show_spectrum(point, dispersion=0, color='k'):
    print 'showing spectrum', point, dispersion
    if not dispersion:
        restack = stack
    else:
        restack = np.array([filters.gaussian_filter(stack[_j, :, :], sigma=dispersion) for _j in range(0, stack.shape[0])])
    # plt.plot(idx, restack[:, point[0], point[1]], 'r+')
    xnew = np.linspace(np.min(idx), np.max(idx), 300)
    smooth = spline(idx, restack[:, point[0], point[1]], xnew)
    plt.plot(xnew, smooth, color)


if __name__ == '__main__':
    idx, stack = create_image_reader()
    # for i in range(0, idx.shape[0]):
    #     plt.title(idx[i])
    #     plt.imshow(stack[i, :, :])
    #     plt.show()
    # idx, stack = (idx[4:8], stack[4:8, :, :])
    # print idx, stack
    position = (350, 335)
    position2 = (400, 275)
    rgb_image = np.apply_along_axis(func1d=rgb_from_spectrum, axis=0, arr=stack)
    rgb_image = np.rollaxis(rgb_image, 0, 3)
    plt.imshow(rgb_image)
    plt.plot(position[0], position[1], 'r.')
    plt.plot(position2[0], position2[1], 'r.')
    plt.show()
    show_spectrum(position, 5, 'k')
    show_spectrum(position2, 5, 'r')
    plt.show()
