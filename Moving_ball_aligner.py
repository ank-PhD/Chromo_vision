__author__ = 'ank'

import os
import PIL
from PIL import Image
from matplotlib import cm
import cv2
import numpy as np
from mayavi import mlab
from skimage import img_as_float, color
from matplotlib import pyplot as plt
from cv2_debug import drawMatches
from collections import defaultdict
from os import path
from skimage import img_as_float, color
from images2gif import writeGif
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from pickle import dump, load
from time import time

timing = True

def time_wrapper(funct):

    def time_execution(*args,**kwargs):
        start = time()
        result = funct(*args, **kwargs)
        if timing:
            print funct.__name__, time()-start
        time_execution.__doc__ = funct.__doc__
        return result

    return time_execution

directory = 'L:/Philippe/experiments/migration assay on polyacrylamide gels/gels loaded with beads/01.14.2015/tiff'
scaling_factor = (1.0, 1.0, 3.0)

def name_image(t, z, c):
    return path.join(directory, 'redt%sz%sc%s.tif'%( format(t, "02d"), format(z, "02d"), c))


def create_image_reader_by_z_stack(clr):
    _stack = []
    for im_name in os.listdir(directory):
        if '.tif' in im_name:
            coordinates = (int(im_name[4:6]), int(im_name[7:9]), int(im_name[10]))
            _stack.append(coordinates)
    t, z, c = tuple(np.max(np.array(_stack), axis=0).tolist())
    print 'maximum', t, z
    for _t in range(1, t+1):
        stack = []
        for _z in range(2, z+1):
            stack.append(color.rgb2gray(img_as_float(PIL.Image.open(name_image(_t, _z, clr)))))
        yield np.array(stack)


def render_single_color(_3D_matrix, v_min=0.95):
    red = (1.0, 0.0, 0.0)
    s1 = mlab.pipeline.scalar_field(_3D_matrix)
    s1.spacing = scaling_factor
    mlab.pipeline.volume(s1, color=red, vmin=v_min, name='RED')

def align_plane(reference, corrected, align_quality = 10):
    MIN_MATCH_COUNT = align_quality

    img1 = (corrected/np.max(corrected)*255).astype(np.uint8)
    img2 = (reference/np.max(reference)*255).astype(np.uint8)

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        return M

    else:
        raise Exception('Match not good!')


def flatten_xy(_3D_array):
        _3D_array[_3D_array < np.percentile(_3D_array, 99)] = 0.0
        res = np.max(_3D_array, 0)
        return res


def align_xy(ref_3D, img_3D):

    def fct(img1):
        img1 = img1[0, :, :]
        return cv2.warpPerspective(img1, M, img1.T.shape)[np.newaxis, :, :]

    M = align_plane(flatten_xy(ref_3D), flatten_xy(img_3D))

    resmat = np.concatenate(tuple(map(fct, np.split(img_3D, img_3D.shape[0]))))
    return resmat


def align_z():
    pass

def basic_aligner():
    reader = create_image_reader_by_z_stack(1)
    time_volume_0 = reader.next()
    im_stack = [flatten_xy(time_volume_0)]

    Multi_M = 0
    ltime = time()
    for _i, time_volume_1 in enumerate(reader):
        print _i, time() - ltime
        ltime = time()
        im_stack.append(flatten_xy(align_xy(time_volume_0, time_volume_1)))

    return im_stack

def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    return np.where(detected_minima)

@time_wrapper
def PIL_render(img):
    return Image.fromarray(cm.gist_earth(img, bytes=True))

if __name__ == '__main__':
    # frames = basic_aligner()
    # dump(frames, open('temp_dump.dmp', 'w'))
    frames = load(open('temp_dump.dmp', 'r'))
    re_frames = map(PIL_render, frames)
    print re_frames

    writeGif('stabilized_flattened_balls.gif', re_frames, duration=0.2)

    # reader = create_image_reader_by_z_stack(1)

    # time_volume_0 = reader.next()
    # # render_single_color(time_volume_0)
    # # mlab.show()
    #
    # time_volume_1 = reader.next()
    # # render_single_color(time_volume_1)
    # # mlab.show()
    #
    #
    # time_volume_1 = align_xy(time_volume_0, time_volume_1)


    # align vertical max render
    # align lateral cut element

    # MIN_MATCH_COUNT = 10
    #
    # img1 = cv2.imread('src.jpg',0)  # queryImage
    # img2 = cv2.imread('src2.jpg',0) # trainImage
    #
    # # Initiate SIFT detector
    # sift = cv2.SIFT()
    #
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    #
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    #
    # matches = flann.knnMatch(des1,des2,k=2)
    #
    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    #
    # if len(good) > MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()
    #
    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts, M)
    #
    #     flatvects = np.int32(dst)
    #
    #     print M
    #     dst = cv2.perspectiveTransform(pts, M)
    #     new_im = cv2.warpPerspective(img1, M, img1.T.shape)
    #
    #     plt.imshow(img2, 'gray')
    #     plt.show()
    #
    #     plt.imshow(new_im, 'gray')
    #     plt.show()
    #
    #     plt.imshow(img2 - new_im)
    #     plt.show()
