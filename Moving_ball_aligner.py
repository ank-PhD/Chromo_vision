__author__ = 'ank'

import os
import PIL
from PIL import Image
from matplotlib import cm
import cv2
import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from os import path
from skimage import img_as_float, color
from images2gif import writeGif
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from pickle import dump, load
from time import time

timing = True

def create_timer():
    inner_time = [time()]

    def _time(msg):
        print msg, time() - inner_time[0]
        inner_time[0] = time()

    return _time

timer1 = create_timer()
timer2 = create_timer()

def time_wrapper(funct):

    def time_execution(*args,**kwargs):
        start = time()
        result = funct(*args, **kwargs)
        if timing:
            print funct.__name__, time()-start
        time_execution.__doc__ = funct.__doc__
        return result

    return time_execution

showtime = [time()]

def show_IO(funct):

    def IO_shower(*args, **kwargs):
        ret = funct(*args, **kwargs)
        if type(args[0]) == np.ndarray:
            pts = np.float32([ [0,0],[0,1],[1,1],[1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, ret)
            print dst[0, 0, :],

        return ret

    return IO_shower


def show_IO_2(funct):

    def IO_shower(*args, **kwargs):
        ret = funct(*args, **kwargs)
        if type(args[0]) == np.ndarray and time() - showtime[0] > 5:
            plt.imshow(np.squeeze(ret) - np.squeeze(args[0]))
            plt.show()
            showtime[0] = time()

        return ret

    return IO_shower

######################################################################################################################

buffer_directory = 'H:/buffer_folder'
directory = 'L:/Philippe/experiments/migration assay on polyacrylamide gels/gels loaded with beads/01.14.2015/tiff'
scaling_factor = (1.0, 1.0, 3.0)

def name_image(t, z, c):
    return path.join(directory, 'redt%sz%sc%s.tif'%( format(t, "02d"), format(z, "02d"), c))


def create_image_reader_by_z_stack(color_channel):
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
            stack.append(color.rgb2gray(img_as_float(PIL.Image.open(name_image(_t, _z, color_channel)))))
        yield np.array(stack)


def render_single_color(_3D_matrix, v_min=0.95):
    red = (1.0, 0.0, 0.0)
    s1 = mlab.pipeline.scalar_field(_3D_matrix)
    s1.spacing = scaling_factor
    mlab.pipeline.volume(s1, color=red, vmin=v_min, name='RED')


@show_IO
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


def flatten_array(_3D_array, axis=0):
        _3D_array[_3D_array < np.percentile(_3D_array, 99)] = 0.0
        return np.max(_3D_array, axis)


def transform_volume(_3D_array, Multi_M_xy, axis):

    def fct1(img):
        _img = np.squeeze(img)
        retim = cv2.warpPerspective(_img,  Multi_M_xy, _img.T.shape)
        return np.reshape(retim, img.shape)

    return np.concatenate(tuple(map(fct1, np.split(_3D_array, _3D_array.shape[axis], axis))), axis=axis)


def basic_aligner(reader, axis = 0):

    prev_time_volume = reader.next()
    # yield prev_time_volume

    Multi_M_xy = 0
    timer1('setup')
    for _i, current_time_volume in enumerate(reader):
        print _i,
        M_xy = align_plane(flatten_array(prev_time_volume), flatten_array(current_time_volume))
        if _i == 0:
            Multi_M_xy = M_xy
        else:
            Multi_M_xy = np.dot(Multi_M_xy, M_xy)

        prev_time_volume = current_time_volume
        timer1('full loop')
        yield Multi_M_xy


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

def flatten_image(img):
    return flatten_array(img, 0)

def PIL_render(img):
    return Image.fromarray(cm.gist_earth(img/np.max(img), bytes=True))

#
# def dumper(_3D_image, image_prefix='0',cargo=[0]):
#     dump(_3D_image, open(os.path.join(buffer_directory, '3D_dump_%s_%s.dmp'%(image_prefix, str(cargo[0]).zfill(2))), 'wb'))
#     cargo[0] += 1
#
#
# def undumper(image_prefix):
#     for fname in sorted(os.listdir(buffer_directory)):
#         if len(fname.split('_')) == 4 and fname.split('_')[2] == image_prefix:
#             yield load(open(os.path.join(buffer_directory, fname), 'rb'))


def transform_reader_flow(reader, transformation_matrix_list, axis):
    yield reader.next()
    for T_mat in transformation_matrix_list:
        yield transform_volume(reader.next(), T_mat, axis)



if __name__ == '__main__':
    rdr1 = create_image_reader_by_z_stack(color_channel = 1)
    rdr2 = create_image_reader_by_z_stack(color_channel = 1)
    rdr3 = create_image_reader_by_z_stack(color_channel = 1)
    rdr4 = create_image_reader_by_z_stack(color_channel = 1)
    rdr5 = create_image_reader_by_z_stack(color_channel = 1)
    rdr6 = create_image_reader_by_z_stack(color_channel = 1)

    zy_T_mats = [zy_T_mat for zy_T_mat in basic_aligner(rdr1, axis=1)]
    ini_xy_T_mats = [xy_T_mat for xy_T_mat in basic_aligner(rdr5, axis=0)]
    xy_T_mats = [xy_T_mat for xy_T_mat in basic_aligner(transform_reader_flow(rdr2, zy_T_mats, axis=1), axis=0)]


    in_frames = map(flatten_image, rdr4)
    in_frames = map(PIL_render, in_frames)

    inter_frames = map(flatten_image, transform_reader_flow(rdr6, zy_T_mats, axis=1))
    inter_frames = map(PIL_render, inter_frames)

    fin_frames = map(flatten_image, transform_reader_flow(transform_reader_flow(rdr3, zy_T_mats, axis=1), xy_T_mats, axis=0))
    fin_frames = map(PIL_render, fin_frames)

    writeGif('non-stabilized_flattened_balls.gif', in_frames, duration=0.3)
    writeGif('xy-stabilized_flattened_balls.gif', inter_frames, duration=0.3)
    writeGif('stabilized_flattened_balls.gif', fin_frames, duration=0.3)
