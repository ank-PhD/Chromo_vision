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
from scipy.spatial.distance import cdist, pdist, squareform
from chiffatools.Linalg_routines import hierchical_clustering

# TODO: show where the agorithm finds the image centers

debug = True
timing = True

def create_timer():
    inner_time = [time()]

    def _time(msg):
        kptime = time() - inner_time[0]
        print msg, kptime
        inner_time[0] = time()
        return kptime

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

######################################################################################################################

# buffer_directory = 'H:/buffer_folder'
directory = 'L:/Philippe/experiments/migration assay on polyacrylamide gels/gels loaded with beads/02.18.2015/TIFF/3'
scaling_factor = (1.0, 1.0, 1.0)
# prefix = 'red'
prefix = 'ko npc beads gel 0.6kpa008'


def name_image(t, z, c):
    return path.join(directory, prefix + 't%sz%sc%s.tif' % ( format(t, "02d"), format(z, "02d"), c))


def create_image_reader_by_z_stack(color_channel):
    _stack = []
    for im_name in os.listdir(directory):
        if '.tif' in im_name:
            lp = len(prefix)
            tl = 2
            zl = 2
            coordinates = (int(im_name[lp+1:lp+1+tl]), int(im_name[lp+2+tl:lp+2+zl+tl]), int(im_name[lp+3+zl+tl]))
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


def render_2_colors(_3D_matrix_1, _3D_matrix_2, v_min=0.95):
    s1 = mlab.pipeline.scalar_field(_3D_matrix_1)
    s1.spacing = scaling_factor

    s2 = mlab.pipeline.scalar_field(_3D_matrix_2)
    s2.spacing = scaling_factor

    red = (1.0, 0.0, 0.0)
    green = (0.0, 1.0, 0.0)

    mlab.pipeline.volume(s1, color=red, vmin=v_min, name='RED')
    mlab.pipeline.volume(s2, color=green, vmin=v_min, name='GREEN')

    mlab.show()

# @show_IO
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
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        return M

    else:
        plt.subplot(211)
        plt.imshow(img1, interpolation='nearest')
        plt.subplot(212)
        plt.imshow(img2, interpolation='nearest')
        plt.show()
        raise Exception('Match not good!')


def flatten_array(_3D_array, axis=0):
        _3D_array[_3D_array < np.percentile(_3D_array, 99)] = 0.0
        return np.max(_3D_array, axis)


def transform_volume(_3D_array, Multi_M_xy, axis):

    def fct1(img):
        _img = np.squeeze(img)
        retim = cv2.warpPerspective(_img,  Multi_M_xy, _img.T.shape)
        return np.reshape(retim, img.shape)

    return  np.concatenate(tuple(map(fct1, np.split(_3D_array, _3D_array.shape[axis], axis))), axis=axis)


def basic_aligner(reader, axis = 0):

    def debug_render():
        fig = plt.subplot(321)
        plt.title('xy projection')
        plt.imshow(flatten_array(prev_time_volume), interpolation='nearest')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        fig =plt.subplot(322)
        plt.title('zx projection')
        plt.imshow(flatten_array(prev_time_volume, 1), interpolation='nearest', aspect='auto')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        fig =plt.subplot(323)
        plt.imshow(flatten_array(current_time_volume), interpolation='nearest')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        fig =plt.subplot(324)
        plt.imshow(flatten_array(current_time_volume, 1), interpolation='nearest', aspect='auto')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        fig =plt.subplot(325)
        plt.imshow(flatten_array(transform_volume(current_time_volume, Multi_M_xy, axis)), interpolation='nearest')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        fig =plt.subplot(326)
        plt.imshow(flatten_array(transform_volume(current_time_volume, Multi_M_xy, axis), 1), interpolation='nearest', aspect='auto')
        fig.set_xticklabels([])
        fig.set_yticklabels([])
        plt.savefig('transformation.axis%s.z%s.%s.png'%(axis, _i, int(time()-tint)))
        plt.clf()

    prev_time_volume = reader.next()
    Multi_M_xy = 0
    timer1('setup')
    tint = time()
    for _i, current_time_volume in enumerate(reader):
        print _i,
        try:
            M_xy = align_plane(flatten_array(prev_time_volume, axis), flatten_array(current_time_volume, axis))
        except Exception:
            render_2_colors(prev_time_volume, current_time_volume)
            raise
        if _i == 0:
            Multi_M_xy = M_xy
        else:
            Multi_M_xy = np.dot(Multi_M_xy, M_xy)

        if debug:
            debug_render()

        prev_time_volume = current_time_volume
        timer1('.')
        yield Multi_M_xy


def detect_local_maxima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_max = (filters.maximum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_max - eroded_background
    return np.where(detected_minima)


def flatten_image(img):
    return flatten_array(img, 0)


def PIL_render(img):
    return Image.fromarray(cm.gist_earth(img/np.max(img), bytes=True))


def transform_reader_flow(reader, transformation_matrix_list, axis):
    yield reader.next()
    for T_mat in transformation_matrix_list:
        yield transform_volume(reader.next(), T_mat, axis)


def create_tri_alignement():
    c_chan = 2

    rdr1 = create_image_reader_by_z_stack(color_channel=c_chan)
    rdr2 = create_image_reader_by_z_stack(color_channel=c_chan)
    rdr3 = create_image_reader_by_z_stack(color_channel=c_chan)
    rdr4 = create_image_reader_by_z_stack(color_channel=c_chan)
    rdr5 = create_image_reader_by_z_stack(color_channel=c_chan)
    rdr6 = create_image_reader_by_z_stack(color_channel=c_chan)

    in_frames = map(flatten_image, rdr4)
    in_frames = map(PIL_render, in_frames)
    writeGif('non-stabilized_flattened_balls.gif', in_frames, duration=0.3)\

    ini_xy_T_mats = [xy_T_mat for xy_T_mat in basic_aligner(rdr5, axis=0)]
    inter_frames = map(flatten_image, transform_reader_flow(rdr6, ini_xy_T_mats, axis=0))
    inter_frames = map(PIL_render, inter_frames)
    writeGif('xy-stabilized_flattened_balls.gif', inter_frames, duration=0.3)

    zy_T_mats = [zy_T_mat for zy_T_mat in basic_aligner(rdr1, axis=1)]
    xy_T_mats = [xy_T_mat for xy_T_mat in basic_aligner(transform_reader_flow(rdr2, zy_T_mats, axis=1), axis=0)]
    fin_frames = map(flatten_image, transform_reader_flow(transform_reader_flow(rdr3, zy_T_mats, axis=1), xy_T_mats, axis=0))
    fin_frames = map(PIL_render, fin_frames)
    writeGif('stabilized_flattened_balls.gif', fin_frames, duration=0.3)


def create_stabilized_matrices():
    c_chan = 2

    rdr1 = create_image_reader_by_z_stack(color_channel=2)
    rdr2 = create_image_reader_by_z_stack(color_channel=2)

    zy_T_mats = [zy_T_mat for zy_T_mat in basic_aligner(rdr1, axis=1)]
    xy_T_mats = [xy_T_mat for xy_T_mat in basic_aligner(transform_reader_flow(rdr2, zy_T_mats, axis=1), axis=0)]

    return zy_T_mats, xy_T_mats


def extract_ball_center(_3D_vol):
    _3D_vol = filters.gaussian_filter(_3D_vol, sigma = 3)
    # neighborhood = morphology.generate_binary_structure(len(_3D_vol.shape), 1)
    _3D_vol[_3D_vol < np.percentile(_3D_vol, 99)] = 0.0
    # _3D_vol = morphology.grey_opening(_3D_vol, structure=neighborhood)
    _3D_vol /= np.max(_3D_vol)
    plmaxs = detect_local_maxima(_3D_vol)
    _3D_maxs = np.zeros(_3D_vol.shape)
    _3D_maxs[plmaxs] = 1
    _3D_maxs = morphology.binary_closing(_3D_maxs).astype(np.float32)
    nz = np.nonzero(_3D_maxs)
    msk = _3D_vol[nz] > 0.1
    nzs = np.array(nz).T[msk, :]
    return tuple(nzs.T.tolist())


def show_ball_centers(_3D_vol, final_points):
    _3D_maxs = np.zeros(_3D_vol.shape)
    _3D_maxs[final_points] = 1
    render_2_colors(_3D_vol, _3D_maxs)


def sorting_inspector(distance_matrix, best_hits_to_show):

    def inner_round(n_1Darray, init_state=[0]):

        def inner_printer(best_argument):
            # print best_argument, '|', "{0:.2f}".format(n_1Darray[best_argument]), '\t',
            pass

        sorting_args = np.argsort(n_1Darray)
        first_sorting_args = sorting_args.tolist()
        # print init_state[0], '\t|\t',
        init_state[0] += 1
        map(inner_printer, first_sorting_args[:best_hits_to_show])
        if n_1Darray[first_sorting_args[0]] < 9.0:
            if n_1Darray[first_sorting_args[1]] > 9.0:
                # print ''
                return first_sorting_args[0]
            else:
                # print 'failed!'
                return -first_sorting_args[0]
        else:
            # print 'failed!'
            return -10000

    index_map = np.apply_along_axis(inner_round, arr=distance_matrix, axis=0)

    return index_map


def calculate_alignement_and_vectors(final_points, final_points2):
    a_final_points = np.array(final_points).T
    a_final_points2 = np.array(final_points2).T
    img = cdist(a_final_points, a_final_points2)
    # get all the forward mappings.
    #    -10 000 are dissapearing, mappings;
    #    indexes that are not in the "mapped to" are the appearing ones
    #    -N are the mappings that can be attributed to several points at the same time => We ignore them for now
    _map = np.abs(sorting_inspector(img, 2))
    init_idx = np.arange(0, _map.shape[0], 1)
    init_idx = init_idx[_map != 10000]
    _map = _map[_map != 10000]

    vects = a_final_points2[init_idx, :] - a_final_points[_map, :]
    compensation = np.median(vects, 0)
    vects = np.apply_along_axis(lambda x: x - compensation, 1, vects)

    vector_lengths = np.apply_along_axis(np.linalg.norm, 1, vects[1:])
    sigvects = vects[vector_lengths > 1.41]  #circle pixel-> out
    sigbases = a_final_points[vector_lengths > 1.41]

    return init_idx, _map, sigbases, sigvects


if __name__ == '__main__':

    create_tri_alignement()

    # zy_T_mats, xy_T_mats = create_stabilized_matrices()
    # dump((zy_T_mats, xy_T_mats), open('loc1_dmp.dmp', 'w'))

    # zy_T_mats, xy_T_mats = load(open('loc1_dmp.dmp'))

    # rdr3 = create_image_reader_by_z_stack(color_channel=1)
    # rdr0 = transform_reader_flow(transform_reader_flow(rdr3, zy_T_mats, axis=1), xy_T_mats, axis=0)
    #
    # _3D_vol = rdr0.next()
    # final_points = extract_ball_center(_3D_vol)
    #
    # fpoints_list = [final_points]
    # av_list = []
    # timer1('setup')
    # for _3D_vol in rdr0:
    #     final_points2 = extract_ball_center(_3D_vol)
    #     a_v = calculate_alignement_and_vectors(final_points, final_points2)
    #     av_list.append(a_v)
    #     final_points = final_points2
    #     fpoints_list.append(final_points)
    #     timer1('.')
    #
    # dump(fpoints_list, open('loc2_dmp.dmp', 'w'))

    # dump(av_list, open('loc3_dmp.dmp', 'w'))

    # av_list = load(open('loc3_dmp.dmp'))

    # rdr4 = create_image_reader_by_z_stack(color_channel=1)
    # rdr0 = transform_reader_flow(transform_reader_flow(rdr4, zy_T_mats, axis=1), xy_T_mats, axis=0)
    # fin_frames = map(flatten_image, rdr0)
    # dump(fin_frames, open('loc4_dmp.dmp', 'w'))

    # fin_frames = load(open('loc4_dmp.dmp'))

    # corrval = 10
    #
    # for _i, (frame, (_, _, sigbases, sigvects)) in enumerate(zip(fin_frames, av_list)):
    #     plt.imshow(frame, cmap='gray', interpolation='nearest')
    #     for sigbase, sigvect in zip(sigbases, sigvects):
    #         sigbase = np.roll(sigbase[1:], 1)
    #         sigvect = np.roll(sigvect[1:], 1)*corrval
    #         plt.arrow(sigbase[0], sigbase[1], sigvect[0], sigvect[1], fc="r", ec="r",
    #                   head_width=0.5*corrval, head_length=corrval)
    #     plt.savefig('temp%s.png'%format(_i, "02d"))
    #     plt.clf()

    # imported_frames = []
    # for f_name in sorted(os.listdir('.')):
    #     if 'temp' and '.png' in f_name:
    #         print f_name
    #         imported_frames.append(PIL.Image.open(f_name))
    #
    # writeGif('Stabilized_alls_with_vectors.gif', imported_frames, duration=0.3)
