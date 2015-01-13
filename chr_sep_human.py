__author__ = 'ank'

import numpy as np
import matplotlib.pyplot as plt
import PIL
import mdp
from itertools import product
from pickle import load, dump
from os import path
from time import time
from skimage.segmentation import random_walker, mark_boundaries
from skimage.morphology import convex_hull_image, label, dilation, disk
from skimage.filter import gaussian_filter
from skimage.measure import perimeter
from matplotlib import colors
from pylab import get_cmap
# from skimage.measure import label
# todo and management for varying window sizes for Gabor filters
# todo: darkest spot => in testing

# todo: threshold-filtering instead of diffusion clustering
# todo:

selem = disk(20)
debug = False
timing = False
plt.figure(figsize=(30.0, 20.0))

def rs(matrix, name):
    plt.title(name)
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    if debug:
        plt.show()
    plt.clf()

def debug_wrapper(funct):

    def check_matrix(*args,**kwargs):
        result = funct(*args, **kwargs)
        if debug:
            if type(result) is not tuple:
                rs(result, funct.__name__)
            else:
                rs(result[0], funct.__name__)
        check_matrix.__name__ = funct.__name__
        check_matrix.__doc__ = funct.__doc__
        return result

    return check_matrix

def time_wrapper(funct):

    def time_execution(*args,**kwargs):
        start = time()
        result = funct(*args, **kwargs)
        if timing:
            print funct.__name__, time()-start
        time_execution.__doc__ = funct.__doc__
        return result

    return time_execution


@time_wrapper
@debug_wrapper
def import_image(image_to_load):
    gray = PIL.Image.open(image_to_load).convert('L')
    bw = np.asarray(gray)
    bw = bw - np.min(bw)
    bw = bw.astype(np.float64)/float(np.max(bw))
    return bw


@time_wrapper
@debug_wrapper
def import_edited(buffer_directory):
    if path.exists(buffer_directory+'-'+"EDIT_ME2.tif"):
        col = PIL.Image.open(buffer_directory+'-'+"EDIT_ME2.tif")
    else:
        col = PIL.Image.open(buffer_directory+'-'+"EDIT_ME.tif")
    gray = col.convert('L')
    bw = np.asarray(gray).copy()
    bw[bw<=120] = 0
    bw[bw>120] = 1
    return bw


@time_wrapper
@debug_wrapper
def gabor(bw_image, freq, scale, scale_distortion=1., self_cross=False, field=10, phi=np.pi, abes=False):

    # gabor filter normalization with respect to the surface convolution
    def check_integral(gabor_filter):
        ones = np.ones(gabor_filter.shape)
        avg = np.average(ones*gabor_filter)
        return gabor_filter-avg

    quality = 16
    pi = np.pi
    orientations = np.arange(0., pi, pi/quality).tolist()
    size = (field, field)
    sgm = (5*scale, 3*scale*scale_distortion)

    nfilters = len(orientations)
    gabors = np.empty((nfilters, size[0], size[1]))
    for i, alpha in enumerate(orientations):
        arr = mdp.utils.gabor(size, alpha, phi, freq, sgm)
        if self_cross:
            if self_cross == 1:
                arr = np.minimum(arr, mdp.utils.gabor(size, alpha+pi/2, phi, freq, sgm))
            if self_cross == 2:
                arr -= mdp.utils.gabor(size, alpha+pi/2, phi, freq, sgm)
        arr = check_integral(arr)
        gabors[i, :, :] = arr
        if debug:
            plt.subplot(6, 6, i+1)
            plt.title('%s, %s, %s, %s'%('{0:.2f}'.format(alpha), '{0:.2f}'.format(phi), freq, sgm))
            plt.imshow(arr, cmap = 'gray', interpolation='nearest')
    if debug:
        plt.show()
        plt.clf()
    node = mdp.nodes.Convolution2DNode(gabors, mode='valid', boundary='fill', fillvalue=0, output_2d=False)

    cim = node.execute(bw_image[np.newaxis, :, :])[0, :, :, :]
    re_cim = np.zeros((cim.shape[0], cim.shape[1] + field - 1, cim.shape[2] + field - 1))
    re_cim[:, field/2-1:-field/2, field/2-1:-field/2] = cim
    cim = re_cim

    if abes:
        sum2 = np.sum(np.abs(cim), axis=0)
        sum2 /= np.max(sum2)
    else:
        sum2 = - np.sum(cim, axis=0)
        sum2[sum2>0] /= np.max(sum2)
        sum2[sum2<0] /= -np.min(sum2)
    return sum2, cim


@time_wrapper
@debug_wrapper
def cluster_by_diffusion(data):
    markers = np.zeros(data.shape, dtype=np.uint8)
    markers[data < -0.15] = 1
    markers[data > 0.15] = 2
    labels2 = random_walker(data, markers, beta=10, mode='cg_mg')
    return labels2


@time_wrapper
@debug_wrapper
def cluster_process(labels, original, activations):
    rbase = np.zeros(labels.shape)
    rubase = np.zeros(labels.shape)
    rubase[range(0,20),:] = 1
    rubase[:,range(0,20)] = 1
    rubase[range(-20,-1),:] = 1
    rubase[:,range(-20,-1)] = 1
    for i in range(1, int(np.max(labels))+1):
        base = np.zeros(labels.shape)
        base[labels==i] = 1
        li = len(base.nonzero()[0])
        if li>0:
            hull = convex_hull_image(base)
            lh =len(hull.nonzero()[0])
            sel_org = base*original
            sel_act = base*activations
            cond = (li > 4000 and float(lh) / float(li) < 1.07 and perimeter(base)**2.0 / li < 30) or np.max(base * rubase) > 0.5
            # print li>4000 and float(lh)/float(li)<1.07, perimeter(base)**2.0/li<30, np.max(base*rubase)>0.5, np.min(original[base>0])
            hard_array =[li > 4000, float(lh) / float(li) < 1.07]
            optional_array = [perimeter(base)**2.0/li < 25,
                              np.percentile(sel_org[sel_org>0], 5) > 0.2,
                              np.percentile(sel_act, 90) - np.percentile(sel_act, 90)]
            print hard_array, optional_array
            if debug and li>1000:
                rs(base,'subspread cluster')
            if cond:
                rbase = rbase + base
    rbase[rubase.astype(np.bool)] = 1
    return dilation(rbase, selem)


@time_wrapper
def repaint_culsters(clusterNo=100):
    prism_cmap = get_cmap('prism')
    prism_vals = prism_cmap(np.arange(clusterNo))
    prism_vals[0] = [0, 0, 0, 1]
    costum_cmap = colors.LinearSegmentedColormap.from_list('my_colormap', prism_vals)
    return costum_cmap


@time_wrapper
@debug_wrapper
def compare_orthogonal_selectors(voluminal_crossed_matrix):
    var2 = np.max(voluminal_crossed_matrix - np.roll(voluminal_crossed_matrix, voluminal_crossed_matrix.shape[0]/2, 0), axis=0)
    var2 /= np.max(var2)
    return var2


@time_wrapper
def human_loop(buffer_directory, image_to_import, stack_type):
    start = time()
    bw = import_image(image_to_import)
    lbw = np.log(bw+0.001)
    lbw = lbw - np.min(lbw)
    lbw = lbw/np.max(lbw)
    # rs(lbw, 'log-bw')
    sum1, _ = gabor(lbw, 1/32., 2, scale_distortion=2., field=40,  phi=np.pi/2, abes=True)

    if stack_type == 0:
        # Human
        sum2,_ = gabor(bw, 1/8., 1, self_cross=1, field=20)

        sum20,_ = gabor(bw, 1/6., 1, field=20)
        sum20[sum20>-0.2] = 0
        sum2 = sum2 + sum20

    elif stack_type == 1:
        # Mice
        sum2,_ = gabor(bw, 1/8., 1, field=20)

        sum20,_ = gabor(bw, 1/4., 0.5, field=20)
        sum20[sum20>-0.1] = 0
        sum2 = sum2 + sum20

        # crossed antisense selctor
        _, st = gabor(bw, 1/8., 1, self_cross=2, field=20)
        sum20 = compare_orthogonal_selectors(st)
        sum20[sum20<0.65] = 0
        if debug:
            rs(sum20, 'selected cross')
        sum2 = sum2 - sum20
    else:
        raise Exception('Unrecognized chromosome type')

    if debug:
            rs(sum2, 'sum2-definitive')

    bw_blur = gaussian_filter(lbw, 5)
    bwth = np.zeros(bw_blur.shape)
    # if debug:
    #     plt.hist(bw)
    #     plt.show()
    #     plt.hist(bw_blur)
    #     plt.show()

    bwth[bw_blur > np.percentile(bw_blur, 80)] = 1 #<"we need somehow to adjust this in a non-parametric way."
    # plt.hist(bw_blur)
    # plt.show()
    bwth[sum1 > 0.45] = 0
    clsts = (label(bwth)+1)*bwth

    # rs(clsts, 'cluster_labels')

    rbase = cluster_process(clsts, bw, sum2)

    rim = PIL.Image.fromarray((rbase*254).astype(np.uint8))
    rim.save(buffer_directory+'-'+"I_AM_UNBROKEN_NUCLEUS.bmp")

    sum22 = np.copy(sum2)
    sum22[sum2<0] = 0
    d_c = cluster_by_diffusion(sum2)
    reim = PIL.Image.fromarray((bw/np.max(bw)*254).astype(np.uint8))
    reim.save(buffer_directory+'-'+"I_AM_THE_ORIGINAL.tif")
    seg_dc = (label(d_c, neighbors=4)+1)*(d_c-1)
    redd = set(seg_dc[rbase>0.01].tolist())
    for i in redd:
        seg_dc[seg_dc==i] = 0
    d_c = d_c*0
    d_c[seg_dc>0] = 1
    int_arr = np.asarray(np.dstack((d_c*254, d_c*254, d_c*0)), dtype=np.uint8)
    if debug:
            rs(int_arr, 'segmentation mask')
    msk = PIL.Image.fromarray(int_arr)
    msk.save(buffer_directory+'-'+"EDIT_ME.tif")
    dump(bw ,open(buffer_directory+'-'+'DO_NOT_TOUCH_ME.dmp','wb'))
    return time()-start


@time_wrapper
def human_afterloop(output_directory, pre_time, fle_name, buffer_directory):
    start2 = time()

    d_c = import_edited(buffer_directory)
    rebw = load(open(buffer_directory+'-'+'DO_NOT_TOUCH_ME.dmp','rb'))
    seg_dc = (label(d_c,neighbors=4)+1)*d_c
    if np.max(seg_dc)<4:
        return 'FAILED: mask for %s looks unsegmented' % fle_name

    colormap = repaint_culsters(int(np.max(seg_dc)))

    segs = len(set(seg_dc.flatten().tolist()))-1

    # shows the result before saving the clustering and printing to the user the number of the images
    plt.subplot(1,2,1)
    plt.title(fle_name)
    plt.imshow(rebw, cmap='gray', interpolation='nearest')

    plt.subplot(1,2,2)
    plt.title('Segmentation - clusters: %s'%str(segs))
    plt.imshow(mark_boundaries(rebw, d_c))
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest', alpha=0.3)

    plt.show()

    plt.imshow(mark_boundaries(rebw, d_c))
    plt.imshow(seg_dc, cmap=colormap, interpolation='nearest', alpha=0.3)

    plt.savefig(path.join(output_directory, fle_name+'_%s_clusters.png'%str(segs)), dpi=500, bbox_inches='tight', pad_inches=0.0)

    return fle_name+'\t clusters: %s,\t total time : %s'%(segs, "{0:.2f}".format(time()-start2+pre_time))

if __name__ == "__main__":
    pass