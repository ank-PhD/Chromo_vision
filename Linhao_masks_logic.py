import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from collections import defaultdict
from csv import writer
import traceback
from chiffatools.high_level_os_methods import safe_dir_create
from skimage.segmentation import random_walker
from skimage.morphology import closing
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk


ImageRoot = "L:\\Users\\linghao\\Spinning Disk\\03182016-Ry129-131\\Ry130\\hs30min"
main_root = "L:\\Users\\linghao\\Data for quantification"

safe_dir_create('verification_bank')
scaling_factor = (1.0, 1.0, 3.5)

red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)

v_min = 0.6
mcc_cutoff = 0.05

translator = {'w1488': 0,
              'w2561': 1,
              'C1': 1,
              'C2': 0}

# 0 is the protein marker
# 1 is the mitochondria marker

dtype2bits = {'uint8': 8,
              'uint16': 16,
              'uint32': 32}

header = ['name pattern', 'GFP', 'mito marker', 'cross',
              'MCC mito in GFP %', 'MCC GFP in mito %',
              'AQVI GFP', 'AQVI mito']


def tiff_stack_2_np_arr(tiff_stack):
    """
    Loads the image from the tiff stack to a 3D numpy array

    :param tiff_stack:
    :return:
    """
    stack = [np.array(tiff_stack)]
    try:
        while 1:
            tiff_stack.seek(tiff_stack.tell() + 1)
            stack.append(np.array(tiff_stack))
    except EOFError:
        pass

    return np.array(stack)


def pre_process(tiff_stack, alpha_clean=5, smoothing_px=1.5, debug=False):
    """
    Performs the initial conversion and de-noising of the tiff stack

    :param tiff_stack:
    :param alpha_clean:
    :param smoothing_px:
    :return:
    """
    current_image = tiff_stack_2_np_arr(tiff_stack)
    bits = dtype2bits[current_image.dtype.name]

    if debug:
        print np.max(current_image), np.min(current_image), np.median(current_image)
        plt.hist(current_image.flatten(), 100)
        plt.show()

    stabilized = (current_image - np.min(current_image))/(float(2**bits) - np.min(current_image))
    stabilized[stabilized < alpha_clean*np.median(stabilized)] = 0

    if debug:
        print np.max(current_image), np.min(current_image), np.median(current_image)
        plt.hist(current_image.flatten(), 100)
        plt.show()

    if smoothing_px:
        for i in range(0, stabilized.shape[0]):
            stabilized[i, :, :] = gaussian_filter(stabilized[i, :, :],
                                                  smoothing_px, mode='constant')
            stabilized[stabilized < 5*np.mean(stabilized)] = 0

    if debug:
        print np.max(current_image), np.min(current_image), np.median(current_image)
        plt.hist(current_image.flatten(), 100)
        plt.show()

        for i in range(0, stabilized.shape[0]):
            plt.imshow(stabilized[i, :, :] > mcc_cutoff, cmap='gray', vmin=0., vmax=1.)
            plt.show()

    return stabilized


def determine_dynamic_outliers(labels2, GFP_collector):
    segments = []
    labels3 = np.zeros_like(labels2).astype(np.float64)
    qualifying_GFP = GFP_collector > np.median(GFP_collector[GFP_collector > 0])
    for i in range(1, np.max(labels2)+1):
        current_mask = labels2 == i
        coll_sel = GFP_collector[np.logical_and(current_mask, qualifying_GFP)]

        # if debug:
        #     print coll_sel.shape
        #     print coll_sel
        #     plt.subplot(221)
        #     plt.imshow(current_mask, cmap='gray', interpolation='nearest')
        #     plt.subplot(222)
        #     plt.imshow(qualifying_GFP, cmap='gray', interpolation='nearest')
        #     plt.subplot(223)
        #     plt.imshow(np.logical_and(current_mask, qualifying_GFP),
        #                cmap='gray', interpolation='nearest')
        #     plt.subplot(224)
        #     plt.imshow(GFP_collector, cmap='gray', interpolation='nearest')
        #     plt.show()

        GFP_percentile = np.percentile(coll_sel, 50)

        GFP_average = np.average(GFP_collector[np.logical_and(current_mask,
                                                                 GFP_collector>GFP_percentile)])
        segments.append(GFP_average)
        labels3[current_mask] = GFP_average

    argsort = np.argsort(np.array(segments))
    segments = sorted(segments)

    support = range(0, len(segments))
    # requires at least 10 pts to work
    slope, intercept, _, _, _ = stats.linregress(np.array(support)[1:5],
                                                 np.array(segments)[1:5])
    # stderr = np.std(np.array(segments)[1:5])
    stderr = 0.05/8.
    predicted = intercept+slope*np.array(support)
    closure = argsort[np.array(support)[np.array(predicted+stderr*8) < np.array(segments)]]
    print closure, closure.tolist() != []
    labels4 = np.zeros_like(labels2).astype(np.uint8)
    if closure.tolist() != []:
        print 'enter closure correction'
        for idx in closure.tolist():
            labels4[labels2 == idx+1] = 1  # indexing starts from 1, not 0 for the labels
            print 'updated %s' % (idx+1)

    return labels4, qualifying_GFP, segments, predicted, labels3, stderr


def segment_out_ill_cells(name_pattern, base, debug=False):
    """
    Logic to segment out overly lumiscent cells

    :param name_pattern:
    :param base:
    :param debug:
    :return:
    """
    selem = disk(2)

    GFP_collector = np.sum(base, axis=0)
    markers = np.zeros(GFP_collector.shape, dtype=np.uint8)
    # watershed segment
    markers[GFP_collector > np.mean(GFP_collector)*2] = 2
    markers[GFP_collector < np.mean(GFP_collector)*0.20] = 1
    labels = random_walker(GFP_collector, markers, beta=10, mode='bf')
    # round up the labels and set the background to 0 from 1.
    labels = closing(labels, selem)
    labels -= 1
    # prepare distances for the watershed
    distance = ndi.distance_transform_edt(labels)
    local_maxi = peak_local_max(distance,
                                indices=False,  # we want the image mask, not peak position
                                min_distance=10,  # about half of a bud with our size
                                threshold_abs=10,  # allows to clear the noise
                                labels=labels)
    # we fuse the labels that are close together that escaped the min distance in local_maxi
    local_maxi = ndi.convolve(local_maxi, np.ones((5, 5)), mode='constant', cval=0.0)
    # finish the watershed
    markers2 = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels2 = watershed(-distance, markers2, mask=labels)
    # there is still the problem with first element not being labeled properly.

    # calculating the excessively luminous outliers
    labels4, qualifying_GFP, segments, predicted, labels3, stderr = \
        determine_dynamic_outliers(labels2, GFP_collector)

    if debug:
        plt.figure(figsize=(20.0, 15.0))
        plt.title(name_pattern)

        plt.subplot(241)
        plt.imshow(GFP_collector, interpolation='nearest')

        plt.subplot(242)
        plt.imshow(markers, cmap='hot', interpolation='nearest')

        plt.subplot(243)
        plt.imshow(labels, cmap='gray', interpolation='nearest')

        plt.subplot(244)
        plt.imshow(labels2, cmap=plt.cm.spectral, interpolation='nearest')

        plt.subplot(245)
        plt.imshow(labels3, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.subplot(246)
        plt.plot(segments, 'ko')
        plt.plot(predicted, 'r')
        plt.plot(predicted+stderr*8, 'g')  # arbitrary coefficient, but works well
        plt.plot(predicted-stderr*8, 'g')

        plt.subplot(247)
        plt.imshow(labels4, cmap='gray', interpolation='nearest')

        plt.subplot(248)
        plt.imshow(qualifying_GFP)

        plt.savefig('verification_bank/%s.png' % name_pattern)
        # plt.show()
        plt.clf()

    return labels4


def analyze(name_pattern, marked_prot, organelle_marker, prefilter=True, debug=False):
    """
    Stitches the analysis pipeline together

    :param name_pattern:
    :param marked_prot:
    :param organelle_marker:
    :param prefilter:
    :param debug:
    :return:
    """

    if debug:
        GFP_collector_1 = np.sum(marked_prot, axis=0)
        mCh_collector_1 = np.sum(organelle_marker, axis=0)

    # GFP-unhealthy cells detection logic
    if prefilter:
        cancellation_mask = segment_out_ill_cells(name_pattern, marked_prot, debug)

        new_w1448 = np.zeros_like(marked_prot)
        new_w2561 = np.zeros_like(organelle_marker)

        new_w1448[:, np.logical_not(cancellation_mask)] = marked_prot[:, np.logical_not(cancellation_mask)]
        new_w2561[:, np.logical_not(cancellation_mask)] = organelle_marker[:, np.logical_not(cancellation_mask)]

        marked_prot = new_w1448
        organelle_marker = new_w2561

    if debug:
        GFP_collector = np.sum(marked_prot, axis=0)
        mCh_collector = np.sum(organelle_marker, axis=0)

        plt.subplot(221)
        plt.imshow(GFP_collector_1, cmap='Greens')

        plt.subplot(222)
        plt.imshow(mCh_collector_1, cmap='Reds')

        plt.subplot(223)
        plt.imshow(GFP_collector, cmap='Greens')

        plt.subplot(224)
        plt.imshow(mCh_collector, cmap='Reds')

        plt.savefig('verification_bank/core-%s.png' % name_pattern)
        # plt.show()
        plt.clf()

    # TODO: normalize 2561 channel to span 0-1. This is not necessary needed since the mitochondria
    # appeared to be well above thershold in practice.
    seg0 = [name_pattern]
    seg1 = [np.sum(marked_prot * marked_prot), np.sum(organelle_marker * organelle_marker), np.sum(marked_prot * organelle_marker)]
    seg2 = [np.sum(organelle_marker[marked_prot > mcc_cutoff]) / np.sum(organelle_marker),
            np.sum(marked_prot[organelle_marker > mcc_cutoff]) / np.sum(marked_prot)]
    seg3 = [np.median(marked_prot[organelle_marker > mcc_cutoff]),
            np.median(organelle_marker[organelle_marker > mcc_cutoff])]

    return seg0 + seg1 + seg2 + seg3


def mammalian_traversal():
    main_root = "L:\\Users\\linghao\\Data for quantification\\Mammalian"
    replicas = defaultdict(lambda: [0, 0])
    results_collector = []
    sucker_list = []

    for current_location, sub_directories, files in os.walk(main_root):
        print current_location
        print '\t', files
        color = None
        name_pattern = None
        if files and 'Splitted' in current_location:
            for img in files:  # TODO: remove the debug flag here
                if '.tif' in img and '_thumb_' not in img:
                    img_codename = img.split('-')
                    prefix = current_location.split('\\')[4:]
                    color = translator[img_codename[0]]
                    name_pattern = ' - '.join(prefix+img_codename[1:])
                    current_image = Image.open(os.path.join(current_location, img))
                    print '%s image was parsed, code: %s %s' % (img, name_pattern, color)
                    replicas[name_pattern][color] = pre_process(current_image)

            for name_pattern, (w1448, w2561) in replicas.iteritems():
                print name_pattern
                try:
                    results_collector.append(analyze(name_pattern, w1448, w2561,
                                                     prefilter=False, debug=True))
                except Exception as my_exception:
                    print traceback.print_exc(my_exception)
                    sucker_list.append(name_pattern)

            replicas = defaultdict(lambda: [0, 0])

    with open('results-nn-mammalian.csv', 'wb') as output:
        csv_writer = writer(output, )
        csv_writer.writerow(header)
        for item in results_collector:
            csv_writer.writerow(item)

    print sucker_list


def yeast_traversal():
    main_root = "L:\\Users\\linghao\\Data for quantification\\Yeast"
    replicas = defaultdict(lambda: [0, 0])
    results_collector = []
    sucker_list = []

    for current_location, sub_directories, files in os.walk(main_root):
        print current_location
        print '\t', files
        color = None
        name_pattern = None
        if files:
            for img in files:
                if '.TIF' in img and '_thumb_' not in img:
                    img_codename = img.split(' ')[0].split('_')
                    prefix = current_location.split('\\')[4:]
                    color = translator[img_codename[-1]]
                    name_pattern = ' - '.join(prefix+img_codename[:-1])
                    current_image = Image.open(os.path.join(current_location, img))
                    print '%s image was parsed, code: %s %s' % (img, name_pattern, color)
                    replicas[name_pattern][color] = pre_process(current_image)

            for name_pattern, (w1448, w2561) in replicas.iteritems():
                print name_pattern
                try:
                    results_collector.append(analyze(name_pattern, w1448, w2561,
                                                     prefilter=True, debug=True))
                except Exception as my_exception:
                    print traceback.print_exc(my_exception)
                    sucker_list.append(name_pattern)

            replicas = defaultdict(lambda: [0, 0])

    with open('results-nn-yeast.csv', 'wb') as output:
        csv_writer = writer(output, )
        csv_writer.writerow(header)
        for item in results_collector:
            csv_writer.writerow(item)

    print sucker_list


if __name__ == "__main__":
    yeast_traversal()
    # mammalian_traversal()
