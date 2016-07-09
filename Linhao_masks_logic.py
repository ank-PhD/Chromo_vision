import os
import traceback
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from collections import defaultdict
from csv import writer
from chiffatools.high_level_os_methods import safe_dir_create
from skimage.segmentation import random_walker
from skimage.morphology import closing
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.morphology import skeletonize, medial_axis
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.filters import threshold_otsu, threshold_adaptive

# ImageRoot = "L:\\Users\\linghao\\Spinning Disk\\03182016-Ry129-131\\Ry130\\hs30min"
# main_root = "L:\\Users\\linghao\\Data for quantification"

safe_dir_create('verification_bank')
scaling_factor = (1.0, 1.0, 3.5)

red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)

v_min = 0.6
mcc_cutoff = 0.05

# 0 is the protein marker
# 1 is the mitochondria marker

translator = {'w1488': 0,
              'w2561': 1,
              'C1': 1,
              'C2': 0}


dtype2bits = {'uint8': 8,
              'uint16': 16,
              'uint32': 32}

header = ['name pattern', 'GFP', 'mito marker', 'cross',
          'MCC mito in GFP %', 'MCC GFP in mito %',
          'AQVI GFP', 'AQVI mito', 'ill', 'cell_no',
          'mean width', 'mean length', 'cells with intact mitochondria %',
          'area of intact mitochondria %']


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


# TODO: this function does two things. In fact it needs to be refactored to only do one
def gamma_stabilize_and_smooth(tiff_stack, alpha_clean=5, smoothing_px=1.5, debug=False):
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


def detect_ill_cells(cell_labels, gfp_collector, debug=False):
    """
    Logic that determines outliers that look like dead cells in the gfp channel projection.
    Requires at least 5 non-dead cells in the image.

    :param cell_labels:
    :param gfp_collector:
    :param debug:
    :return:
    """
    cells_average_gfp = []
    average_gfp_in_cell = np.zeros_like(cell_labels).astype(np.float64)
    qualifying_gfp = gfp_collector > np.median(gfp_collector[gfp_collector > 0])

    for i in range(1, np.max(cell_labels)+1):

        current_mask = cell_labels == i
        current_cell_gfp = gfp_collector[np.logical_and(current_mask, qualifying_gfp)]

        if len(current_cell_gfp) == 0:
            continue

        if debug:
            print current_cell_gfp.shape
            print current_cell_gfp
            plt.subplot(221)
            plt.imshow(current_mask, cmap='gray', interpolation='nearest')
            plt.subplot(222)
            plt.imshow(qualifying_gfp, cmap='gray', interpolation='nearest')
            plt.subplot(223)
            plt.imshow(np.logical_and(current_mask, qualifying_gfp),
                       cmap='gray', interpolation='nearest')
            plt.subplot(224)
            plt.imshow(gfp_collector, cmap='gray', interpolation='nearest')
            plt.show()

        gfp_percentile = np.percentile(current_cell_gfp, 50)
        gfp_average = np.average(gfp_collector[np.logical_and(current_mask,
                                                              gfp_collector > gfp_percentile)])
        cells_average_gfp.append(gfp_average)
        average_gfp_in_cell[current_mask] = gfp_average

    arg_sort = np.argsort(np.array(cells_average_gfp))
    cells_average_gfp = sorted(cells_average_gfp)

    cell_no = range(0, len(cells_average_gfp))
    # requires at least 5 pts to work
    regression_base = min(len(cells_average_gfp)-5, 10)
    slope, intercept, _, _, _ = stats.linregress(np.array(cell_no)[1:regression_base],
                                                 np.array(cells_average_gfp)[1:regression_base])

    # std_err = np.std(np.array(cells_average_gfp)[1:5])
    std_err = (np.max(np.array(cells_average_gfp)[1:regression_base])
               - np.min(np.array(cells_average_gfp)[1:regression_base]))/2
    # std_err = 0.05/8.

    non_dying_predicted = intercept + slope * np.array(cell_no)
    non_dying_cells = arg_sort[np.array(cell_no)[np.array(non_dying_predicted+std_err*8) <
                                                     np.array(cells_average_gfp)]]
    # print non_dying_cells, non_dying_cells.tolist() != []
    non_dying_cells_mask = np.zeros_like(cell_labels).astype(np.uint8)

    if non_dying_cells.tolist() != []:
        print 'enter non_dying_cells correction'
        for idx in non_dying_cells.tolist():
            non_dying_cells_mask[cell_labels == idx + 1] = 1  # indexing starts from 1, not 0 for the labels
            print 'updated %s' % (idx+1)

    return non_dying_cells_mask, qualifying_gfp, cells_average_gfp,\
           non_dying_predicted, average_gfp_in_cell, std_err


def segment_out_ill_cells(name_pattern, base, debug=False):
    """
    Logic to segment out overly gfp luminiscent cells, taken as a proxi for the state of illness

    :param name_pattern:
    :param base:
    :param debug:
    :return:
    """

    # TODO: spearate segmentation from ill cells deletion

    # TODO: use OTSU threshold for GFP thresholding

    sel_elem = disk(2)

    gfp_collector = np.sum(base, axis=0)
    gfp_clustering_markers = np.zeros(gfp_collector.shape, dtype=np.uint8)

    # random walker segment
    gfp_clustering_markers[gfp_collector > np.mean(gfp_collector)*2] = 2
    gfp_clustering_markers[gfp_collector < np.mean(gfp_collector)*0.20] = 1
    labels = random_walker(gfp_collector, gfp_clustering_markers, beta=10, mode='bf')

    # round up the labels and set the background to 0 from 1
    labels = closing(labels, sel_elem)
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
    expanded_maxi_markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    segmented_cells = watershed(-distance, expanded_maxi_markers, mask=labels)
    # there is still the problem with first element not being labeled properly.

    # calculating the excessively luminous outliers
    non_dying_cells_mask, qualifying_gfp, segments, predicted, labels3, std_err = \
        detect_ill_cells(segmented_cells, gfp_collector)

    if debug:
        plt.figure(figsize=(20.0, 15.0))
        plt.title(name_pattern)

        plt.subplot(241)
        plt.imshow(gfp_collector, interpolation='nearest')

        plt.subplot(242)
        plt.imshow(gfp_clustering_markers, cmap='hot', interpolation='nearest')

        plt.subplot(243)
        plt.imshow(labels, cmap='gray', interpolation='nearest')

        plt.subplot(244)
        plt.imshow(segmented_cells, cmap=plt.cm.spectral, interpolation='nearest')

        plt.subplot(245)
        plt.imshow(labels3, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.subplot(246)
        plt.plot(segments, 'ko')
        plt.plot(predicted, 'r')
        plt.plot(predicted+std_err*8, 'g')  # arbitrary coefficient, but works well
        plt.plot(predicted-std_err*8, 'g')

        plt.subplot(247)
        plt.imshow(non_dying_cells_mask, cmap='gray', interpolation='nearest')

        plt.subplot(248)
        plt.imshow(qualifying_gfp)

        plt.savefig('verification_bank/%s.png' % name_pattern)
        # plt.show()
        plt.clf()

    return non_dying_cells_mask, segmented_cells


def skeletonize_mitochondria(mCh_channel):

    mch_collector = np.max(mCh_channel, axis=0)  # TODO: check how max affects v.s. sum
    labels = np.zeros(mch_collector.shape, dtype=np.uint8)

    # thresh = np.max(mch_collector)/2.
    thresh = threshold_otsu(mch_collector)
    # TODO: use adaptative threshold? => otsu seems to be sufficient in this case
    # http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html#sphx
    # -glr-auto-examples-xx-applications-plot-thresholding-py
    #  log-transform? => Nope, does not work
    # TODO: hessian/laplacian of gaussian blob detection?

    labels[mch_collector > thresh] = 1
    skeleton2 = skeletonize(labels)
    skeleton, distance = medial_axis(labels, return_distance=True)
    active_threshold = np.mean(mch_collector[labels]) * 5

    # print active_threshold
    transform_filter = np.zeros(mch_collector.shape, dtype=np.uint8)
    transform_filter[np.logical_and(skeleton > 0, mch_collector > active_threshold)] = 1
    skeleton = transform_filter * distance

    skeleton_ma = np.ma.masked_array(skeleton, skeleton > 0)
    skeleton_convolve = ndi.convolve(skeleton_ma, np.ones((3, 3)), mode='constant', cval=0.0)
    divider_convolve = ndi.convolve(transform_filter, np.ones((3, 3)), mode='constant', cval=0.0)
    skeleton_convolve[divider_convolve > 0] = skeleton_convolve[divider_convolve > 0] \
                                              / divider_convolve[divider_convolve > 0]
    new_skeleton = np.zeros_like(skeleton)
    new_skeleton[skeleton2] = skeleton_convolve[skeleton2]
    skeleton = new_skeleton

    return labels, mch_collector, skeleton, transform_filter


def measure_skeleton_stats(numbered_labels, skeleton, transform_filter):

    numbered_skeleton, object_no = ndi.label(transform_filter, structure=np.ones((3, 3)))

    # print numbered_skeleton.shape, np.min(numbered_skeleton), np.max(numbered_skeleton)
    collector = []
    paint_area = np.zeros_like(numbered_labels)
    paint_length = np.zeros_like(numbered_labels)

    # TODO: if cell skeleton is inside the numbered_labels contig, paint the whole label with it to get area.
    # problem: double match
    # solution: restrict skeleton to one, select numbered_labels on non-nul skeleton, then use the numbered_labels

    for contig_no in range(1, object_no + 1):
        vals = skeleton[numbered_skeleton == contig_no]
        current_label = np.max(numbered_labels[numbered_skeleton == contig_no])
        area, support = (np.sqrt(np.sum((numbered_labels == current_label).astype(np.int8))),
                         len(vals))

        if area < 3:
            skeleton[numbered_skeleton == contig_no] = 0
            transform_filter[numbered_skeleton == contig_no] = 0

        else:
            paint_area[numbered_labels == current_label] = area
            paint_length[numbered_labels == current_label] = support
            collector.append([area, support])

    collector = np.array(collector)

    return collector, paint_length, paint_area


def compute_mito_fragmentation(name_pattern, labels, mch_collector, skeleton, transform_filter,
                               segmented_cells, debug=False):

    numbered_lables, lables_no = ndi.label(labels, structure=np.ones((3, 3)))

    collector, paint_length, paint_area = measure_skeleton_stats(numbered_lables,
                                                                  skeleton,
                                                                  transform_filter)

    classification_pad = np.zeros_like(segmented_cells)
    classification_roll = []

    for i in range(1, np.max(segmented_cells)+1):
        pre_mask = segmented_cells == i
        current_mask = np.logical_and(pre_mask, labels > 0)
        if len(paint_length[current_mask]) == 0:
            classification_roll.append(-1)
            classification_pad[pre_mask] = -1
        else:
            length = np.mean(np.unique(paint_length[current_mask]))
            area = np.mean(np.unique(paint_area[current_mask]))
            if length < 20 or area < 5:
                classification_pad[pre_mask] = 1
                classification_roll.append(1)
            else:
                classification_pad[pre_mask] = 2
                classification_roll.append(2)

    intact = np.logical_and(paint_length > 20, paint_area > 5)
    broken = np.logical_and(np.logical_or(paint_length < 20, paint_area < 5), paint_area > 1)

    mito_summary = np.sum(intact.astype(np.int8)) / \
                   (np.sum(intact.astype(np.int8)) + np.sum(broken.astype(np.int8)))

    if len(collector) == 0:
        mean_width, mean_length = [np.NaN, np.NaN]
    else:
        mean_width, mean_length = np.mean(collector, axis=0).tolist()

    if debug:

        ax1 = plt.subplot(231)
        plt.title(name_pattern)
        plt.imshow(mch_collector, cmap='Reds')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='k')

        plt.subplot(232, sharex=ax1, sharey=ax1)
        plt.title('width ; length - av: %.2f ; %.2f' % (mean_width, mean_length))
        plt.imshow(skeleton, cmap=plt.cm.spectral, interpolation='nearest')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='w')

        plt.subplot(233, sharex=ax1, sharey=ax1)
        plt.imshow(segmented_cells, cmap=plt.cm.spectral, interpolation='nearest')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='w')

        plt.subplot(234, sharex=ax1, sharey=ax1)
        plt.imshow(numbered_lables, cmap=plt.cm.spectral, interpolation='nearest')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='w')

        plt.subplot(235, sharex=ax1, sharey=ax1)
        plt.imshow(classification_pad, cmap=plt.cm.spectral, interpolation='nearest')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='w')

        plt.subplot(236, sharex=ax1, sharey=ax1)
        plt.imshow(paint_area, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.contour(labels, [0.5], colors='w')

        # plt.show()

    classification_array = np.array(classification_roll)
    classification_array = classification_array[classification_array > 0] - 1

    return collector, [mean_width, mean_length, np.mean(classification_array), mito_summary]


def analyze(name_pattern, marked_prot, organelle_marker,
            segment_out_ill=True, debug=False, per_cell=False):
    """
    Stitches the analysis pipeline together

    :param name_pattern:
    :param marked_prot:
    :param organelle_marker:
    :param segment_out_ill:
    :param debug:
    :return:
    """

    if per_cell and not segment_out_ill:
        raise Exception("cannot perform per-cell output without proper segmentation")

    if debug:
        gfp_collector_1 = np.sum(marked_prot, axis=0)
        mch_collector_1 = np.sum(organelle_marker, axis=0)

    # GFP-unhealthy cells detection logic
    if segment_out_ill:
        cancellation_mask, segmented_cells = segment_out_ill_cells(name_pattern, marked_prot, debug)

        new_w1448 = np.zeros_like(marked_prot)
        new_w2561 = np.zeros_like(organelle_marker)

        new_w1448[:, np.logical_not(cancellation_mask)] = marked_prot[:, np.logical_not(cancellation_mask)]
        new_w2561[:, np.logical_not(cancellation_mask)] = organelle_marker[:, np.logical_not(cancellation_mask)]

        marked_prot = new_w1448
        organelle_marker = new_w2561

    if debug:
        gfp_collector = np.sum(marked_prot, axis=0)
        mch_collector = np.sum(organelle_marker, axis=0)

        plt.subplot(221)
        plt.imshow(gfp_collector_1, cmap='Greens')

        plt.subplot(222)
        plt.imshow(mch_collector_1, cmap='Reds')

        plt.subplot(223)
        plt.imshow(gfp_collector, cmap='Greens')

        plt.subplot(224)
        plt.imshow(mch_collector, cmap='Reds')

        plt.savefig('verification_bank/core-%s.png' % name_pattern)
        # plt.show()
        plt.clf()

    labels, mch_collector, skeleton, transform_filter = skeletonize_mitochondria(organelle_marker)

    if debug:
        compute_mito_fragmentation(name_pattern, labels, mch_collector, skeleton, transform_filter,
                                   segmented_cells, debug=True)
        plt.savefig('verification_bank/mitochondria-%s.png' % name_pattern)
        plt.clf()

    if per_cell:

        seg_stack = []

        for cell_no in range(1, np.max(segmented_cells)+1):
            current_mask = segmented_cells == cell_no
            current_mask = current_mask[:, :]
            ill = np.median(cancellation_mask[current_mask])

            _organelle_marker = np.zeros_like(organelle_marker)
            _organelle_marker[:, current_mask] = organelle_marker[:, current_mask]

            _marked_prot = np.zeros_like(marked_prot)
            _marked_prot[:, current_mask] = marked_prot[:, current_mask]

            _labels = np.zeros_like(labels)
            _labels[current_mask] = labels[current_mask]

            _mch_collector = np.zeros_like(mch_collector)
            _mch_collector[current_mask] = mch_collector[current_mask]

            _skeleton = np.zeros_like(skeleton)
            _skeleton[current_mask] = skeleton[current_mask]

            _transform_filter = np.zeros_like(transform_filter)
            _transform_filter[current_mask] = transform_filter[current_mask]

            mito_char_collector, seg4 = compute_mito_fragmentation(
                name_pattern,
                _labels, _mch_collector, _skeleton, _transform_filter, segmented_cells)

            if debug:
                plt.subplot(221)
                plt.imshow(current_mask, cmap='Greens')

                plt.subplot(222)
                mp_2d = np.sum(_marked_prot, axis=0)
                plt.imshow(mp_2d, cmap='Greens')

                plt.subplot(223)
                om_2d = np.sum(_organelle_marker, axis=0)
                plt.imshow(om_2d, cmap='Reds')

                # plt.show()
                plt.clf()

            seg0 = [name_pattern]
            seg1 = [np.sum(_marked_prot * _marked_prot),
                    np.sum(_organelle_marker * _organelle_marker),
                    np.sum(_marked_prot * _organelle_marker)]
            seg2 = [np.sum(_organelle_marker[_marked_prot > mcc_cutoff]) / np.sum(
                _organelle_marker),
                    np.sum(_marked_prot[_organelle_marker > mcc_cutoff]) / np.sum(_marked_prot)]
            seg3 = [np.median(_marked_prot[_organelle_marker > mcc_cutoff]),
                    np.median(_organelle_marker[_organelle_marker > mcc_cutoff]),
                    ill,
                    cell_no]

            seg_stack += [seg0 + seg1 + seg2 + seg3 + seg4]

        return seg_stack

    else:
        # suggested: normalize 2561 channel to span 0-1. This is not necessary needed since the
        # mitochondria appeared to be well above thershold in practice.

        mito_char_collector, seg4 = compute_mito_fragmentation(name_pattern,
            labels, mch_collector, skeleton, transform_filter, segmented_cells)

        seg0 = [name_pattern]
        seg1 = [np.sum(marked_prot * marked_prot),
                np.sum(organelle_marker * organelle_marker),
                np.sum(marked_prot * organelle_marker)]
        seg2 = [np.sum(organelle_marker[marked_prot > mcc_cutoff]) / np.sum(organelle_marker),
                np.sum(marked_prot[organelle_marker > mcc_cutoff]) / np.sum(marked_prot)]
        seg3 = [np.median(marked_prot[organelle_marker > mcc_cutoff]),
                np.median(organelle_marker[organelle_marker > mcc_cutoff]),
                'NaN',
                'NaN']

        return [seg0 + seg1 + seg2 + seg3 + seg4]


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
                    replicas[name_pattern][color] = gamma_stabilize_and_smooth(current_image)

            for name_pattern, (w1448, w2561) in replicas.iteritems():
                print name_pattern
                try:
                    results_collector.append(analyze(name_pattern, w1448, w2561,
                                                     segment_out_ill=False, debug=True))
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


def yeast_traversal(per_cell):
    # main_root = "L:\\Users\\linghao\\Data for quantification\\Yeast\\NEW data for analysis"
    main_root = "L:\\Users\\jerry\\Image\\ForAndrei"
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
                    replicas[name_pattern][color] = gamma_stabilize_and_smooth(current_image)

            for name_pattern, (w1448, w2561) in replicas.iteritems():
                print name_pattern
                try:
                    results_collector += analyze(name_pattern, w1448, w2561,
                                                 segment_out_ill=True, debug=True,
                                                 per_cell=per_cell)
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
    yeast_traversal(per_cell=False)
    # mammalian_traversal()
