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
from skimage.filters import threshold_otsu


safe_dir_create('verification_bank')
scaling_factor = (1.0, 1.0, 3.5)

red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)

v_min = 0.6
mcc_cutoff = 0.02

# 0 is the protein marker
# 1 is the mitochondria marker

translator = {'w1488': 0,
              'w2561': 1,
              'C1': 1,
              'C2': 0}


dtype2bits = {'uint8': 8,
              'uint16': 16,
              'uint32': 32}


header = ['name pattern', 'DHE total intensity', 'Average qualifying DHE voxel intensity']


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


def segment_out_ill_cells(name_pattern, dhe_marker, debug=True):
    """
    Logic to segment out overly gfp luminescent cells, taken as a proxi for the state of illness

    :param name_pattern:
    :param base:
    :param debug:
    :return:
    """
    dead_cell_strategy = 'dynamic'
    # dead_cell_strategy = 'fixed'

    # TODO: use OTSU threshold for  thresholding

    sel_elem = disk(2)

    dhe_collector = np.sum(dhe_marker, axis=0)
    dhe_max_projection = np.max(dhe_marker, axis=0)
    dhe_clustering_markers = np.zeros(dhe_collector.shape, dtype=np.uint8)

    print dhe_collector.shape
    print dhe_marker.shape

    # random walker segment

    print np.mean(dhe_collector)*2, np.mean(dhe_collector)*0.20, threshold_otsu(dhe_collector)

    dhe_clustering_markers[dhe_collector > np.mean(dhe_collector)*1.5] = 2
    dhe_clustering_markers[dhe_collector < np.mean(dhe_collector)*0.10] = 1
    labels = random_walker(dhe_collector, dhe_clustering_markers, beta=10, mode='bf')

    # round up the labels and set the background to 0 from 1
    labels = closing(labels, sel_elem)
    labels -= 1

    # prepare distances for the watershed
    distance = ndi.distance_transform_edt(labels)
    local_maxi = peak_local_max(distance,
                                indices=False,  # we want the image mask, not peak position
                                min_distance=5,  # about half of a bud with our size
                                threshold_abs=5,  # allows to clear the noise
                                labels=labels)

    # we fuse the labels that are close together that escaped the min distance in local_maxi
    local_maxi = ndi.convolve(local_maxi, np.ones((5, 5)), mode='constant', cval=0.0)

    # finish the watershed
    expanded_maxi_markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    segmented_cells = watershed(-distance, expanded_maxi_markers, mask=labels)
    # there is still the problem with first element not being labeled properly.

    # calculating the excessively luminous outliers
    # dying_cells_mask = dhe_collector > threshold_otsu(dhe_collector)
    dying_cells_mask, qualifying_gfp, segments, predicted, labels3, std_err = \
        detect_ill_cells(segmented_cells, dhe_collector)

    fixed_dying = dhe_max_projection > 0.3

    print np.percentile(dhe_max_projection, 99), np.median(dhe_max_projection)

    if dead_cell_strategy == 'fixed':
        final_dying_cells_mask = fixed_dying

    if dead_cell_strategy == 'dynamic':
        final_dying_cells_mask = np.logical_or(dying_cells_mask, fixed_dying)

    my_cmap = plt.cm.prism
    my_cmap.set_under('k')

    if debug:
        plt.figure(figsize=(20.0, 15.0))

        ax1 = plt.subplot(241)
        plt.title('raw image - current strategy %s'% dead_cell_strategy)
        plt.imshow(dhe_collector, interpolation='nearest')

        plt.subplot(242, sharex=ax1, sharey=ax1)
        plt.title('diffusion segmentation markers')
        plt.imshow(dhe_clustering_markers, cmap='hot', interpolation='nearest')

        plt.subplot(243, sharex=ax1, sharey=ax1)
        plt.title('segmentation mask')
        plt.imshow(labels, cmap='gray', interpolation='nearest')

        plt.subplot(244, sharex=ax1, sharey=ax1)
        plt.title('region split')
        plt.imshow(segmented_cells, cmap=my_cmap, interpolation='nearest', vmin=.001)

        plt.subplot(245, sharex=ax1, sharey=ax1)
        plt.title('average intensity')
        plt.imshow(labels3, cmap='hot', interpolation='nearest')
        plt.colorbar()

        plt.subplot(246)
        plt.title('sorted average intensities')
        plt.plot(segments, 'ko')
        plt.plot(predicted, 'r')
        plt.plot(predicted+std_err*8, 'g')  # arbitrary coefficient, but works well
        plt.plot(predicted-std_err*8, 'g')

        plt.subplot(247, sharex=ax1, sharey=ax1)
        plt.title('exclusion mask')
        plt.imshow(final_dying_cells_mask, cmap='gray', interpolation='nearest')

        plt.subplot(248, sharex=ax1, sharey=ax1)
        plt.title('remaining regions mask')
        plt.imshow(qualifying_gfp)

        plt.savefig('verification_bank/%s.png' % name_pattern)
        # plt.show()
        plt.clf()

    return final_dying_cells_mask, segmented_cells


def analyze(name_pattern, dhe_marker, segment_out_ill=True, debug=False):
    """
    Stitches the analysis pipeline together

    :param name_pattern:
    :param marked_prot:
    :param dhe_marker:
    :param segment_out_ill:
    :param debug:
    :return:
    """
    marked_prot = None

    old_dhe_collector = np.sum(dhe_marker, axis=0)

    if segment_out_ill:
        cancellation_mask, segmented_cells = segment_out_ill_cells(name_pattern, dhe_marker, debug)
        new_dhe_marker = np.zeros_like(dhe_marker)
        new_dhe_marker[:, np.logical_not(cancellation_mask)] = dhe_marker[:, np.logical_not(cancellation_mask)]
        dhe_marker = new_dhe_marker

    new_dhe_collector = np.sum(dhe_marker, axis=0)

    if debug:

        ax1 = plt.subplot(221)
        plt.title('raw image - current strategy')
        plt.imshow(old_dhe_collector, interpolation='nearest')

        plt.subplot(222, sharex=ax1, sharey=ax1)
        plt.title('diffusion segmentation markers')
        plt.imshow(cancellation_mask, cmap='Greys', interpolation='nearest')

        plt.subplot(223, sharex=ax1, sharey=ax1)
        plt.title('segmentation mask')
        plt.imshow(new_dhe_collector, interpolation='nearest')

        plt.subplot(224, sharex=ax1, sharey=ax1)
        plt.title('region split')
        plt.imshow(np.sum((dhe_marker > mcc_cutoff).astype(np.int8), axis=0),
                   interpolation='nearest', vmin=.001)

        plt.savefig('verification_bank/%s-mcc_cutoff.png' % name_pattern)
        # plt.show()
        plt.clf()

    seg0 = [name_pattern]
    seg1 = [np.sum(dhe_marker * dhe_marker)]
    seg3 = [np.median(dhe_marker[dhe_marker > mcc_cutoff])]

    return [seg0 + seg1 + seg3]
    # return []

def yeast_traversal(per_cell, debug=False):
    # main_root = "L:\\Users\\linghao\\Data for quantification\\Yeast\\NEW data for analysis"
    # main_root = "L:\\Users\\jerry\\Image\\ForAndrei"
    main_root = "C:\\Users\\Andrei\\Downloads\\raw data\\fluorescense"
    replicas = defaultdict(lambda: [0, 0])
    results_collector = []
    sucker_list = []

    for current_location, sub_directories, files in os.walk(main_root):
        print current_location
        print '\t', files
        if files:
            for img in files:
                if '.TIF' in img or '.tiff' in img:
                    img_codename = img.split('.')[0].split(' ')
                    prefix = current_location.split('\\')[4:]
                    name_pattern = ' - '.join(prefix+img_codename[:-1])
                    image_path = Image.open(os.path.join(current_location, img))
                    print '%s image was parsed, code: %s' % (img, name_pattern)
                    dhe_marker = gamma_stabilize_and_smooth(image_path)
                    try:
                        result = analyze(name_pattern, dhe_marker,
                                          segment_out_ill=True, debug=debug)
                        print result
                        results_collector += result
                    except Exception as my_exception:
                        print traceback.print_exc(my_exception)
                        sucker_list.append(name_pattern)

    with open('results-nn-yeast-kai-dynamic.csv', 'wb') as output:
        csv_writer = writer(output, )
        csv_writer.writerow(header)
        for item in results_collector:
            csv_writer.writerow(item)

    print sucker_list


if __name__ == "__main__":
    yeast_traversal(per_cell=False, debug=True)
    # mammalian_traversal()
