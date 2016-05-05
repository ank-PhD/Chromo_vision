import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from chiffatools.dataviz import smooth_histogram
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
from scipy.stats import pearsonr, ks_2samp
from csv import writer
import traceback

from skimage.segmentation import random_walker
from skimage.morphology import opening, closing, erosion
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.feature import blob_dog, blob_log, blob_doh


plt.figure(figsize=(20.0, 15.0))

ImageRoot = "L:\\Users\\linghao\\Spinning Disk\\03182016-Ry129-131\\Ry130\\hs30min"
main_root = "L:\\Users\\linghao\\Data for quantification"
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
        plt.histogram(current_image.flatten(), 100)
        plt.show()

    stabilized = (current_image - np.min(current_image))/(float(2**bits) - np.min(current_image))
    stabilized[stabilized < alpha_clean*np.median(stabilized)] = 0

    if debug:
        print np.max(current_image), np.min(current_image), np.median(current_image)
        plt.histogram(current_image.flatten(), 100)
        plt.show()

    if smoothing_px:
        for i in range(0, stabilized.shape[0]):
            stabilized[i, :, :] = gaussian_filter(stabilized[i, :, :],
                                                  smoothing_px, mode='constant')
            stabilized[stabilized < 5*np.mean(stabilized)] = 0

    if debug:
        print np.max(current_image), np.min(current_image), np.median(current_image)
        plt.histogram(current_image.flatten(), 100)
        plt.show()

        for i in range(0, stabilized.shape[0]):
            plt.imshow(stabilized[i, :, :] > mcc_cutoff, cmap='gray', vmin=0., vmax=1.)
            plt.show()

    return stabilized


def watershed_segment(base):
    selem = disk(2)

    GFP_collector = np.sum(base, axis=0)
    markers = np.zeros(GFP_collector.shape, dtype=np.uint8)
    # watershed segment
    markers[GFP_collector > np.mean(GFP_collector)*5] = 2
    markers[GFP_collector < np.mean(GFP_collector)*0.25] = 1
    labels = random_walker(GFP_collector, markers, beta=10, mode='bf')
    # round up the labels and set the background to 0 from 1.
    labels = closing(labels, selem)
    labels -= 1
    # prepare distances for the watershed
    distance = ndi.distance_transform_edt(labels)
    local_maxi = peak_local_max(distance,
                                indices=False,  # we want the image mask, not peak position
                                min_distance=15,  # about half of a bud with our size
                                threshold_abs=5,  # allows to clear the noise
                                labels=labels)
    # we fuse the labels that are close together that escaped the min distance in local_maxi
    local_maxi = ndi.convolve(local_maxi, np.ones((5, 5)), mode='constant', cval=0.0)
    # finish the watershed
    markers2 = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels2 = watershed(-distance, markers2, mask=labels)

    plt.subplot(231)
    plt.imshow(GFP_collector, interpolation='nearest')

    plt.subplot(232)
    plt.imshow(markers, cmap='hot', interpolation='nearest')

    plt.subplot(233)
    plt.imshow(labels, cmap='gray', interpolation='nearest')

    plt.subplot(234)
    plt.imshow(distance, cmap='gray', interpolation='nearest')

    plt.subplot(235)
    plt.imshow(markers2, cmap=plt.cm.spectral, interpolation='nearest')

    plt.subplot(236)
    plt.imshow(labels2, cmap=plt.cm.spectral, interpolation='nearest')

    plt.show()

def analyze(name_pattern, w1448, w2561, prefilter=True):

    if prefilter:
        watershed_segment(w1448)

    # TODO: for yeast, add over-fluorescent cell removing logic on w1448 channel
    #  A way to do is to project all the non-negative pixel mask onto a single plane
    # then perform random walker segmentation and use the segmented mask as the reference

    # TODO: normalize 2561 channel to span 0-1, ALWAYS, since it is our detection back-bone
    seg0 = [name_pattern]
    seg1 = [np.sum(w1448*w1448), np.sum(w2561*w2561), np.sum(w1448*w2561)]
    seg2 = [np.sum(w2561[w1448 > mcc_cutoff])/np.sum(w2561),
            np.sum(w1448[w2561 > mcc_cutoff])/np.sum(w1448)]
    seg3 = [np.mean(w1448[w2561 > mcc_cutoff]), np.mean(w2561[w2561 > mcc_cutoff])]

    return seg0 + seg1 + seg2 + seg3



def test_yeast():
    replicas = defaultdict(lambda: [0, 0])

    for img in os.listdir(ImageRoot):
        if '.TIF' in img and '_thumb_' not in img:
            img_codename = img.split(' ')[0].split('_')

            print '%s image was parsed, code: %s' % (img, img_codename)

            current_image = Image.open(os.path.join(ImageRoot, img))
            replicas[img_codename[0]+'-'+img_codename[1]][translator[img_codename[2]]] =\
                pre_process(current_image)

    for replica, (w1448, w2561) in replicas.iteritems():
        print analyze(replica, w1448, w2561)
        # print replica
        # # reference for the quantities calculated: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074624/
        # print 'L2 stats'
        # print '\t w1448: %s \n\t w2561: %s \n\t cross: %s' % (np.sum(w1448*w1448),
        #                                                       np.sum(w2561*w2561),
        #                                                       np.sum(w1448*w2561))
        # # print '\t PCC: %s, p-val: %s' % pearsonr(w1448.flatten(), w2561.flatten())
        # # print '\t KS: %s, p-val: %s' % ks_2samp(w1448.flatten(), w2561.flatten())
        # # print '\t MOC:', np.sum(w1448*w2561)/np.sqrt(np.sum(w1448*w1448)*np.sum(w2561*w2561))
        # print 'current cutoff:', mcc_cutoff*100, '% of max intensity'
        # print 'MCC \n\t w2561 in w1448:', np.sum(w2561[w1448 > mcc_cutoff])/np.sum(w2561)*100, '%'
        # print '\t > w1448 in w2561:', np.sum(w1448[w2561 > mcc_cutoff])/np.sum(w1448)*100, '%'
        # print 'average qualifying voxel intensity:'
        # print '\t w1448', np.mean(w1448[w2561 > mcc_cutoff])
        # print '\t w2561', np.mean(w2561[w2561 > mcc_cutoff])
        # print '\n'


def test_blobs_in_yeast():
    pass


def test_mammalians():
    ImageRoot = "L:\\Users\\linghao\\Data for quantification\\Mammalian\\DM_Splitted channels"

    for img in os.listdir(ImageRoot):
        if '.tif' in img:
            img_codename = img.split(' ')[0].split('_')
            print '%s image was parsed, code: %s' % (img, img_codename)
            current_image = Image.open(os.path.join(ImageRoot, img))
            pre_process(current_image, 30, 0)


def mammalian_traversal(path):
    main_root = "L:\\Users\\linghao\\Data for quantification\\Mammalian"


def yeast_traversal(path):
    main_root = "L:\\Users\\linghao\\Data for quantification\\Yeast"


def main_traversal(path):
    replicas = defaultdict(lambda: [0, 0])

    results_collector = []
    sucker_list = []

    for current_location, sub_directories, files in os.walk(path):
        print current_location
        print '\t', files
        color = None
        name_pattern = None

        # TODO: re-implement the iterator as matching pair poppinq queue to save up on RAM

        if files:
            if 'Mammalian' in current_location:
                if 'Splitted' not in current_location:
                    continue
                else:
                    print 'case 2'
                    for img in files:
                        if ('.TIF' in img or '.tif' in img) and '_thumb_' not in img:
                            print img,
                            img_codename = img.split('-')
                            prefix = current_location.split('\\')[4:]
                            color = translator[img_codename[0]]
                            name_pattern = ' - '.join(prefix+img_codename[1:])
                            current_image = Image.open(os.path.join(current_location, img))
                            replicas[name_pattern][color] = pre_process(current_image, 30, 0)
                            print '-', name_pattern, color
            else:
                print 'case 1'
                for img in files:
                    if ('.TIF' in img or '.tif' in img) and '_thumb_' not in img:
                        print img,
                        img_codename = img.split(' ')[0].split('_')
                        prefix = current_location.split('\\')[4:]
                        color = translator[img_codename[-1]]
                        name_pattern = ' - '.join(prefix+img_codename[:-1])
                        # current_image = Image.open(os.path.join(current_location, img))
                        # replicas[name_pattern][color] = pre_process_yeast(current_image)
                        print '-', name_pattern, color

        for name_pattern, (w1448, w2561) in replicas.iteritems():
            # TODO: for yeast, add over-fluorescent cell removing logic on w1448 channel
            #  A way to do is to project all the non-negative pixel mask onto a single plane
            # then perform random walker segmentation and use the segmented mask as the reference

            # TODO: normalize 2561 channel to span 0-1, ALWAYS, since it is our detection back-bone
            print name_pattern
            try:
                seg0 = [name_pattern]
                seg1 = [np.sum(w1448*w1448), np.sum(w2561*w2561), np.sum(w1448*w2561)]
                seg2 = [np.sum(w2561[w1448 > mcc_cutoff])/np.sum(w2561),
                        np.sum(w1448[w2561 > mcc_cutoff])/np.sum(w1448)]
                seg3 = [np.mean(w1448[w2561 > mcc_cutoff]),
                        np.mean(w2561[w2561 > mcc_cutoff])]
                results_collector.append(seg0+seg1+seg2+seg3)
            except Exception as my_exception:
                print traceback.print_exc(my_exception)
                sucker_list.append(name_pattern)

        replicas = defaultdict(lambda: [0, 0])

    with open('results-nn.csv', 'wb') as output:
        csv_writer = writer(output, )
        csv_writer.writerow(header)
        for item in results_collector:
            csv_writer.writerow(item)

    print sucker_list

if __name__ == "__main__":
    test_yeast()
    # main_traversal(main_root)
    test_mammalians()
