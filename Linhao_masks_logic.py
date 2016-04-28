import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from chiffatools.dataviz import smooth_histogram
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
from scipy.stats import pearsonr, ks_2samp

plt.figure(figsize=(20.0, 15.0))

ImageRoot = "L:\\Users\\linghao\\Spinning Disk\\03182016-Ry129-131\\Ry130\\hs30min"
scaling_factor = (1.0, 1.0, 3.5)
smoothing_px = 1.5
red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)
v_min = 0.6
mcc_cutoff = 0.05

translator = {'w1488': 0,
              'w2561': 1}


def preprocess(current_image):

    stack = [np.array(current_image)]

    try:
        while 1:
            current_image.seek(current_image.tell()+1)
            stack.append(np.array(current_image))

    except EOFError:
        pass

    current_image = np.array(stack)

    # line below removed because we are interested only in the intensity of x inside the y

    # # print np.max(current_image), np.min(current_image), np.median(current_image)
    #
    # stabilized = (current_image - np.min(current_image)) / \
    #              float(np.max(current_image) - np.min(current_image))

    stabilized = (current_image - np.min(current_image))/(float(2**16) - np.min(current_image))
    stabilized[stabilized < 10*np.median(stabilized)] = 0

    # line below removed because we are interested only in the intensity of x inside the y

    for i in range(0, stabilized.shape[0]):
        stabilized[i, :, :] = gaussian_filter(stabilized[i, :, :], smoothing_px,
                                              mode='constant')

    stabilized[stabilized < 5*np.mean(stabilized)] = 0

    # # print np.max(stabilized), np.min(stabilized), np.median(stabilized), np.mean(stabilized)
    #
    # stabilized = (current_image - np.min(current_image)) / \
    #              float(np.max(current_image) - np.min(current_image))

    # smooth_histogram(stabilized.flatten())
    # plt.show()

    # for i in range(0, stabilized.shape[0]):
    #     plt.imshow(stabilized[i, :, :] > mcc_cutoff, cmap='gray', vmin=0., vmax=1.)
    #     plt.show()

    return stabilized


replicas = defaultdict(lambda: [0, 0])

for img in os.listdir(ImageRoot):
    if '.TIF' in img and '_thumb_' not in img:
        img_codename = img.split(' ')[0].split('_')

        # print '%s image was parsed, code: %s' % (img, img_codename)

        current_image = Image.open(os.path.join(ImageRoot, img))
        replicas[img_codename[0]+'-'+img_codename[1]][translator[img_codename[2]]] =\
            preprocess(current_image)


for replica, (w1448, w2561) in replicas.iteritems():
    print replica
    # reference for the quantities calculated: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074624/
    print 'L2 stats'
    print '\t w1448: %s \n\t w2561: %s \n\t cross: %s' % (np.sum(w1448*w1448),
                                                          np.sum(w2561*w2561),
                                                          np.sum(w1448*w2561))
    # print '\t PCC: %s, p-val: %s' % pearsonr(w1448.flatten(), w2561.flatten())
    # print '\t KS: %s, p-val: %s' % ks_2samp(w1448.flatten(), w2561.flatten())
    # print '\t MOC:', np.sum(w1448*w2561)/np.sqrt(np.sum(w1448*w1448)*np.sum(w2561*w2561))
    print 'current cutoff:', mcc_cutoff*100, '% of max intensity'
    print 'MCC \n\t w2561 in w1448:', np.sum(w2561[w1448 > mcc_cutoff])/np.sum(w2561)*100, '%'
    print '\t > w1448 in w2561:', np.sum(w1448[w2561 > mcc_cutoff])/np.sum(w1448)*100, '%'
    print 'average qualifying voxel intensity:'
    print '\t w1448', np.mean(w1448[w2561 > mcc_cutoff])
    print '\t w2561', np.mean(w2561[w2561 > mcc_cutoff])
    print '\n'
