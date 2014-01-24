__author__ = 'ank'

import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
import math
from time import time


ImageRoot = "/home/ank/Documents/var"
gray = PIL.Image.open(ImageRoot + '/n4.png')
bw = np.asarray(gray).copy()
X, Y = np.meshgrid(range(bw.shape[0] + 1), range(bw.shape[1] + 1))


def Ising_Round(matrix, decay_Factor, max_dist, threshold):
    # TODO: pay attention to the range_assymetry
    start = time()
    new_mat = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if i in range(max_dist + 1, matrix.shape[0] - max_dist - 1) \
                and j in range(max_dist + 1, matrix.shape[1] - max_dist - 1) \
                and matrix[i, j] > threshold:
                new_index = 0
                for k in range(i - max_dist, i + max_dist):
                    for l in range(j - max_dist, j + max_dist):
                        new_index += matrix[k, l] * (decay_Factor) ** (math.sqrt((k - i) ** 2 + (l - j) ** 2))
                new_mat[i, j] = new_index
    new_mat = new_mat * (2 ** 8 - 1) / new_mat.max()
    print time() - start
    return new_mat


def remap(matrix, contrast_threshold, ising_threshold, ins_lum_thresh, real_ins_lum_thresh):
    grad = np.gradient(matrix)
    absgrad = np.absolute(grad[1]) + np.absolute(grad[0])
    absgrad = absgrad * (2 ** 8 - 1) / absgrad.max()
    dergrad = Ising_Round(absgrad, 0.9, 2, ising_threshold)

    non_sign_grad = dergrad < contrast_threshold
    sign_grad = dergrad > contrast_threshold
    gradient_average = np.average(matrix[sign_grad])
    insufficent_lum = matrix < ins_lum_thresh * gradient_average
    real_insufficient_lum = matrix < real_ins_lum_thresh * gradient_average
    too_low = np.logical_or(np.logical_and(non_sign_grad, insufficent_lum), real_insufficient_lum)

    filtered_matrix = matrix.copy()
    filtered_matrix = filtered_matrix * (2 ** 8 - 1) / filtered_matrix.max()

    filtered_matrix[np.logical_not(too_low)] = \
        (filtered_matrix[np.logical_not(too_low)] - filtered_matrix[np.logical_not(too_low)].min())\
        * 255 / filtered_matrix[np.logical_not(too_low)].max()
    filtered_matrix[too_low] = 0

    filtered_matrix=np.round(filtered_matrix)
    plt.hist(filtered_matrix.flatten(), bins=256, log=True)
    plt.show()

    im = plt.pcolormesh(X, Y, filtered_matrix.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()


remap(bw,40,10, 0.7, 0.5)

def segment(matrix):

    position2Cluster = {}
    ClusterNumber2Cluster = {}
    Typic_cluster_dict = {
        'size':0,
        'members':[],
        'member_values':[],
    }

    toRun = np.nonzero(matrix)

    def inital_cluster(matrix):
        seeds = matrix == matrix.max()


    def expand_clusters():
        pass


def border_detect(matrix, contrast_threshold, ising_threshold, ins_lum_thresh):
    grad = np.gradient(matrix)
    absgrad = np.absolute(grad[1]) + np.absolute(grad[0])
    absgrad = absgrad * (2 ** 8 - 1) / absgrad.max()
    dergrad = Ising_Round(absgrad, 0.9, 2, ising_threshold)
    # dergrad=absgrad
    im = plt.pcolormesh(X, Y, dergrad.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()

    non_sign_grad = dergrad < contrast_threshold
    sign_grad = dergrad > contrast_threshold
    gradient_average = np.average(matrix[sign_grad])
    insufficent_lum = matrix < ins_lum_thresh * gradient_average
    print np.logical_and(non_sign_grad, insufficent_lum)

    filtered_matrix = matrix.copy()
    filtered_matrix = filtered_matrix * (2 ** 8 - 1) / filtered_matrix.max()

    filtered_matrix[sign_grad] = np.round(filtered_matrix[sign_grad] / 255.0) * 255

    im = plt.pcolormesh(X, Y, filtered_matrix.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()