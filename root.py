__author__ = 'ank'

import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
import math
from time import time
import random as rand


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

def to255(matrix):
    return (matrix-matrix.min()).copy()*255.0/(matrix-matrix.min()).max()


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
    # filtered_matrix = filtered_matrix * (2 ** 8 - 1) / filtered_matrix.max()

    filtered_matrix[np.logical_not(too_low)] = \
        (filtered_matrix[np.logical_not(too_low)] - filtered_matrix[np.logical_not(too_low)].min())\
        * (2 ** 15 - 1) / filtered_matrix[np.logical_not(too_low)].max()
    filtered_matrix[too_low] = 0
    filtered_matrix=np.round(filtered_matrix)

    filtered_255_matrix = to255(filtered_matrix)

    return filtered_255_matrix

def render_matrix(matrix):
    im = plt.pcolormesh(X, Y, matrix.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()

def render_with_path(matrix,path_matrix):
    im =  plt.pcolormesh(X, Y, to255(matrix).transpose(), cmap='gray')
    im2 = plt.pcolormesh(X, Y, to255(path_matrix).transpose(), cmap='Reds')
    im2.set_alpha(0.5)
    plt.colorbar(im)
    plt.colorbar(im2)
    plt.show()

def Directional_gradient(matrix):
    right, up = np.gradient(matrix,1)
    left, down = (-1*right.copy(), -1*up.copy())
    return up, down, right, left

def segment(matrix):

    position2Cluster = {}
    ClusterNumber2Cluster = {}
    Typic_cluster_dict = {
        'size':0,
        'members':[],
        'member_values':[],
        'current_fronteer_tolerance':(0,255),
        'finished':(),
    }

    gl_toRun = matrix > 0.1

    up, down, right, left = Directional_gradient(matrix)

    def neighbours(i_j, toRun):
        i=i_j[0]
        j=i_j[1]
        neigh2grad = {}
        neigh2grad[(i+1,j)] = (right[i,j], toRun[i+1,j]) # right
        neigh2grad[(i,j-1)] = (down[i,j], toRun[i,j-1]) # down
        neigh2grad[(i,j+1)] = (up[i,j], toRun[i,j+1]) # up
        neigh2grad[(i-1,j)] = (left[i,j], toRun[i-1,j]) # left

        return neigh2grad

    def grad_walk(error_accept):
        rw_matrix = np.zeros(matrix.shape)
        seeds = matrix == matrix.max()
        toRun = np.logical_and(gl_toRun, np.logical_not(seeds))
        starter = (0,0)
        while True:
            rand_i = rand.randint(1,matrix.shape[0]-1)
            rand_j = rand.randint(1,matrix.shape[1]-1)
            if toRun[rand_i, rand_j]:
                starter = (rand_i, rand_j)
                break

        run=[starter]
        pointer=0
        flag=True
        while True:
            if run[-1][0] not in range(1,matrix.shape[0]-1) or run[-1][1] not in range(1,matrix.shape[1]-1):
                break
            nb = neighbours(run[-1],toRun)
            sort_nb = sorted(nb.items(), key =lambda x:x[1][0] ,reverse=True)
            for key,val in sort_nb:
                grad,acc = val
                flag = True
                if acc and grad < 0:
                    run.append(key)
                    toRun[key] = False
                    rw_matrix[key] = pointer
                    pointer += - grad
                    flag = False
                    break
            if flag:
                break

        render_with_path(matrix, rw_matrix)
        return run

    def forked_grad_walk(error_accept):
        rw_matrix = np.zeros(matrix.shape)
        seeds = matrix == matrix.max()
        toRun = np.logical_and(gl_toRun, np.logical_not(seeds))
        starter = (0,0)
        while True:
            rand_i = rand.randint(1,matrix.shape[0]-1)
            rand_j = rand.randint(1,matrix.shape[1]-1)
            if toRun[rand_i, rand_j]:
                starter = (rand_i, rand_j)
                break

        Visited=[]
        toVisit=[starter]
        pointer=0

        flag=True

        while toVisit!=[]:
            random.shuffle(toVisit)
            if run[-1][0] not in range(1,matrix.shape[0]-1) or run[-1][1] not in range(1,matrix.shape[1]-1):
                break
            nb = neighbours(run[-1],toRun)
            sort_nb = sorted(nb.items(), key =lambda x:x[1][0] ,reverse=True)
            for key,val in sort_nb:
                grad,acc = val
                flag = True
                if acc and grad < -1:
                    run.append(key)
                    toRun[key] = False
                    rw_matrix[key] = pointer
                    pointer += - grad
                    flag = False

            if flag:
                break

        render_with_path(matrix, rw_matrix)
        return run




    def initalize_clusters(to_parse):
        cluster2Dict = {}
        seeds = matrix == matrix.max()
        toRun = np.logical_and(to_parse, np.logical_not(seeds))
        for i,j in np.nonzero(seeds):
            if not any(elt in cluster2Dict.keys() for elt in neighbours(i,j)):
                cluster2Dict[(i,j)] = [(i,j)]
            else:
                if sum(1 for index in (elt in cluster2Dict.keys() for elt in neighbours(i,j)) if index)>1:
                    pass

        return toRun

    def expand_clusters():
        pass

    print grad_walk(10)


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


if __name__ == "__main__":
    fmat = remap(bw,40,10, 0.7, 0.5)
    segment(fmat)
