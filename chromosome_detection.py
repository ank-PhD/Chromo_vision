__author__ = 'ank'

import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
import math
from time import time
import random as rand
from pprint import PrettyPrinter
from copy import copy
import mdp

ImageRoot = "/home/ank/Documents/var"
col = PIL.Image.open(ImageRoot + '/n4.jpeg')
gray=col.convert('L')
bw = np.asarray(gray).copy()

plt.imshow(bw, cmap = 'gray')
plt.show()

X, Y = np.meshgrid(range(bw.shape[0] + 1), range(bw.shape[1] + 1))


# Deprecated
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

# Deprecated
def remap(matrix, contrast_threshold, ising_threshold, ins_lum_thresh, real_ins_lum_thresh):
    grad = np.gradient(matrix, 0.1)
    absgrad = np.absolute(grad[1]) + np.absolute(grad[0])
    absgrad = absgrad * float(2 ** 8 - 1) / float(absgrad.max())
    dergrad = Ising_Round(absgrad, 0.9, 2, ising_threshold)

    hist_of_vals(absgrad)
    render_matrix(absgrad)
    render_matrix(dergrad)

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


def hist_of_vals(matrix):
    array = matrix.flatten()
    plt.hist(array)
    plt.show()

# deprecated
def render_matrix(matrix):
    im = plt.pcolormesh(X, Y, matrix.transpose(), cmap = 'gray')
    plt.colorbar(im)
    plt.show()


def render_clusters(matrix,path_matrix):
    im =  plt.pcolormesh(X, Y, to255(matrix).transpose(), cmap='gray')
    im2 = plt.pcolormesh(X, Y, to255(path_matrix).transpose(), cmap='Paired')
    im2.set_alpha(0.5)
    plt.colorbar(im)
    plt.colorbar(im2)
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

    up, down, right, left = Directional_gradient(matrix)

    def reverse_heritage(dict):
        retdict = {}
        for key,val in dict.iteritems():
            if val not in retdict.keys():
                retdict[val]=[]
            retdict[val].append(key)
        return retdict

    def neighbours(i_j, toRun):
        i=i_j[0]
        j=i_j[1]
        neigh2grad = {}
        neigh2grad[(i+1,j)] = (right[i,j], toRun[i+1,j]) # right
        neigh2grad[(i,j-1)] = (down[i,j], toRun[i,j-1]) # down
        neigh2grad[(i,j+1)] = (up[i,j], toRun[i,j+1]) # up
        neigh2grad[(i-1,j)] = (left[i,j], toRun[i-1,j]) # left

        return neigh2grad

    def get_geometry(reversed_heritage):
        forks = 0
        roots = 0
        for key,vallist in reversed_heritage.iteritems():
            roots += 1
            forks += len(vallist)
        return  roots, forks

    def grad_walk_up(error_accept):
        gl_toRun = matrix > 0.1
        rw_matrix = np.zeros(matrix.shape)
        seeds = matrix == matrix.max()
        toRun = np.logical_and(gl_toRun, np.logical_not(seeds))

        toRun[0,:] = False
        toRun[:,0] = False
        toRun[:,-1] = False
        toRun[-1,:] = False

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
            nb = neighbours(run[-1],toRun)
            sort_nb = sorted(nb.items(), key =lambda x:x[1][0])
            flag = True
            for key,val in sort_nb:
                grad,acc = val
                if acc and grad > error_accept:
                    run.append(key)
                    toRun[key] = False
                    rw_matrix[key] = pointer
                    pointer += grad
                    flag = False
                    break
            if flag:
                break
        print
        render_with_path(matrix, rw_matrix)
        return run

    def forked_grad_walk(error_accept, inertia, starter, toRun, white_cut_off):

        rw_matrix = np.zeros(matrix.shape)
        forkTree = {}
        toRun[starter] = False

        Visited = []
        toVisit = [starter]
        source2Energy = {starter: inertia}

        staging={1:[]}
        stage = 1
        stage_trigger = starter

        while toVisit!=[]:
            current = toVisit.pop(0)
            Visited.append(current)
            nb = neighbours(current,toRun)
            sort_nb = sorted(nb.items(), key =lambda x:x[1][0], reverse=True)
            for key,val in sort_nb:
                grad,acc = val
                if acc and grad < error_accept:
                    if grad < 0.01:
                        source2Energy[key] = source2Energy[current]
                    else:
                        if matrix[current] > white_cut_off:
                            source2Energy[key] = source2Energy[current]
                        else:
                            if source2Energy[current] < 0:
                                continue
                            else:
                                source2Energy[key] = source2Energy[current]-1
                    forkTree[key] = current
                    toVisit.append(key)
                    toRun[key] = False
            rw_matrix[current] = stage
            staging[stage].append(current)
            if current == stage_trigger and toVisit!=[]:
                stage += 1
                staging[stage] = []
                tpl=toVisit[-1]
                stage_trigger = copy(tpl)

        i,j = get_geometry(reverse_heritage(forkTree))
        if i<1:
            i=1
        avg_white = np.sum(matrix[rw_matrix > 0])/i

        # print i, i/stage
        #render_with_path(matrix, rw_matrix)
        # print [(stage, len(staging[stage])) for stage in staging.keys() ]

        return toRun, rw_matrix > 0, i, avg_white

    def initalize_seeds(init_toRun):
        mx = matrix[init_toRun].max()
        pre_seeds = matrix == mx
        seeds = np.logical_and(init_toRun, pre_seeds)
        return seeds, init_toRun.copy(), mx

    def cancel_limits(init_toRun):
        toRun=init_toRun.copy()
        toRun[0,:] = False
        toRun[:,0] = False
        toRun[:,-1] = False
        toRun[-1,:] = False
        return toRun

    def full_loop(max_grad, inertia, black_cut_off, white_cut_off):
        gl_toRun = matrix > black_cut_off
        initseeds, toRun, mx = initalize_seeds(cancel_limits(gl_toRun))
        pointer=0
        term=np.zeros(matrix.shape)
        while True:
            if np.nonzero(toRun)<10:
                 break
            if mx < 30:
                break
            initseeds, toRun, mx = initalize_seeds(toRun)
            nz = np.nonzero(initseeds)
            seed = (nz[0][0],nz[1][0])
            toRun, mask, pixs, avg_white = forked_grad_walk(max_grad, inertia, seed, toRun, white_cut_off)
            if pixs < 1000 and pixs > 10:
                pointer+=1
                term[mask]= pointer
                print pixs, avg_white
        term[term==0] = -10
        render_clusters(matrix,term)
        print pointer

    print full_loop(255, 0, 25, 230)

# TODO: deparametrize
# TODO: get a better colormap
# TODO: troubleshoot individual misinterpretation
# TODO: simultaneous random walk
# TODO: correct the gradient problem

def border_detect(matrix, contrast_threshold, ising_threshold, ins_lum_thresh):
    grad = np.gradient(matrix)
    absgrad = np.absolute(grad[1]) + np.absolute(grad[0])
    absgrad = absgrad * (2 ** 8 - 1) / absgrad.max()
    dergrad = Ising_Round(absgrad, 0.9, 2, ising_threshold)
    im = plt.pcolormesh(X, Y, dergrad.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()

    non_sign_grad = dergrad < contrast_threshold
    sign_grad = dergrad > contrast_threshold
    gradient_average = np.average( matrix[sign_grad] )
    insufficent_lum = matrix < ins_lum_thresh * gradient_average
    print np.logical_and(non_sign_grad, insufficent_lum)

    filtered_matrix = matrix.copy()
    filtered_matrix = filtered_matrix * (2 ** 8 - 1) / filtered_matrix.max()

    filtered_matrix[sign_grad] = np.round(filtered_matrix[sign_grad] / 255.0) * 255

    im = plt.pcolormesh(X, Y, filtered_matrix.transpose(), cmap='gray')
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    fmat = remap(bw, 0, 0, 0.7, 0.5)
    # segment(fmat)
