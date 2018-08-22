# -*- coding: utf-8 -*-
"""

~
albin.a.correya@gmail.com
2018

"""
import numpy as np

def slice_2d_sim_matrix(input_array):
    """Slice a 2d matrix into 4 slices"""
    x_idx = input_array.shape[0] / 2
    y_idx = input_array.shape[1] / 2
    slices = []
    slice_one = input_array[:x_idx, :y_idx]
    slice_two = input_array[:x_idx, y_idx:]
    slice_three = input_array[x_idx:, y_idx:]
    slice_four = input_array[x_idx:, :y_idx]
    slices.extend([slice_one, slice_two, slice_three, slice_four])
    return slices, (x_idx, y_idx)

def plot_2darray_slices(array_slices):
    """Plot slices of an 2d array in order"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,8))
    for idx, slice in enumerate(array_slices):
        plt.imshow(slice, origin='lower')
        plt.xlabel("%sth slice of matrix" % str(idx+1))
        plt.show()
    return

def plot_sim_matrix(sim_matrix):
    """Plot similarity matrix of two songs"""
    import matplotlib.pyplot as plt
    plt.imshow(sim_matrix, origin='lower')
    plt.title("Similarity matrix")
    plt.xlabel("Reference_song (ms)")
    plt.ylabel("Query song (ms)")
    plt.show()
    return

def plot_qmax_matrix(score_matric, distance):
    """plot the qmax,dmax score matrix"""
    import matplotlib.pyplot as plt
    plt.imshow(score_matrix, origin='lower')
    plt.title("Qmax distance: %s" % distance)
    plt.xlabel("Reference_song (ms)")
    plt.ylabel("Query song (ms)")
    plt.show()
    return
