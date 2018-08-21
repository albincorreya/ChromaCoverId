# -*- coding: utf-8 -*-

"""
Some set of functions for computing audio similarity measures for the task of cover song detection

* cross_recurrent_plots
* qmax_measure [1]
* dmax_measure [2]
* chroma_binary_by_oti [3]

[1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.
     New Journal of Physics.

[2]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia
     Tools and Applications.

[3]. Serra, Joan, et al. "Chroma binary similarity and local alignment applied to cover song identification."
         IEEE Transactions on Audio, Speech, and Language Processing 16.6 (2008): 1138-1151.



Thanks to Romain Hennequin

------
Albin Andrew Correya
@2017
"""

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np



def global_hpcp(chroma):
    """Computes global hpcp of a input chroma vector"""
    return np.sum(chroma, axis=1) / np.max(np.sum(chroma, axis=1))


def optimal_transposition_index(chromaA, chromaB, bin_size=12):
    """
    Computes optimal transposition index (OTI) for the chromaB to be transposed in the same key as of chromaA
    (Note : Chroma vectors are resized to the shape of smallest array shape if the shapes of chroma vectors are not same.)
    Input :
            chromaA :
            chromaB :
        Params:
                bin_size: Number of bins in the chroma vector
    Output : Integer value specifying optimal transposition index for transposing chromaB to chromaA to be in same key
    """
    global_hpcpA = global_hpcp(chromaA)
    global_hpcpB = global_hpcp(chromaB)

    idx = list()

    if len(global_hpcpA)>len(global_hpcpB):
        resized_h1 = np.resize(global_hpcpA, len(global_hpcpB))
        for index in range(bin_size):
            idx.append(np.dot(resized_h1, np.roll(global_hpcpB, index)))
        #oti = idx.index(max(idx))
        oti = np.argmax(idx)

    elif len(global_hpcpB)>len(global_hpcpA):
        resized_h2 = np.resize(global_hpcpB, len(global_hpcpA))
        for index in range(bin_size):
            idx.append(np.dot(global_hpcpA, np.roll(resized_h2, index)))
        #oti = idx.index(max(idx))
        oti = np.argmax(idx)

    else:
        for index in range(bin_size):
            idx.append(np.dot(global_hpcpA, np.roll(global_hpcpB, index)))
        #oti = idx.index(max(idx))
        oti = np.argmax(idx)

    return oti


def transpose_by_oti(chromaB, oti=0):
    """
    Transpose the chromaB vector to a common key by a value of optimal transposition index
    Input :
            chromaB :
    Output : chromaB vector transposed to a factor of specified OTI
    """
    print "\nOptimal Transposition Index (OTI) :", oti
    return np.roll(chromaB, oti)


def to_timeseries(featureArray, tau=1, m=9):
    """
    Construct a time series with delay embedding 'tau' and embedding dimension 'm' from the input audio feature vector
    Input :
            featureArray :
    Params :
            tau (default : 1):
            m (default : 9):
    Output : Time series representation of the input audio feature vector
    """

    timeseries = list()
    for startTime in range(0, featureArray.shape[0] - m*tau, tau):
        stack = list()
        for idx in range(startTime, startTime + m*tau, tau):
            stack.append(featureArray[idx])
        timeseries.append(np.ndarray.flatten(np.array(stack)))
    return np.array(timeseries)


def gamma_state(value, gammaO=0.5, gammaE=0.5):
    """
    This prevents negative entry in Qi,j while calculating Qmax
    Input :
            value :
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disruption extension
    Output : gammaO if value is 1 or gammaE if value is 0
    """
    if value == 1.:
        return gammaO
    elif value == 0.:
        return gammaE
    else:
        raise ValueError("Non-binary numbers found in the cross recurrent matrix...")


def cross_recurrent_plot(featureA, featureB, tau=1, m=9, kappa=0.095, transpose=True, swapaxis=False):
    '''
    Constructs the Cross Recourrent Plot of two audio feature vector as mentioned in [1]
    Inputs :
            featureA :
            featureB :
    Params :
            kappa (default=0.095)       :
            tau (default=1)             : delay embedding
            m (default=9)               : embedding dimension for the time series
            swapaxis (default=False)    : swapaxis of the feature array if it not in the shape (x,12) where x is the time axis
            transpose (default=True)    :
    Output : Binary similarity matrix where 1 constitutes a similarity between
            two feature vectors at ith and jth position respectively and 0 denotes non-similarity
    '''


    if transpose:
        oti = optimal_transposition_index(featureA, featureB)
        featureB = transpose_by_oti(featureB, oti) #transpose featureB to the key of featureA by a oti value

    if swapaxis:
        featureA = np.swapaxes(featureA,1,0)
        featureB = np.swapaxes(featureB,1,0)

    timespaceA = to_timeseries(featureA, tau=tau, m=m)
    timespaceB = to_timeseries(featureB, tau=tau, m=m)
    pdistances = euclidean_distances(timespaceA, timespaceB)
    transposed_pdistances = pdistances.T
    eph_x = np.percentile(pdistances, kappa*100, axis=1)
    eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)
    x = eph_x[:,None] - pdistances
    y = eph_y[:,None] - transposed_pdistances
    #apply heaviside function to the array (Binarize the array)
    x = np.piecewise(x, [x<0, x>=0], [0,1])
    y = np.piecewise(y, [y<0, y>=0], [0,1])
    cross_recurrent_plot = x*y.T
    return cross_recurrent_plot


def qmax_measure(crp, gammaO=0.5, gammaE=0.5):
    """
    Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]
    Inputs :
            crp : 2-d binary matrix of cross recurrent plot
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disurption extension
    Output : qmax similarity measure from a crp matrix

    [1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover
        song identification. New Journal of Physics, 11.

    """
    print "\nCRP with shape :", crp.shape
    Nx = crp.shape[0]
    Ny = crp.shape[1]
    cum_matrix = np.zeros(crp.shape)
    #for the moment doing it in the matlab way
    for i in range(2,Nx):
        for j in range(2,Ny):
            if int(crp[i,j]) == 1:
                cum_matrix[i][j] = np.max([cum_matrix[i-1][j-1], cum_matrix[i-2][j-1], cum_matrix[i-1][j-2]]) + 1
            else:
                cum_matrix[i][j] = np.max([0,
                                            (cum_matrix[i-1][j-1] - gamma_state(crp[i-1][j-1], gammaO, gammaE)),
                                            (cum_matrix[i-2][j-1] - gamma_state(crp[i-2][j-1], gammaO, gammaE)),
                                            (cum_matrix[i-1][j-2] - gamma_state(crp[i-1][j-2], gammaO, gammaE))]
                                            )
    #print "Cumulative Matrix computed with shape :", cum_matrix.shape
    if np.max(cum_matrix)==0:
        print "*Cum_matrix max is Zero"
    qmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))
    return qmax, cum_matrix


def dmax_measure(crp, gammaO=0.5, gammaE=0.5):
    """
    Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]

    Inputs :
            crp : 2-d binary matrix of cross recurrent plot
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disurption extension
    Output : dmax similarity measure from a crp matrix
    [TODO : to optimize and re-implement it by pythonic numpy way]

    [1]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification.
         Multimedia Tools and Applications.
    """

    print "\n====Dmax distance measure for cover song identification===="
    print "\nCrp with shape :", crp.shape
    Nx = crp.shape[0]
    Ny = crp.shape[1]
    cum_matrix = np.zeros(crp.shape)
    #for the moment doing it in the matlab way
    for i in range(3,Nx):
        for j in range(3,Ny):
            if int(crp[i,j]) == 1:
                cum_matrix[i][j] = np.max([cum_matrix[i-1][j-1],
                                            cum_matrix[i-2][j-1] + crp[i-1][j],
                                            cum_matrix[i-1][j-2] + crp[i][j-1],
                                            cum_matrix[i-3][j-1] + crp[i-2][j] + crp[i-1][j],
                                            cum_matrix[i-1][j-3] + crp[i][j-2], crp[i][j-1]]) + 1
            else:
                cum_matrix[i][j] = np.max([0,
                                            (cum_matrix[i-1][j-1] - gamma_state(crp[i-1][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-2][j-1] + crp[i-1][j]) - gamma_state(crp[i-2][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-1][j-2] + crp[i][j-1]) - gamma_state(crp[i-1][j-2], gammaO, gammaE)),
                                            ((cum_matrix[i-3][j-1] + crp[i-2][j] + crp[i-1][j]) -  gamma_state(crp[i-3][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-1][j-3] + crp[i][j-2] + crp[i][j-1]) -  gamma_state(crp[i-1][j-3], gammaO, gammaE))])
    #print "Cumulative Matrix computed with shape :", cum_matrix.shape
    dmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))
    return dmax, cum_matrix


def oti(input_x, input_y, n_bins=12):
    idx = list()
    for index in range(n_bins):
        idx.append(np.dot(input_x, np.roll(input_y, index)))
    return np.argmax(idx)


def chroma_to_binary_by_oti(chroma_a, chroma_b, match=1, mismatch=-0.9):
    """
    [TODO]
    [NOTE: not yet finished the implementation]

    [1]. Serra, Joan, et al. "Chroma binary similarity and local alignment applied to cover song identification."
         IEEE Transactions on Audio, Speech, and Language Processing 16.6 (2008).
    """
    otidx = optimal_transposition_index(chroma_a, chroma_b)
    chroma_b = transpose_by_oti(chroma_b, otidx)
    Nx = chroma_a.shape[0]
    Ny = chroma_b.shape[0]
    binary_matrix = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            o_indx = oti(chroma_a[i], chroma_b[j])
            if o_indx in [0,1]:
                sim_matrix[i][j] = match
            else:
                binary_matrix[i][j] = mismatch
    return binary_matrix


def dtw(input_x, input_y, dist=euclidean):
    """TODO"""
    distance, path = fastdtw(input_x, input_y, dist=dist)
    return distance


def smith_waterman_score(similarity_matrix):
    """TODO"""
    return
