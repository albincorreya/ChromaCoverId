# -*- coding: utf-8 -*-

"""
Some set of functions for computing audio similarity measures for the task of cover song detection

CrossRecurrentPlots
QmaxMeasure
DmaxMeasure

------
@2017
Albin Andrew Correy
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances




def optimalTranspositionIndex(chromaA, chromaB):
    """
    Computes optimal transposition index (OTI) for the chromaB to be transposed in the same key as of chromaA
    (Note : Chroma vectors are resized to the shape of smallest array shape if the shapes of chroma vectors are not same.)
    Input :
            chromaA : a
            chromaB :
    Output : Integer value specifying optimal transposition index for transposing chromaB to chromaA to be in same key
    """
    #computes global hpcp for both chroma vectors
    global_hpcpA = np.sum(chromaA, axis=1) / np.max(np.sum(chromaA, axis=1))
    global_hpcpB = np.sum(chromaB, axis=1) / np.max(np.sum(chromaB, axis=1))

    idx = list()

    if len(global_hpcpA)>len(global_hpcpB):
        resized_h1 = np.resize(global_hpcpA, len(global_hpcpB))
        for index in range(12):
            idx.append(np.dot(resized_h1, np.roll(global_hpcpB, index)))
        oti = idx.index(max(idx))

    elif len(global_hpcpB)>len(global_hpcpA):
        resized_h2 = np.resize(global_hpcpB, len(global_hpcpA))
        for index in range(12):
            idx.append(np.dot(global_hpcpA, np.roll(resized_h2, index)))
        oti = idx.index(max(idx))

    else:
        for index in range(12):
            idx.append(np.dot(global_hpcpA, np.roll(global_hpcpB, index)))
        oti = idx.index(max(idx))

    return oti


def tranposebyOTI(chromaB, oti=0):
    """
    Transpose the chromaB vector to a common key by a value of optimal transposition index
    Input :
            chromaB :
    Output : chromaB vector transposed to a factor of specified OTI
    """
    print "\nOptimal Transposition Index (OTI) :", oti
    return np.roll(chromaB, oti)



class RecurrentPlots():
    """
    RecurrentPlots
    Methods available :
                        * Time series with delay embedding
                        * Cross recurrent plots of two audio feature vectors
                        * Qmax similarity measure from cross recurrent plots
                        * Dmax similarity measure from cross recurrent plots
    Example use :
                    RP = RecurrencePlots()

                    x_timeseries = RP.to_timeSeries(audio1_feature1)
                    # get cross recurrence plot of audio feature arrays of two audio tracks
                    crp = RP.CrossRecurrentPlot(audio1_feature1, audio2_feature2)
                    qmax = RP.QmaxMeasure(crp)
                    dmax = RP.DmaxMeasure(crp)
    """
    def __init__(self):
        return

    def to_timeSeries(self, featureArray, tau=1, m=9):
        """
        Construct a time series with delay embedding tau and embedding dimension m from the input audio feature vector
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


    def crossRecurrentPlot(self, featureA, featureB, tau=1, m=9, kappa=0.095, transpose=True, swapaxis=False):
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
            oti = optimalTranspositionIndex(featureA, featureB)
            featureB = tranposebyOTI(featureB, oti) #transpose featureB to the key of featureA by a oti value

        if swapaxis:
            featureA = np.swapaxes(featureA,1,0)
            featureB = np.swapaxes(featureB,1,0)

        timespaceA = self.to_timeSeries(featureA, tau=tau, m=m)
        timespaceB = self.to_timeSeries(featureB, tau=tau, m=m)

        pdistances = euclidean_distances(timespaceA, timespaceB)
        transposed_pdistances = pdistances.T

        eph_x = np.percentile(pdistances, kappa*100, axis=1)
        eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)

        x = eph_x[:,None] - pdistances
        y = eph_y[:,None] - transposed_pdistances

        #apply heaviside function to the array (Binarize the array)
        x = np.piecewise(x, [x<0, x>=0], [0,1])
        y = np.piecewise(y, [y<0, y>=0], [0,1])

        crossRecurrentPlot = x*y.T

        return crossRecurrentPlot


    def qmaxMeasure(self, crp, gammaO=0.5, gammaE=0.5):
        """
        Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]
        [NOTE] The function is implemented in the matlab way for clearing the idea
        
        Inputs :
                crp : 2-d binary matrix of cross recurrent plot
        Params :
                gammaO : penalty for a disurption onset
                gammaE : penalty for a disurption extension
        Output : qmax similarity measure from a crp matrix
        
        [1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover
            song identification. New Journal of Physics, 11.    
       
        """

        print "\ncrp with shape :", crp.shape

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
                                                (cum_matrix[i-1][j-1] - self.gammaState(crp[i-1][j-1], gammaO, gammaE)),
                                                (cum_matrix[i-2][j-1] - self.gammaState(crp[i-2][j-1], gammaO, gammaE)),
                                                (cum_matrix[i-1][j-2] - self.gammaState(crp[i-1][j-2], gammaO, gammaE))]
                                              )

        print "Cumulative Matrix computed with shape :", cum_matrix.shape

        if np.max(cum_matrix)==0:
            print "*****Cum_matrix max is Zero"

        Qmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))

        return Qmax


    def dmaxMeasure(self, crp, gammaO=0.5, gammaE=0.5):
        """
        Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]
        
        The function is implemented in the matlab way for clearing the idea
        Inputs :
                crp : 2-d binary matrix of cross recurrent plot
        Params :
                gammaO : penalty for a disurption onset
                gammaE : penalty for a disurption extension
        Output : dmax similarity measure from a crp matrix
        [TODO : to optimize and re-implement it by pythonic numpy way]
        
        
        [1]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications, pp. 1–24. 
        """

        print "\n====Dmax distance measure for cover song identification===="

        print "\ncrp with shape :", crp.shape

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
                                                (cum_matrix[i-1][j-1] - self.gammaState(crp[i-1][j-1], gammaO, gammaE)),
                                                ((cum_matrix[i-2][j-1] + crp[i-1][j]) - self.gammaState(crp[i-2][j-1], gammaO, gammaE)),
                                                ((cum_matrix[i-1][j-2] + crp[i][j-1]) - self.gammaState(crp[i-1][j-2], gammaO, gammaE)),
                                                ((cum_matrix[i-3][j-1] + crp[i-2][j] + crp[i-1][j]) -  self.gammaState(crp[i-3][j-1], gammaO, gammaE)),
                                                ((cum_matrix[i-1][j-3] + crp[i][j-2] + crp[i][j-1]) -  self.gammaState(crp[i-1][j-3], gammaO, gammaE))])


        print "Cumulative Matrix computed with shape :", cum_matrix.shape

        Dmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))

        return Dmax

    def gammaState(self, value, gammaO=0.5, gammaE=0.5):
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
        if value == 0.:
            return gammaE
        else:
            raise ValueError("Non-binary numbers found in the cross recurrent matrix...")
        return


    def plotCRP(self, crp, cmap='hot'):
        plt.imshow(crp, cmap=cmap)
        plt.show()
        return
