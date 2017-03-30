
import dp4gp
import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def bin_data(Xtest,X,step,ys,aggregation):
    """
    Bin data X into equally sized bins defined by Xtest and step.
    Xtest is the coordinates of the corner of each bin.
    step is a vector of step sizes.
    ys are the outputs (to be summed and averaged)
    
    Returns:
    bincounts
    bintotals
    binaverages = bintotals/bincounts
    """
    bintotals = np.zeros(Xtest.shape[0])
    bincounts = np.zeros(Xtest.shape[0])
    if aggregation=='median':
        binagg = [list([]) for _ in xrange(Xtest.shape[0])]

    for i,tile in enumerate(Xtest): #loop through the tiles
        for x,y in zip(X,ys): #loop through the data
            intile = True
            for tiled,xd,s in zip(tile,x,step): #loop through the dimensions of the current tile, data and step
                if (xd<tiled) or (xd>tiled+s):
                    intile = False
                    break
            if intile:
                bintotals[i]+=y
                bincounts[i]+=1
                if aggregation=='median':
                    binagg[i].append(y)
    if aggregation=='mean':             
        binaverages = bintotals/bincounts
    if aggregation=='median':
        binaverages = np.zeros(Xtest.shape[0])
        for i, b in enumerate(binagg):
            binaverages[i] = np.median(b)
    return bincounts, bintotals, binaverages

class DPGP_histogram(dp4gp.DPGP):
    """Using the histogram method"""
    
    def __init__(self,sens,epsilon,delta):      
        super(DPGP_histogram, self).__init__(None,sens,epsilon,delta)

    def prepare_model(self,Xtest,X,step,ys,variances=1.0,lengthscale=1,aggregation='mean'):
        """
        Prepare the model, ready for making predictions"""
        bincounts, bintotals, binaverages = bin_data(Xtest,X,step,ys,aggregation)
        if aggregation=='median':
            raise NotImplementedError
        if aggregation=='mean':            
            sens_per_bin = self.sens/bincounts
        c = np.sqrt(2*np.log(1.25/self.delta)) #1.25 or 2 over delta?
        bin_sigma = c*sens_per_bin/self.epsilon #noise standard deviation to add to each bin
        #add DP noise to the binaverages
        dp_binaverages=binaverages+np.random.randn(binaverages.shape[0])*bin_sigma

        #we need to build the input for the integral kernel
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest+step
        newXtest[:,1::2] = Xtest

        #we don't want outputs that have no training data in.
        empty = np.isnan(dp_binaverages)
        dp_binaverages[empty] = 0 #we'll make those averages zero

        self.Xtest = newXtest
        self.dp_binaverages = dp_binaverages
        return bincounts, bintotals, binaverages, sens_per_bin, bin_sigma, dp_binaverages
    
    def optimize(self):
        self.model.optimize()
     
    def draw_prediction_samples(self,Xtest,N=1):
        assert N==1, "DPGP_histogram only returns one DP prediction sample (you will need to rerun prepare_model to get an additional sample)"
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest
        newXtest[:,1::2] = 0
        preds = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            v = np.ones(self.Xtest.shape[0],dtype=bool)
            for idx in range(0,Xtest.shape[1]):
                v = v & ((Xtest[i,idx] < self.Xtest[:,idx*2]) & (Xtest[i,idx] > self.Xtest[:,idx*2+1]))        
            preds[i] = self.dp_binaverages[v]
        return preds, None
