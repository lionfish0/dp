
import dp4gp
import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def bin_data(Xtest,X,step,ys):
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
    binaverages = bintotals/bincounts
    return bincounts, bintotals, binaverages

class DPGP_integral_histogram(dp4gp.DPGP):
    """Using the histogram method"""
    
    def __init__(self,sens,epsilon,delta):      
        super(DPGP_integral_histogram, self).__init__(None,sens,epsilon,delta)

    def prepare_model(self,Xtest,X,step,ys,variances=1.0,lengthscale=1):
        """
        Prepare the model, ready for making predictions"""
        bincounts, bintotals, binaverages = bin_data(Xtest,X,step,ys)
        sens_per_bin = self.sens/bincounts
        #Gaussian Mechanism
        #c = np.sqrt(2*np.log(1.25/self.delta)) #1.25 or 2 over delta?
        #bin_sigma = c*sens_per_bin/self.epsilon #noise standard deviation to add to each bin
        ##add DP noise to the binaverages
        #dp_binaverages=binaverages+np.random.randn(binaverages.shape[0])*bin_sigma
        s = np.array(sens_per_bin / self.epsilon)
        dp_binaverages=binaverages+np.random.laplace(scale=s)#,size=binaverages.shape[0])
        

        #we need to build the input for the integral kernel
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest+step
        newXtest[:,1::2] = Xtest

        #we don't want outputs that have no training data in.
        keep = ~np.isnan(dp_binaverages)
        finalXtest = newXtest[keep,:]
        final_dp_binaverages = dp_binaverages[keep]
        s = s[keep]

        
        #the integral kernel takes as y the integral... 
        #eg. if there's one dimension we're integrating over, km
        #then we need to give y in pound.km
        self.meanoffset = np.mean(final_dp_binaverages)
        final_dp_binaverages-= self.meanoffset
        finalintegralbinaverages = final_dp_binaverages * np.prod(step) 
        final_sigma = 2.0*(s**2) #I've no idea but optimizer will find this later... #bin_sigma[keep]
        finalintegralsigma = final_sigma * np.prod(step)
        
        #generate the integral model
        kernel = GPy.kern.Multidimensional_Integral_Limits(input_dim=newXtest.shape[1], variances=variances, lengthscale=lengthscale)
        #we add a kernel to describe the DP noise added
        kernel = kernel + GPy.kern.WhiteHeteroscedastic(input_dim=newXtest.shape[1], num_data=len(finalintegralsigma), variance=finalintegralsigma**2)
        self.model = GPy.models.GPRegression(finalXtest,finalintegralbinaverages[:,None],kernel)
    
    def optimize(self):
        self.model.optimize()
     
    def draw_prediction_samples(self,Xtest,N=1):
        assert N==1, "DPGP_histogram only returns one DP prediction sample (you will need to rerun prepare_model to get an additional sample)"
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest
        newXtest[:,1::2] = 0
        mean, cov = self.model.predict(newXtest)
        return mean+self.meanoffset, cov
