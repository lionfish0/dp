import dp4gp
import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class DPGP_histogram(DPGP):
    """Using the histogram method"""
    
    def __init__(self,model,sens,epsilon,delta):      
        super(DPGP_histogram, self).__init__(model,sens,epsilon,delta)

    def draw_prediction_samples(self,Xtest,N=1):
        GPymean, covar = self.model.predict(Xtest)
        mean, noise, cov = self.draw_noise_samples(Xtest,N,Nattempts,Nits)
        #TODO: In the long run, remove DP4GP's prediction code and just use GPy's
        #print GPymean-mean
        assert np.max(GPymean-mean)<1e-3, "DP4GP code's posterior mean prediction differs from GPy's"
        return mean + noise.T, mean, cov
        
    def draw_noise_samples(self,Xtest,N=1,Nattempts=7,Nits=1000):
        """
        Provide N samples of the DP noise
        """
        sigmasqr = self.model.Gaussian_noise.variance[0]
        K_NN = self.model.kern.K(self.model.X)
        K_NNinv = np.linalg.inv(K_NN+sigmasqr*np.eye(K_NN.shape[0]))
        K_Nstar = self.model.kern.K(Xtest,self.model.X)
        C = np.dot(K_Nstar,K_NNinv)

        print C.shape
        cs = []
        for i in range(C.shape[1]):
            cs.append(C[:,i][:,None])
        
        ls = self.findLambdas_repeat(cs,Nattempts,Nits)
        M = self.calcM(ls,cs)
        
        c = np.sqrt(2*np.log(2/self.delta))
        Delta = self.calcDelta(ls,cs)
        #in Hall13 the constant below is multiplied by the samples,
        #here we scale the covariance by the square of this constant.
        sampcov = ((self.sens*c*Delta/self.epsilon)**2)*M
        samps = np.random.multivariate_normal(np.zeros(len(sampcov)),sampcov,N)
        
        ###This code is only necessary for finding the mean
        mu = np.dot(C,self.model.Y)
        ###
        return mu, samps, sampcov
    
    def plot(self,fixed_inputs=[],legend=False,plot_data=False, steps=10, N=10, Nattempts=1, Nits=500):
        """
        Plot the DP predictions, etc.
        
        In 2d it shows one DP sample, the size of the circles represent the prediction values
        the alpha how much DP noise has been added (1->no noise, 0->20% of max-min prediction
        
        fixed_inputs = list of pairs
        legend = whether to plot the legend
        plot_data = whether to plot data
        steps = resolution of plot
        N = number of DP samples to plot (in 1d)
        Nattempts = number of times a DP solution will be looked for (can help avoid local minima)
        Nits = number of iterations when finding DP solution
        (these last two parameters are passed to the draw_prediction_samples method).
        """
        Xtest, free_inputs = compute_Xtest(self.model.X, fixed_inputs, extent_lower={}, steps=steps)
        preds, mu, cov = self.draw_prediction_samples(Xtest,N,Nattempts=1,Nits=Nits)
        self.model.plot(fixed_inputs=fixed_inputs,legend=legend,plot_data=False)
        DPnoise = np.sqrt(np.diag(cov))
        indx = 0
        if len(free_inputs)==2:
            minpred = np.min(mu)
            maxpred = np.max(mu)
            scaledpreds = 1+1000*(preds[:,indx]-minpred) / (maxpred-minpred)
            scalednoise = 1-5*DPnoise/(maxpred-minpred) #proportion of data
            #any shade implies the noise is less than 20% of the total change in the signal
            scalednoise[scalednoise<0] = 0
            rgba = np.zeros([len(scalednoise),4])
            rgba[:,0] = 1.0
            rgba[:,3] = scalednoise
            plt.scatter(Xtest[:,free_inputs[0]],Xtest[:,free_inputs[1]],scaledpreds,color=rgba)
            plt.scatter(Xtest[:,free_inputs[0]],Xtest[:,free_inputs[1]],scaledpreds,facecolors='none')
            if plot_data: #do this bit ourselves
                plt.plot(self.model.X[:,free_inputs[0]],self.model.X[:,free_inputs[1]],'.k',alpha=0.2)


        if len(free_inputs)==1:
            plt.plot(Xtest[:,free_inputs[0]],preds,alpha=0.2,color='black')
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]-DPnoise,'--k',lw=2)
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]+DPnoise,'--k',lw=2)

