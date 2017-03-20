# Methods for combining differential privacy with Gaussian Processes

import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def compute_Xtest(X,fixed_inputs=[],extent_lower={},extent_upper={},percent_extra=0.1,steps=10):
    """
    Produce a matrix of test points, does roughly what meshgrid does, but for
    arbitrary numbers of dimensions, and handles fixed_inputs, etc etc.
        - Pass X (training data)
        - can also specify extent
            - extent_lower/upper are dictionaries, e.g. {0:5.2,1:4.3}, with the
              index of the dimension and the start value. Note that if you
              specify fixed_inputs then the extents will be overridden.
        - if extend is not specified, the method will use the data's extent
              it adds an additional "percent_extra" to each dimension.
        - steps = number of steps, either an integer or a list of integers (one
              for each dimension)
              
    Example:
            X is 4d, we fix dimensions 0 and 1. We make dimension 2 start at zero
            and have 3 steps in that dimension and 10 in the last dimension, giving
            us 30 points in our output.
        Xtest = compute_Xtest(X, [(1,180e3),(0,528e3)], extent_lower={2:0},steps=[1,1,3,10])   
    """
    rangelist = []
    lower = np.zeros(X.shape[1])
    upper = lower.copy()
    step = lower.copy()
    
    free_inputs = []
    
    if type(steps)==int:
        steps = np.ones(X.shape[1])*steps
        
    for i,(start,finish) in enumerate(np.array([np.min(X,0),np.max(X,0)]).T):
        extra = (finish-start)*percent_extra
        if i not in extent_lower:
            lower[i] = start-extra
        else:
            lower[i] = extent_lower[i]
        if i not in extent_upper:
            upper[i] = finish+extra
        else:
            upper[i] = extent_upper[i]

        step[i] = (upper[i]-lower[i])/steps[i]
        rangelist.append('lower[%d]:upper[%d]:step[%d]'%(i,i,i))
        if i not in [f[0] for f in fixed_inputs]:
            free_inputs.append(i)
        else:
            lower[i] = [f[1] for f in fixed_inputs if f[0]==i][0]
            upper[i] = lower[i]+0.1
            step[i] = 1 #just ensure one item is added

    evalstr = 'np.mgrid[%s]'%(','.join(rangelist))
    res = eval(evalstr)
    
    #handles special case, when ndim=1, mgrid doesn't have an outer array
    if np.ndim(res)==1: 
        res = res[None,:]
    
    res_flat = []
    for i in range(len(res)):
        res_flat.append(res[i].flatten())
    Xtest = np.zeros([len(res_flat[0]),X.shape[1]])
   
    for i, r in enumerate(res_flat):
        Xtest[:,i] = r
        
    return Xtest, free_inputs, step

class DPGP(object):
    """(epsilon,delta)-Differentially Private Gaussian Process predictions"""
    
    def __init__(self,model,sens,epsilon,delta):
        """
        Parameters:
            model = Pass a GPy model object
            sens = data sensitivity (how much can one output value vary due to one person
            epsilon = epsilon (DP parameter)
            delta = delta (DP parameter) [probability of providing DP]
            
        """
        self.model = model
        self.sens = sens
        self.epsilon = epsilon
        self.delta = delta
    
    def draw_prediction_samples(self,Xtest,N=1,Nattempts=7,Nits=1000):
        GPymean, covar = self.model.predict(Xtest)
        mean, noise, cov = self.draw_noise_samples(Xtest,N,Nattempts,Nits)
        #TODO: In the long run, remove DP4GP's prediction code and just use GPy's
        #print GPymean-mean
        assert np.max(GPymean-mean)<1e-2, "DP4GP code's posterior mean prediction differs from GPy's"
        return mean + noise.T, mean, cov
    
    def plot(self):
        raise NotImplementedError #need to implemet in a subclass
        
class DPGP_prior(DPGP):
    """
    DP provided by adding a sample from the prior
    """
    
#    def __init__(self,model,sens,epsilon,delta):      
#        super(DPGP_prior, self).__init__(model,sens,epsilon,delta)
        
    def calc_msense(self,A):
        """
        originally returned the infinity norm*, but we've developed an improved value from
        this norm which only cares about values of the same sign (it is assumed that
        those of the opposite sign will work to reduce the sensitivity). We'll call
        this the matrix_sensitivity or msense
        * np.max(np.sum(np.abs(A),1))
        """
        v1 = np.max(np.abs(np.sum(A.copy().clip(min=0),1)))
        v2 = np.max(np.abs(np.sum((-A.copy()).clip(min=0),1)))
        return np.max([v1,v2])

    def draw_cov_noise_samples(self,test_cov,msense,N=1):        
        """
        Produce differentially private noise for this covariance matrix
        """
        G = np.random.multivariate_normal(np.zeros(len(test_cov)),test_cov,N)
        noise = G*self.sens*np.sqrt(2*np.log(2/self.delta))/self.epsilon
        noise = noise * msense
        print msense*self.sens*np.sqrt(2*np.log(2/self.delta))/self.epsilon
        return np.array(noise), test_cov*(msense*self.sens*np.sqrt(2*np.log(2/self.delta))/self.epsilon)**2

    def draw_noise_samples(self,Xtest,N=1,Nattempts=7,Nits=1000):
        raise NotImplementedError #need to implemet in a subclass
        
    #def draw_prediction_samples(self,Xtest,N=1):
    #    GPymean, covar = self.model.predict(Xtest)
    #    mean, noise, _ = self.draw_noise_samples(Xtest,N)
    #    #TODO: In the long run, remove DP4GP's prediction code and just use GPy's
    #    assert np.max(GPymean-mean)<1e-3, "DP4GP code's posterior mean prediction differs from GPy's"
    #    return mean + noise.T
    
#    def plot(self):
#        p = self.model.plot(legend=False)
#        xlim = p.axes.get_xlim()
#        Xtest = np.arange(xlim[0],xlim[1],(xlim[1]-xlim[0])/100.0)[:,None]
#        noisy_mu, _, _ = self.draw_prediction_samples(Xtest,20)
#        plt.plot(Xtest,noisy_mu,'-k',alpha=0.3);

    def plot(self,fixed_inputs=[],legend=False,plot_data=False, steps=None, N=10, Nattempts=1, Nits=500, extent_lower={}, extent_upper={},ys_std=1.0,ys_mean=0.0,plotGPvar=True,confidencescale=1.0):
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
        confidencescale = how wide the CI should be (default = 1 std.dev)
        """
        if steps is None:
            dims = self.model.X.shape[1]-len(fixed_inputs) #get number of dims
            steps = int(100**(1/dims)) #1d=>100 steps, 2d=>10 steps
        Xtest, free_inputs, _ = compute_Xtest(self.model.X, fixed_inputs, extent_lower=extent_lower, extent_upper=extent_upper, steps=steps)

        preds, mu, cov = self.draw_prediction_samples(Xtest,N,Nattempts=1,Nits=Nits)
        preds *= ys_std
        preds += ys_mean        
        mu *= ys_std
        mu += ys_mean
        cov *= (ys_std**2)

        assert len(free_inputs)<=2, "You can't have more than two free inputs in a plot"
        if len(free_inputs)==1:
            pltlim = [np.min(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[0]])]
        if len(free_inputs)==2:
            pltlim = [[np.min(Xtest[:,free_inputs[0]]),np.min(Xtest[:,free_inputs[1]])],[np.max(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[1]])]] 

        
        DPnoise = np.sqrt(np.diag(cov))
        indx = 0
        if len(free_inputs)==2:
            self.model.plot(plot_limits=pltlim,fixed_inputs=fixed_inputs,legend=legend,plot_data=plot_data)
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
            print "One free dimension - 1d plot"
            gpmus, gpcovs = self.model.predict(Xtest)
            gpmus *= ys_std
            gpmus += ys_mean
            gpcovs *= ys_std**2
            print "Plotting mean (%d)" % len(gpmus)
            plt.plot(Xtest[:,free_inputs[0]],gpmus)
            ax = plt.gca()           
            if plotGPvar: 
                ax.fill_between(Xtest[:,free_inputs[0]], (gpmus-np.sqrt(gpcovs))[:,0], (gpmus+np.sqrt(gpcovs))[:,0],alpha=0.1,lw=0)
            plt.plot(Xtest[:,free_inputs[0]],preds,alpha=0.2,color='black')
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]-DPnoise*confidencescale,'--k',lw=2)
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]+DPnoise*confidencescale,'--k',lw=2)
            plt.xlim([np.min(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[0]])])
            
            bound = np.std(self.model.X,0)*0.35
            keep = np.ones(self.model.X.shape[0], dtype=bool)
            for finp in fixed_inputs:
               keep = (keep) & (self.model.X[:,finp[0]]>finp[1]-bound[finp[0]]) & (self.model.X[:,finp[0]]<finp[1]+bound[finp[0]])
            plt.plot(self.model.X[keep,free_inputs[0]],ys_mean+self.model.Y[keep]*ys_std,'k.',alpha=0.4)
            
            #gpmu, gpvar = self.model.predict(Xtest,full_cov=False)
            #plt.plot(Xtest[:,free_inputs[0]],gpmu[:,0]-1.96*np.sqrt(gpvar[:,0]+np.diag(cov)),'-k',lw=2,alpha=0.4)
            #plt.plot(Xtest[:,free_inputs[0]],gpmu[:,0]+1.96*np.sqrt(gpvar[:,0]+np.diag(cov)),'-k',lw=2,alpha=0.4)
    
class DPGP_normal_prior(DPGP_prior):
    def __init__(self,model,sens,epsilon,delta):      
        super(DPGP_normal_prior, self).__init__(model,sens,epsilon,delta)
        self.calc_invCov()
        
    def calc_invCov(self):
        """
        TODO
        """
        sigmasqr = self.model.Gaussian_noise.variance[0]
        K_NN_diags = self.model.kern.Kdiag(self.model.X)
        K_NN = self.model.kern.K(self.model.X)
        invCov = np.linalg.inv(K_NN+sigmasqr*np.eye(K_NN.shape[0]))
        self.invCov = invCov
        
    def draw_noise_samples(self,Xtest,N=1,Nattempts=7,Nits=1000):
        """
        For a given set of test points, find DP noise samples for each
        """
        test_cov = self.model.kern.K(Xtest,Xtest)
        msense = self.calc_msense(self.invCov)
        print msense
        ##This code is only necessary for finding the mean (for testing it matches GPy's)
        sigmasqr = self.model.Gaussian_noise.variance[0]
        K_NN = self.model.kern.K(self.model.X)
        K_Nstar = self.model.kern.K(self.model.X,Xtest)
        mu = np.dot(np.dot(K_Nstar.T,np.linalg.inv(K_NN+sigmasqr*np.eye(K_NN.shape[0]))),self.model.Y)
        ##
        samps, samp_cov = self.draw_cov_noise_samples(test_cov,msense,N)
        return mu, samps, samp_cov
      
    
class DPGP_pseudo_prior(DPGP_prior):
    def draw_noise_samples(self,Xtest,N=1,Nattempts=7,Nits=1000):
        """
        For a given set of test points, find DP noise samples for each
        """
        self.model.inference_method = GPy.inference.latent_function_inference.FITC()
        test_cov = self.model.kern.K(Xtest,Xtest)
        sigmasqr = self.model.Gaussian_noise.variance[0]
        K_NN_diags = self.model.kern.Kdiag(self.model.X)
        K_NN = self.model.kern.K(self.model.X)
        
        K_star = self.model.kern.K(Xtest,self.model.Z.values)
        K_NM = self.model.kern.K(self.model.X,self.model.Z.values)
        K_MM = self.model.kern.K(self.model.Z.values)
        invK_MM = np.linalg.inv(K_MM)
        
        #lambda values are the diagonal of the training input covariances minus 
        #(cov of training+pseudo).(inv cov of pseudo).(transpose of cov of training+pseudo)
        lamb = np.zeros(len(self.model.X))
        for i,t_in in enumerate(self.model.X):
            lamb[i] = K_NN_diags[i] - np.dot(np.dot(K_NM[i,:].T,invK_MM),K_NM[i,:])

        #this finds (\Lambda + \sigma^2 I)^{-1}
        diag = 1.0/(lamb + sigmasqr) #diagonal values

        #rewritten to be considerably less memory intensive (and make it a little quicker)
        Q = K_MM + np.dot(K_NM.T * diag,K_NM)

        #find the mean at each test point
        pseudo_mu = np.dot(     np.dot(np.dot(K_star, np.linalg.inv(Q)),K_NM.T) *  diag  ,self.model.Y)
        #un-normalise our estimates of the mean (one using the pseudo inputs, and one using normal GP regression)

        #find the covariance for the two methods (pseudo and normal)
        #K_pseudoInv is the matrix in: mu = k_* K_pseudoInv y
        #i.e. it does the job of K^-1 for the inducing inputs case
        K_pseudoInv = np.dot(np.linalg.inv(Q),K_NM.T) * diag

        invlambplussigma = np.diag(1.0/(lamb + sigmasqr)) 
        assert (K_pseudoInv == np.dot(np.dot(np.linalg.inv(Q),K_NM.T),invlambplussigma)).all() #check our optimisation works

        #find the sensitivity for the pseudo (inducing) inputs
        pseudo_msense = self.calc_msense(K_pseudoInv)

        samps, samp_cov = self.draw_cov_noise_samples(test_cov,pseudo_msense,N)
        return pseudo_mu, samps, samp_cov    

class DPGP_cloaking(DPGP):
    """Using the cloaking method"""
    
    def __init__(self,model,sens,epsilon,delta):      
        super(DPGP_cloaking, self).__init__(model,sens,epsilon,delta)
        assert epsilon<=1, "The proof in Hall et al. 2013 is restricted to values of epsilon<=1."

    def calcM(self,ls,cs):
        """
        Find the covariance matrix, M, as the lambda weighted sum of c c^T
        """
        d = len(cs[0])
        M = np.zeros([d,d])
        ccTs = []
        for l,c in zip(ls,cs):        
            ccT = np.dot(c,c.T)
            #print c,ccT,l,M
            M = M + l*ccT       
            ccTs.append(ccT)
        return M

    def L(self,ls,cs):
        """
        Find L = -log |M| + sum(lambda_i * (1-c^T M^-1 c))
        """
        M = self.calcM(ls,cs)
        Minv = np.linalg.pinv(M)
        t = 0
        for l,c in zip(ls,cs):        
            t += l*(1-np.dot(np.dot(c.T,Minv),c))[0,0]

        return (np.log(np.linalg.det(Minv)) + t)
        #return t
        
    def dL_dl(self,ls,cs):
        """
        Find the gradient dL/dl_j
        """
        M = self.calcM(ls,cs)
        Minv = np.linalg.pinv(M)            
        grads = np.zeros(len(ls))    
        for j in range(len(cs)):        
            grads[j] = -np.trace(np.dot(Minv,np.dot(cs[j],cs[j].T)))     
        return np.array(grads)+1
    
    def findLambdas_grad(self, cs, maxit=700):
        """
        Gradient descent to find the lambda_is

        Parameters:
            cs = list of column vectors (these are the gradients of df*/df_i)

        Returns:
            ls = vector of lambdas

        """
        ls = np.ones(len(cs))*0.7
        lr = 0.05 #learning rate
        for it in range(maxit): 
            lsbefore = ls.copy()
            delta_ls = -self.dL_dl(ls,cs)*lr
            ls =  ls + delta_ls
            ls[ls<0] = 0
            #lr*=0.995
            if np.max(np.abs(lsbefore-ls))<1e-5:
                return ls
            print ".",
        print "Stopped before convergence"
        return ls
    
    def findLambdas_scipy(self,cs, maxit=1000):
        """
        Find optimum value of lambdas, start optimiser with random lambdas.
        """
        #ls = np.ones(len(cs))*0.7
        ls = np.random.rand(len(cs))+0.5
        cons = ({'type':'ineq','fun':lambda ls:np.min(ls)})
        #cons = []
        #for i in range(len(ls)):
        #    cons.append({'type':'ineq', 'fun':lambda ls:ls[i]})
        res = minimize(self.L, ls, args=(cs), method='SLSQP', options={'ftol': 1e-12, 'disp': True, 'maxiter': maxit}, constraints=cons, jac=self.dL_dl)
        ls = res.x 
        #print ls
        return ls
    
    def findLambdas_repeat(self,cs,Nattempts=7,Nits=1000):
        """
        Call findLambdas repeatedly with different start lambdas, to avoid local minima
        """
        bestLogDetM = np.Inf
        bestls = None        
        for it in range(Nattempts):
            print "*"
            import sys
            sys.stdout.flush()
            
            ls = self.findLambdas_grad(cs,Nits)
            if np.min(ls)<-0.01:
                continue
            M = self.calcM(ls,cs)
            logDetM = np.log(np.linalg.det(M))
            if logDetM<bestLogDetM:
                bestLogDetM = logDetM
                bestls = ls.copy()
        if bestls is None:
            print "Failed to find solution"
        #print bestls
        return bestls
    
    def calcDelta(self,ls,cs):
        """
        We want to find a \Delta that satisfies sup{D~D'} ||M^-.5(v_D-v_D')||_2 <= \Delta
        this is equivalent to finding the maximum of our c^T M^-1 c.
        """
        M = self.calcM(ls,cs)
        Minv = np.linalg.pinv(M)
        maxcMinvc = -np.Inf
        for l,c in zip(ls,cs):
            cMinvc = np.dot(np.dot(c.transpose(), Minv),c)
            if cMinvc>maxcMinvc:
                maxcMinvc = cMinvc
        return maxcMinvc

    def checkgrad(self,ls,cs):
        """
        Gradient check (test if the analytical derivative dL/dlambda_i almost equals the numerical one)"""
        approx_dL_dl = []
        d = 0.0001
        for i in range(len(ls)):
            delta = np.zeros_like(ls)
            delta[i]+=d
            approx_dL_dl.append(((self.L(ls+delta,cs)-self.L(ls-delta,cs))/(2*d)))
        approx_dL_dl = np.array(approx_dL_dl)

        print "Value:"
        print self.L(ls,cs)
        print "Approx"
        print approx_dL_dl
        print "Analytical"
        print self.dL_dl(ls,cs)
        print "Difference"
        print approx_dL_dl-self.dL_dl(ls,cs)
        print "Ratio"
        print approx_dL_dl/self.dL_dl(ls,cs)

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
        print(self.sens,c,Delta,self.epsilon,np.linalg.det(M))
        sampcov = ((self.sens*c*Delta/self.epsilon)**2)*M
        samps = np.random.multivariate_normal(np.zeros(len(sampcov)),sampcov,N)
        
        ###This code is only necessary for finding the mean
        mu = np.dot(C,self.model.Y)
        ###
        return mu, samps, sampcov
    
    def plot(self,fixed_inputs=[],legend=False,plot_data=False, steps=None, N=10, Nattempts=1, Nits=500, extent_lower={}, extent_upper={},ys_std=1.0,ys_mean=0.0,plotGPvar=True,confidencescale=1.0):
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
        confidencescale = how wide the CI should be (default = 1 std.dev)
        """
        if steps is None:
            dims = self.model.X.shape[1]-len(fixed_inputs) #get number of dims
            steps = int(100.0**(1.0/dims)) #1d=>100 steps, 2d=>10 steps
        Xtest, free_inputs, _ = compute_Xtest(self.model.X, fixed_inputs, extent_lower=extent_lower, extent_upper=extent_upper, steps=steps)

        preds, mu, cov = self.draw_prediction_samples(Xtest,N,Nattempts=1,Nits=Nits)
        preds *= ys_std
        preds += ys_mean        
        mu *= ys_std
        mu += ys_mean
        cov *= (ys_std**2)

        assert len(free_inputs)<=2, "You can't have more than two free inputs in a plot"
        if len(free_inputs)==1:
            pltlim = [np.min(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[0]])]
        if len(free_inputs)==2:
            pltlim = [[np.min(Xtest[:,free_inputs[0]]),np.min(Xtest[:,free_inputs[1]])],[np.max(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[1]])]] 

        print free_inputs[0]
        print Xtest[:,free_inputs[0]]
        print pltlim
        DPnoise = np.sqrt(np.diag(cov))
        indx = 0
        if len(free_inputs)==2:
            self.model.plot(plot_limits=pltlim,fixed_inputs=fixed_inputs,legend=legend,plot_data=plot_data)
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
            #print "One free dimension - 1d plot"
            gpmus, gpcovs = self.model.predict(Xtest)
            gpmus *= ys_std
            gpmus += ys_mean
            gpcovs *= ys_std**2
            #print "Plotting mean (%d)" % len(gpmus)
            plt.plot(Xtest[:,free_inputs[0]],gpmus)
            ax = plt.gca()           
            if plotGPvar: 
                ax.fill_between(Xtest[:,free_inputs[0]], (gpmus-np.sqrt(gpcovs))[:,0], (gpmus+np.sqrt(gpcovs))[:,0],alpha=0.1,lw=0)
            plt.plot(Xtest[:,free_inputs[0]],preds,alpha=0.2,color='black')
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]-DPnoise*confidencescale,'--k',lw=2)
            plt.plot(Xtest[:,free_inputs[0]],mu[:,0]+DPnoise*confidencescale,'--k',lw=2)
            plt.xlim([np.min(Xtest[:,free_inputs[0]]),np.max(Xtest[:,free_inputs[0]])])
            
            bound = np.std(self.model.X,0)*0.35
            keep = np.ones(self.model.X.shape[0], dtype=bool)
            for finp in fixed_inputs:
               keep = (keep) & (self.model.X[:,finp[0]]>finp[1]-bound[finp[0]]) & (self.model.X[:,finp[0]]<finp[1]+bound[finp[0]])
            plt.plot(self.model.X[keep,free_inputs[0]],ys_mean+self.model.Y[keep]*ys_std,'k.',alpha=0.4)
            
            #gpmu, gpvar = self.model.predict(Xtest,full_cov=False)
            #plt.plot(Xtest[:,free_inputs[0]],gpmu[:,0]-1.96*np.sqrt(gpvar[:,0]+np.diag(cov)),'-k',lw=2,alpha=0.4)
            #plt.plot(Xtest[:,free_inputs[0]],gpmu[:,0]+1.96*np.sqrt(gpvar[:,0]+np.diag(cov)),'-k',lw=2,alpha=0.4)
            
            
        
class Test_DPGP_cloaking(object):
    def test(self):
        sens = 2
        eps = 1.0
        delta = 0.01
        trainX = np.random.randn(50,1)*10 
        #trainX = np.arange(0,10,0.2)[:,None]
        trainy = np.sin(trainX)+np.random.randn(len(trainX),1)*0.5
        Xtest = np.arange(0,10,2)[:,None] #0.2

        mod = GPy.models.GPRegression(trainX,trainy)
        mod.Gaussian_noise = 0.5**2
        mod.rbf.lengthscale = 1.0
        dpgp = DPGP_cloaking(mod,sens,eps,delta)
        mean, noise, sampcov = dpgp.draw_noise_samples(Xtest,2)

        largest_notDP = -np.Inf
        #dpgp, noise, sampcov = get_noise(trainX,trainy,Xtest,sens,eps,delta)
        for perturb_index in range(50): 
            mod = GPy.models.GPRegression(trainX,trainy)
            mod.Gaussian_noise = 0.5**2
            mod.rbf.lengthscale = 1.0
            dpgp = DPGP_cloaking(mod,sens,eps,delta)
            muA, _ = dpgp.model.predict(Xtest)
            pert_trainy = np.copy(trainy)
            pert_trainy[perturb_index]+=sens
            mod = GPy.models.GPRegression(trainX,pert_trainy)
            mod.Gaussian_noise = 0.5**2
            mod.rbf.lengthscale = 1.0
            dpgp = DPGP_cloaking(mod,sens,eps,delta)
            muB, _ = dpgp.model.predict(Xtest)


            dist = multivariate_normal(muA[:,0],sampcov)
            dist_shift = multivariate_normal(muB[:,0],sampcov)
            N = 200000
            #print("These two numbers should be less than delta=%0.4f" % dpgp.delta)
            #print("Note epsilon = %0.4f" % dpgp.epsilon)
            pos = np.random.multivariate_normal(muA[:,0],sampcov,N)
            proportion_notDP_A = np.mean( (dist.pdf(pos)/dist_shift.pdf(pos))>np.exp(dpgp.epsilon) )
            pos = np.random.multivariate_normal(muB[:,0],sampcov,N)
            proportion_notDP_B = np.mean( (dist_shift.pdf(pos)/dist.pdf(pos))>np.exp(dpgp.epsilon) )
            assert proportion_notDP_A < dpgp.delta
            assert proportion_notDP_B < dpgp.delta

            largest_notDP = np.max([largest_notDP,proportion_notDP_A,proportion_notDP_B])
        print "The largest proportion of values exceeding the epsilon-DP constraint is %0.6f. This should be less than delta, which equals %0.6f" % (largest_notDP, dpgp.delta)
