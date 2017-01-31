# Methods for combining differential privacy with Gaussian Processes

import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

######### Functions to help with adding noise to the inputs

def buildlist(bins):
    '''
    Produces a matrix. Each row describes one bin.
    bins: a list of D numpy arrays (one for each dimension),
    each one specifying the boundaries of each bin (Mx bins)
    e.g. [np.array([1,2,3]),np.array([4,6,8,10])]
    the arrays contain varying number of bin boundaries,
    M1+1,M2+1..MD+1
    
    outputs: a D dimensional array containing the values
    
    returns
    a matrix of size (M1*M2*...*MD) x (2*D)
    each row is one bin in the D-dimensional histogram
    each pair of columns are the boundary values of a bin
    in one dimension, so in the example above the result
    would be:
     1 2 4  6
     1 2 6  8
     1 2 8 10
     2 3 4  6
     2 3 6  8
     2 3 8 10
    M1=2, M2=3, so the matrix size is (2*3) x (2*2) = 6x4
    
    This is the same order of items that feeding the histogram
    into squeeze produces.
    '''
    
    boundaries = None
    for b2,b1 in zip(bins[0][0:-1],bins[0][1:]):
        if len(bins)>1:
            new = np.array(buildlist(bins[1:]))
            old = np.repeat(np.array([b1,b2])[None,:],len(new),axis=0)
            
            newrows = np.hstack([old,new])
            
        else:
            newrows = np.array([b1,b2])[None,:]
        if boundaries is None:
            boundaries = newrows
            
        else:
            boundaries = np.vstack([boundaries,newrows])

    return boundaries

def findcentres(area_row_list):
    '''
    Takes the matrix (Lx(2D)) buildlist returns and
    Finds the mean of each pair of bin boundaries
    reults in an LxD matrix
    
    for the example given in buildlist's comment,
    the output of this function would be:
     1.5 5
     1.5 7
     1.5 9
     2.5 5
     2.5 7
     2.5 9
    '''
    out = []
    for d in range(0,area_row_list.shape[1],2):
        out.append(np.mean(area_row_list[:,d:(d+2)],axis=1))
    return(np.array(out).T)

def get_Gaussian_DP_noise(epsilon,delta,sensitivity):
    '''
    Given DP parameters (epsilon, delta and the sensitivity)
    returns the standard deviation of the Gaussian noise we should add.
    '''
    csqr = 2*np.log(1.25/delta)
    sigma = np.sqrt(csqr)*sensitivity/epsilon
    return sigma

def generate_Gaussian_DP_noise(sigma,shape):
    '''
    Given the standard deviation of the noise, sigma
    generates an array specified by shape.
    '''
    noise = np.random.normal(0,sigma,shape)
    return noise

def bin_dataframe(df,axes,density=True,verbose=False):
    '''
    Bins the data in a dataframe, by the list of tuples in 'axes'.
    each tuple specifies the name of a column and the range and step size to bin it with, e.g.:
    [('seconds',0,24*3600,60*5),('gender',None,None,1)] or set the ranges to None,
    and just provide a single step size, and let the tool decide on the bounds

    If density is true, then the result is divided by the area of the bins, to give a
    density. Useful if comparing between different bin sizes, etc.

    returns:
    output = histogram
    represent the data of the histogram in a list:
    point_row_form = each row is one histogram cell's centroid location
    area_row_form = each row is one histogram cell's location, with the bounds specified
    output_row_form = each row is the value of that histogram cell
    bins = list of arrays, each one a list of boundaries for each bin
    '''
    bins = []
    for i,axis in enumerate(axes):
        column = axis[0]
        start = axis[1]
        end = axis[2]
        step = axis[3]
        if (start==None):
            print("Warning: Tool automatically deciding on histgram bounds.");
            s = df[column]
            s = s[~np.isnan(s)]
            s = np.sort(s)
            N = s.shape[0]
            start = s[int(N*0.03)] #get rid of outliers
            end = s[int(N*0.97)]
            delta= (end-start)*0.04 #add 10% on each end, to catch any outside the range.
            start -= delta
            end += delta
            #step = (end-start)/step            
        axes[i] = column,start,end,step
        bins.append(np.arange(start,end+step,step))
    data = df[[axis[0] for axis in axes]].as_matrix()
    
    output = np.histogramdd(data,bins)[0]
    area = np.prod([b[3] for b in axes])
    output /= area
    if verbose:
        bincount = np.prod([len(b)-1 for b in bins])
        print("Bin count: %d" % bincount)
        datacount =  (df.shape[0])
        print("Data Length: %d" % datacount)
        print("Average occupancy: %0.2f" % (1.0*datacount/bincount))
        print("Area: %0.5f" % area)
        print("Area x Density: %0.4f" % (np.sum(output)*area))
        print("%0.2f%% of data were not included." % (100*(1-((np.sum(output)*area)/datacount))))
    
    area_row_form = buildlist(bins)
    point_row_form = findcentres(area_row_form)
    output_row_form = output.flatten() #TODO: CHECK ORDER IS CORRECT
    #assert np.sum(output)==df.shape[0], "Not all data points have been counted"
    return output,point_row_form,area_row_form,output_row_form,bins

def transform(data,normalisation_mean=np.NAN,normalisation_std=np.NAN):
    '''
    Normalises the data, or scales using the normalisation mean and std. Returns the used mean and std
    '''
    #data[data<0] = 0
    #result = np.sqrt(data)
    result = data
    if normalisation_mean is np.NAN:
        normalisation_mean = np.mean(result)
        normalisation_std = np.std(result)
        result = result - normalisation_mean
        result = result / normalisation_std
        return result, normalisation_mean, normalisation_std
    else:
        result = result - normalisation_mean
        result = result / normalisation_std
        return result



def untransform(data,normalisation_mean,normalisation_std):
    '''
    Un-normalises the data using the mean and std specified
    '''
    data = data * normalisation_std
    data = data + normalisation_mean
    #result = data**2
    result = data
    return result
    
    
######### Functions for adding DP noise to the outputs

def rbf(x,xprime,l):
    """
    kernel function: takes two values x and xprime, and finds the covariance between them.
      x, xprime = vectors each representing one point each (so Dx1)
      l = lengthscale vector (Dx1).
    
    returns a scalar value describing the covariance between these locations.
    """
    
    return np.exp(-.5*np.sum(((x-xprime)/l)**2))
    #return m.kern.K(np.array([[x]]),np.array([[xprime]])) #could try using GPy kernels in future?

def k(x,xprime,l,scale,kern):
    """
    calculate covariance: takes two values x and xprime, and finds the covariance between them.
      x, xprime = vectors each representing one point each (so Dx1)
      l = lengthscale vector (Dx1).
      scale = dictionary of scalings for locations in input domain, or None if not scaled
      kern = kernel function
    
    returns a scalar value describing the covariance between these locations.
    """
    if scale is None:
        return kern(x,xprime,l)
    else:
        return scale[x]*scale[xprime]*kern(x,xprime,l)

def msense(A):
    """
    originally returned the infinity norm*, but we've developed an improved value from
    this norm which only cares about values of the same sign (it is assumed that
    those of the opposite sign will work to reduce the sensitivity). We'll call
    this the matrix_sensitivity or msense
    * np.max(np.sum(np.abs(A),1))
    """
    v1 = np.max(np.abs(np.sum(A.copy().clip(min=0),1)))
    v2 = np.max(np.abs(np.sum((-A.copy()).clip(min=0),1)))
    #print "msense"
    #print np.abs(np.sum(A.copy().clip(min=0),1))
    return np.max([v1,v2])


def get_noise_scale(y_in,test_inputs,training_inputs,pseudo_inputs,lengthscales,sigma,verbose=False,calc_normal=False,scale=None):
    '''
    Finds the sensitivity using the inverse covariance matrix
    and the sensitivity of the $Q^{-1} K_{uf} (\Lambda + \sigma^2 I)^{-1}$
    matrix (which relies on inducing inputs to reduce the value
    of this norm) See the Differentially Private Gaussian Processes
    paper and [1] for more details.
    
    This assumes we're using the RBF kernel. It also uses the lengthscales
    specified.
    
    [1] Hall, Rinaldo and Wasserman. DP for functions and functional data. 
    Journal of Machine Learning Research 14.Feb (2013): 703-727.
    
    parameters:
    y = the output values (currently only handles single output functions)
    test_inputs = the input values to test with
    training_inputs = the training data (describes the GP)
    pseudo_inputs = the inducing inputs
    lengthscales = a vector of lengthscales (one for each input dimension)
    sigma = noise standard deviation.
    verbose (optional) = set to true for verbiage.
    calc_normal = whether to also calculate the result using the inverse cov mat
    scale = None default = set to a dictionary of scalings. The dictionary should cover all possible inputs. 
              This scales the variances of the outputs.
    
    returns:
     test_cov = the covariance matrix (based on the kernel and the
                locations of the test inputs.
    
     normal_msense = the matrix sensitivity (formerly the infinity norm) of the inverse covariance matrix
     pseudo_msense = the equivalent when using inducing inputs

    normalise outputs (y): zero mean, unit variance
    '''

    y = y_in.copy()
    sigmasqr = sigma**2
    
    normal_peroutput_msense = None
    normal_msense = None
    normal_mu = None  
    K_normal = None

    #covariance between test inputs
    print "Calculating covariance between test inputs"
    sys.stdout.flush()
    test_cov = np.zeros([len(test_inputs),len(test_inputs)])
    for i,t_in1 in enumerate(test_inputs):
        for j,t_in2 in enumerate(test_inputs):
            test_cov[i,j] = k(t_in1,t_in2,lengthscales,scale,rbf)
    #print "Minimum eigen value: %0.3f" % np.min(np.linalg.eig(test_cov)[0])
    #print "Maximum K** %0.2f" % np.max(test_cov) #make this an assertion <=1    

    #covariance between training inputs and test inputs
    print "Calculating covariance between training inputs and test inputs"
    sys.stdout.flush()
    K_Nstar = np.zeros([len(training_inputs),len(test_inputs)])
    for i,t_in in enumerate(training_inputs):
        for j,p_in in enumerate(test_inputs):
            K_Nstar[i,j] = k(t_in,p_in,lengthscales,scale,rbf)
    #print "Maximum k* %0.2f" % np.max(K_Nstar) #make this an assertion <=1

    #covariance between training inputs and pseudo inputs
    print "Calculating K_NM"
    sys.stdout.flush()
    K_NM = np.zeros([len(training_inputs),len(pseudo_inputs)])
    for i,t_in in enumerate(training_inputs):
        for j,p_in in enumerate(pseudo_inputs):
            K_NM[i,j] = k(t_in,p_in,lengthscales,scale,rbf)

    #covariance between pseudo inputs
    print "Calculating K_MM"
    sys.stdout.flush()
    K_MM = np.zeros([len(pseudo_inputs),len(pseudo_inputs)])
    for i,p_in1 in enumerate(pseudo_inputs):
        for j,p_in2 in enumerate(pseudo_inputs):
            K_MM[i,j] = k(p_in1,p_in2,lengthscales,scale,rbf)
    invK_MM = np.linalg.inv(K_MM)

    #variance of training inputs
    print "Calculating K_NN diagonals"
    sys.stdout.flush()
    K_NN_diags = np.zeros([len(training_inputs)])
    for i,t_in1 in enumerate(training_inputs):
        K_NN_diags[i] = k(t_in1,t_in1,lengthscales,scale,rbf)

    #covariance between test inputs and pseudo inputs
    print "Calculating K_star"
    sys.stdout.flush()
    K_star = np.zeros([len(test_inputs),len(pseudo_inputs)])
    for i,t_in in enumerate(test_inputs):
        for j,p_in in enumerate(pseudo_inputs):
            K_star[i,j] = k(t_in,p_in,lengthscales,scale,rbf)
    print "Maximum k* %0.2f" % np.max(K_star) #make this an assertion <=1            
            
    #covariance between training inputs
    if calc_normal: #whether to calculate this (as we might run out of memory/time)
        print "Calculating K_NN"
        sys.stdout.flush()
        K_NN = np.zeros([len(training_inputs),len(training_inputs)])
        for i,t_in1 in enumerate(training_inputs):
            for j,t_in2 in enumerate(training_inputs):
                K_NN[i,j] = k(t_in1,t_in2,lengthscales,scale,rbf)

    #lambda values are the diagonal of the training input covariances minus 
    #(cov of training+pseudo).(inv cov of pseudo).(transpose of cov of training+pseudo)
    print "Calculating lambda"
    sys.stdout.flush()
    lamb = np.zeros(len(training_inputs))
    for i,t_in in enumerate(training_inputs):
        lamb[i] = K_NN_diags[i] - np.dot(np.dot(K_NM[i,:].T,invK_MM),K_NM[i,:])

    #this finds (\Lambda + \sigma^2 I)^{-1}
    diag = 1.0/(lamb + sigmasqr) #diagonal values
    
    #rewritten to be considerably less memory intensive (and make it a little quicker)
    Q = K_MM + np.dot(K_NM.T * diag,K_NM)

    #find the mean at each test point
    pseudo_mu = np.dot(     np.dot(np.dot(K_star, np.linalg.inv(Q)),K_NM.T) *  diag  ,y)
    if calc_normal:
        normal_mu = np.dot(np.dot(K_Nstar.T,np.linalg.inv(K_NN+sigmasqr*np.eye(K_NN.shape[0]))),y)
        normal_covars = test_cov - np.dot(np.dot(K_Nstar.T,np.linalg.inv(K_NN+sigmasqr*np.eye(K_NN.shape[0]))),K_Nstar)
   
    #un-normalise our estimates of the mean (one using the pseudo inputs, and one using normal GP regression)

    #find the covariance for the two methods (pseudo and normal)
    #K_pseudoInv is the matrix in: mu = k_* K_pseudoInv y
    #i.e. it does the job of K^-1 for the inducing inputs case
    K_pseudoInv = np.dot(np.linalg.inv(Q),K_NM.T) * diag
    
    invlambplussigma = np.diag(1.0/(lamb + sigmasqr)) 
    assert (K_pseudoInv == np.dot(np.dot(np.linalg.inv(Q),K_NM.T),invlambplussigma)).all() #check our optimisation works
    
    #find the sensitivity for the pseudo (inducing) inputs
    pseudo_msense = msense(K_pseudoInv)
    
    #if we're finding the 'normal' GP results
    if calc_normal:
        K_normal = K_NN + sigmasqr * np.eye(K_NN.shape[0])
        invCov = np.linalg.inv(K_normal)
        #normal GP sensitivity
        normal_msense = msense(invCov)
        #this finds the sensitivity per test point (using the training inputs)
        normal_peroutput_msense = np.max(np.abs(np.dot(K_Nstar.T, invCov)),1)
        print "THING"
        print np.dot(K_Nstar.T, invCov).shape
        print np.dot(K_Nstar.T, invCov)
       # print test_cov**-.5
        tot = np.dot(test_cov**-.5,np.max(np.dot(K_Nstar.T, invCov),1))
       # print tot
        print np.sqrt(np.sum(tot**2))
       # print "---"
       # normal_msense = np.sqrt(np.sum(tot**2))
        
        #s = np.zeros([len(training_inputs),len(test_inputs)])
        #for m,t_in3 in enumerate(training_inputs):
        #    for j,t_in2 in enumerate(test_inputs):
        #        for i,t_in in enumerate(training_inputs):
        #            print invCov[i,m] * K_Nstar[i,j]
        #            s[m,j] += invCov[i,m] * K_Nstar[i,j]
        #np.max(s) should be equal to max(normal_peroutput_msense). Need assert?
        #print "K_Nstar!"
        #print K_Nstar
        #print "s!"
        #print s
        #print "invCov"
        #print invCov
        #print "normal msense"
        #print normal_msense
    #print "MAX:"
    #print np.max(K_Nstar)
        
    #this finds the sensitivity per test point (using the inducing inputs)
    pseudo_peroutput_msense = np.max(np.abs(np.dot(K_star, K_pseudoInv)),1)
    
    #sample_covariance values are the diagonal of the test input covariances minus 
    #(cov of test+pseudo).(inv cov of pseudo).(transpose of cov of test+pseudo)
    return test_cov, normal_msense, pseudo_msense, normal_peroutput_msense, pseudo_peroutput_msense, normal_mu, pseudo_mu, K_normal, K_pseudoInv, normal_covars
    
def draw_sample(test_cov, test_inputs, mu, msense, sens, delta, eps, verbose=False):
    """
    Produce a differentially private set of predictions at the test inputs,
    using the covariance specified in test_cov.
    
    test_cov = covariance between test points
    test_inputs = locations of test inputs
    mu = un-DP GP mean function
    msense = 'matrix sensitivity'
    sens = data sensitivity (how much can one output value vary due to one person
    delta = probability of providing DP
    eps = epsilon (DP parameter)
    """
    G = np.random.multivariate_normal(np.zeros(len(test_inputs)),test_cov)
    noise = G*sens*np.sqrt(2*np.log(2/delta))/eps
    noise = noise * msense #we want to do element-wise product 
    dp_mu = np.array(mu) + noise
    
    if verbose:
        print("y Sensitivity: %0.4f" % sens)
        print("M sense:       %0.4f" % msense)
        print("Noise scale:   %0.4f" % (sens*np.sqrt(2*np.log(2/delta))/eps))
        print("Total noise:   %0.4f" % (msense*sens*np.sqrt(2*np.log(2/delta))/eps))
        
        
            
    return dp_mu
