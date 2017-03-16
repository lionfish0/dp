# Methods for combining differential privacy with Gaussian Processes

import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


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
        normal_msense = msense(invCov)

    #sample_covariance values are the diagonal of the test input covariances minus 
    #(cov of test+pseudo).(inv cov of pseudo).(transpose of cov of test+pseudo)
    return test_cov, normal_msense, pseudo_msense, normal_mu, pseudo_mu, K_normal, K_pseudoInv, normal_covars
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##### Functions for vector-DP methods

#The test and training input locations
#tests = np.arange(-10,20,0.5) #np.array([0.0,1.0,2.0]) #np.arange(-5,15,1.0) #
#train = np.arange(0,10,0.1) #np.array([0.0,1.5]) #np.arange(0,5,0.1) # #

#other parameters (lengthscale, noise-std, (eps,delta)-DP)
#lengthscales = 4.0
#sigma = 1.0
#delta = 0.01
#eps = 2.0
#sens = 8 #sens=\Delta_y i.e. how much the outputs can vary by.

def calc_covariances(train,tests,lengthscales,sigma):
    #calculate covariance between test+train, and between training points
    Kstar = np.zeros([len(tests),len(train)])
    for i,x in enumerate(tests):
        for j,y in enumerate(train):
            Kstar[i,j] = rbf(x,y,lengthscales)
    K = np.zeros([len(train),len(train)])
    for i,x in enumerate(train):
        for j,y in enumerate(train):
            K[i,j] = rbf(x,y,lengthscales)
    K+=sigma**2*np.eye(len(train)) #add diagonal sample noise variance
    return K, Kstar

def calcM(ls,cs):
    """
    Find the covariance matrix, M, as the lambda weighted sum of c c^T
    """
    d = len(cs[0])
    M = np.zeros([d,d])
    ccTs = []
    for l,c in zip(ls,cs):
        ccT = l*np.dot(c,c.transpose())
        M = M + ccT
        ccTs.append(ccT)
    return M, ccTs

def calc_Delta(M,cloak):
    halfM = np.real(scipy.linalg.sqrtm(M)) #occasionally goes minutely complex.
    invhalfM = np.linalg.pinv(halfM)
    assert np.sum(np.dot(halfM,halfM)-M)<0.001 #(M^.5).(M^.5) should equal M.
    inner = np.dot(invhalfM,cloak) #this is how much M^-.5 (v_D - v_D') move for each test/training point pair
    Delta = np.max(np.sqrt(np.sum(inner**2,0))) #here we find the norm_2 for each column, and find the max of these norms    
    return Delta
    
    
#### use approximate 'greedy' method
def findM_greedy(cs): #TODO Combine cloak (C) and cs!
    ccTs = []
    d = len(cs[0])
    tempeye = 1e-8
    M = np.eye(d)*tempeye
    C = np.zeros([d,0])
    
    for c in cs:
        ccT = np.dot(c,c.transpose())
        ccTs.append(ccT)

    for it in np.random.permutation(range(len(cs))): #pick one random training point at a time
        c = cs[it]
        C = np.hstack([C,c])
        ccT = ccTs[it]
        
        Minv = np.linalg.inv(M)
        while np.any(np.diag(np.dot(np.dot(C.T,Minv),C))>1):
            M = M + ccT * 0.1 #increase this lambda a little
            Minv = np.linalg.inv(M)
    return M
    
def findM_greedy_loop(cs):
    bestM = None
    bestTrace = np.Inf
    for attempt in range(100):
        M = findM_greedy(cs)
        tr = np.trace(M)
        if (tr<bestTrace):
            bestTrace = tr
            bestM = M
        if attempt % 10 == 0: print ".",
    return bestM
        
    
#### use scipy grad descent
def f(ls,cs):
    M,ccTs = calcM(ls,cs)
    return -np.log(np.linalg.det(M))

def findLambdas_scipy(cs,tempcloak):
    ls = np.ones(len(cs))
    M, ccTs = calcM(ls,cs)
    rank = np.linalg.matrix_rank(M) #TODO
    rank = np.min(tempcloak.shape) #might not be true
    print "rank=%d" % rank
    cons = ({'type': 'eq', 'fun' : lambda ls: np.sum(ls)-rank},{'type':'ineq','fun':lambda ls:np.min(ls)})
    #res = minimize(f, ls, args=(cs), method='BFGS', jac=fgrad, options={'xtol': 1e-8, 'disp': True})
    #res = minimize(f, ls, args=(cs), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    res = minimize(f, ls, args=(cs), method='SLSQP', options={'xtol': 1e-8, 'disp': True}, constraints=cons)
    ls = res.x 
    return ls
    
    
#### use homebrew method
def findLambdas(cs,tempcloak,max_its = 10000, lr = 1.0):
    #TODO Rewrite with just a matrix C, instead of a messy list of vectors.
    #also tempcloak is here just to let us temporarily compute Delta for debugging purposes.
    #this matrix is the 'C' matrix which will be the only parameter.

    #lambdas  (how to initialise?)
    ls = np.ones(len(cs))  
    print cs
    #gradient descent
    lsbefore = ls.copy()
    best_delta = np.Inf
    best_ls = ls.copy()
    for it in range(max_its):
        M,ccTs = calcM(ls,cs)
        Minv = np.linalg.pinv(M)

        #find new Trace (P sum(cc^T)) for each c
        TrPccTs = []
        for ccT in ccTs:
            TrPccTs.append(np.trace(np.dot(Minv,ccT)))####<<<TODO CHECK THIS SHOULDN'T BE M!!!
            #TrPccTs.append(np.trace(np.dot(M,ccT)))

        #normalise lambda (should sum to either n or d)!!!
        deltals = np.array(TrPccTs)*lr
        #deltals = np.dot(rand_bin_array(25,len(deltals)),deltals)
        ls = np.array(ls) - deltals
        print ls
    #    ls /= np.sum(ls)
    #    ls *= np.linalg.matrix_rank(M)
        #ls[ls>0.95] = 1.0
        #ls[ls<0.05] = 0.0
    #    if (np.sum((lsbefore-ls)**2)<1e-10*lr):#convergence
    #        print("Converged after %d iterations" % it)
    #        break #reached ~maximum
            
    #    M,ccTs = calcM(ls,cs)
    #    Delta = calc_Delta(M,tempcloak) #temporarily calculated!
    #    if it % (max_its/10) == 1:
    #        lr = lr * 0.5
    #        print("sum squared change in lambdas: %e. Delta=%0.4f. Delta^2 x det(M)=%e, (lr = %e)" % (np.sqrt(np.sum((lsbefore-ls)**2))/lr,Delta,Delta**2 * np.linalg.det(M),lr))
    ##    if Delta<best_delta:
    ##        best_delta = Delta
    ##        best_ls = np.array(ls.copy())
    ##    lsbefore = ls.copy()
    #if it==max_its-1:
    #    #TODO Throw exception? 
    #    print("Ended before convergence")
    #    ls = best_ls #we'll go with the best ones we've found.
    return ls
####
    
    
def calc_DP_noise(train,tests,lengthscales,delta_Y,sigma,eps,delta,method='greedy_loop',max_its = 10000, lr = 1.0):
    #TODO Add comment describing fn
    #TODO Add asserts on shapes of train and test etc
    K,Kstar = calc_covariances(train,tests,lengthscales,sigma)

    
    #cloak is a matrix describing how much each training point affects each test point
    cloak = np.dot(Kstar,np.linalg.inv(K))
    cs = []
    for c in (cloak.transpose()):  #TODO We need to turn loops into matrix operations for speed
        cs.append(c[:,None])

    if method=='homebrew':
        ls = findLambdas(cs,cloak,max_its, lr)
        M,ccTs = calcM(ls,cs)

    if method=='greedy':
        #this almost uses the lambdas, but has a slightly modified different M
        M = findM_greedy(cs)
        
    if method=='greedy_loop':
        #tries the greedy method repeatedly
        M = findM_greedy_loop(cs)
        
    print np.linalg.det(M)
    print M.trace()    
    
    Delta = calc_Delta(M,cloak)
    print("Delta = %0.2f" % Delta) #we expect this to be <=1 because of our method for creating M
    if Delta>1.0000001: #<=1 (but added .0001 for numerical instability. Note: doesn't really matter for DP if it is >1)
        print("WARNING: Delta should be <=1, but is %0.5f" % Delta) 
    #DP calculation to find scaled GP DP noise samples
    c = np.sqrt(2*np.log(2/delta))
    sampcov = ((delta_Y*c*Delta/eps)**2)*M
    return sampcov, K, Kstar
