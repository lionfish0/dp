

def findLambdas(cs,tempcloak):
    #TODO Rewrite with just a matrix C, instead of a messy list of vectors.
    #also tempcloak is here just to let us temporarily compute Delta for debugging purposes.
    #this matrix is the 'C' matrix which will be the only parameter.

    #lambdas  (how to initialise?)
#    ls = np.ones(len(cs))
    ls = 1+np.random.randn(len(cs))
    
    
    #learning rate
    lr = 1.0 #TODO Pick reasonable number?
    #gradient descent
    lsbefore = ls.copy()
    max_its = 1000
    best_delta = np.Inf
    best_ls = ls.copy()
    for it in range(max_its):
        M,ccTs = calcM(ls,cs)
        #M = M + np.eye(len(M))*0.0000001
        Minv = np.linalg.pinv(M)

        #find new Trace (P sum(cc^T)) for each c
        TrPccTs = []
        for ccT in ccTs:
            TrPccTs.append(np.trace(np.dot(Minv,ccT)))

        #normalise lambda (should sum to either n or d)!!!
        deltals = -np.array(TrPccTs)*lr
        #deltals = np.dot(rand_bin_array(25,len(deltals)),deltals)
        ls = np.array(ls) + deltals
        ls /= np.sum(ls)
        ls *= np.linalg.matrix_rank(M)
        #ls[ls>0.95] = 1.0
        #ls[ls<0.05] = 0.0
        if (np.sum((lsbefore-ls)**2)<1e-20):
            print("Converged after %d iterations" % it)
            break #reached ~maximum
        Delta = calc_Delta(M,tempcloak) #temporarily calculated!
        if it % 100 == 1:
            lr = lr * 0.8
            print("sum squared change in lambdas: %e. Delta=%0.4f. Delta^2 x det(M)=%e, (lr = %e)" % (np.sqrt(np.sum((lsbefore-ls)**2))/lr,Delta,Delta**2 * np.linalg.det(M),lr))
        if Delta<best_delta:
            best_delta = Delta
            best_ls = ls
        lsbefore = ls.copy()
        
    if it==max_its-1:
        #TODO Throw exception? 
        print("Ended before convergence")
    ls = best_ls #we'll go with the best ones we've found!
    return ls
