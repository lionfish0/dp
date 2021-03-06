import dp4gp_datasets
import dp4gp
import random
import numpy as np
import GPy
import dp4gp_integral_histogram
import pandas as pd

def get_house_prices():
    #Skip this if you want, and load precomputed data from the csv file below:
    #dp4gp_datasets.prepare_preloaded_prices('test.csv', boundingbox=[480e3, 130e3, 580e3, 230e3], N=10000, col_list=['QS501EW'])
    #Load precomputed dataset:
    dataset = pd.read_csv('price_dataset10k.csv') #london, 10k purchases

    #Reduce the size of the dataset and split into training and test data:
    #dataset = dataset[dataset['years']>2005]
    dataset = dataset.ix[random.sample(dataset.index, 200)]

    #get into useful form
    #east, north, time, education -> price
    inputs = np.vstack([dataset['easting'].values,dataset['northing'].values,dataset['seconds'].values,(dataset['QS501EW_6']/dataset['QS501EW_0']).values]).T

    #inputs = inputs[:,0:2]
    ys = dataset['price'].values
    
    return inputs, ys
    
def get_citibike_data():
    allcb = dp4gp_datasets.load_citibike(station=None)
    subcb = allcb[allcb['usertype']=='Subscriber']

    cb = subcb.ix[random.sample(subcb.index, 5000)]
    inputs = np.c_[cb['start station latitude'],cb['end station latitude'],cb['start station longitude'],cb['end station longitude']]
    ys = cb['tripduration'].values
    
    return inputs, ys
    

def get_noDP_prediction(training_inputs, training_ys, test_inputs, sens, eps, delta,noise,modvar,kernval,kern_ls,steps):
    rbf = GPy.kern.RBF(training_inputs.shape[1],kernvar, kern_ls,ARD=True)
    mod = GPy.models.GPRegression(training_inputs,training_ys,rbf)
    mod.Gaussian_noise = noise
    dpgp = dp4gp.DPGP_cloaking(mod,sens,eps,delta)
    preds, mu, cov = dpgp.draw_prediction_samples(test_inputs,1,1,0)
    return mu, None


def get_cloaking_prediction(training_inputs, training_ys, test_inputs, sens, eps, delta,noise,modvar,kernval,kern_ls,steps):
    rbf = GPy.kern.RBF(training_inputs.shape[1],kernvar, kern_ls,ARD=True)
    #rbf = GPy.kern.RBF(training_inputs.shape[1],modvar,[5e3,5e3],ARD=True)
    mod = GPy.models.GPRegression(training_inputs,training_ys,rbf)
    mod.Gaussian_noise = noise
    dpgp = dp4gp.DPGP_cloaking(mod,sens,eps,delta)
    preds, mu, cov = dpgp.draw_prediction_samples(test_inputs,1,2,1000)
    return preds, cov
    
def get_integral_prediction(training_inputs, training_ys, test_inputs, sens, eps, delta,noise,modvar,kernval,kern_ls,steps):
    Xtest, free_inputs, step = dp4gp.compute_Xtest(training_inputs,steps=steps)
    print step
    dpgp = dp4gp_integral_histogram.DPGP_integral_histogram(sens,eps,delta)
    dpgp.prepare_model(Xtest,training_inputs,step,training_ys,lengthscale=kern_ls)
    #dpgp.optimize()
    dpgp.model.optimize(messages=True)#, max_iters=2)
    preds, cov = dpgp.draw_prediction_samples(test_inputs)
    return preds, cov

def get_standard_prediction(training_inputs, training_ys, test_inputs, sens, eps, delta,noise,modvar,kernval,kern_ls,steps):
    rbf = GPy.kern.RBF(training_inputs.shape[1],kernvar,kern_ls,ARD=True)
    mod = GPy.models.GPRegression(training_inputs,training_ys,rbf)
    mod.Gaussian_noise = noise
    dpgp = dp4gp.DPGP_normal_prior(mod,sens,eps,delta)
    preds, mu, cov = dpgp.draw_prediction_samples(test_inputs,1)
    return preds, cov

def get_pseudo_prediction(training_inputs, training_ys, test_inputs, sens, eps, delta,noise,modvar,kernval,kern_ls,steps):
    rbf = GPy.kern.RBF(training_inputs.shape[1],kernvar,kern_ls,ARD=True)
    print training_ys.ndim
    print training_inputs.shape
    mod = GPy.models.SparseGPRegression(training_inputs,training_ys,kernel=rbf,num_inducing=40) #no idea how many inducing!
    mod.inference_method = GPy.inference.latent_function_inference.FITC()
    mod.set_Z(training_inputs[0:40,:]) #grab random inputs as pseudoinputs
    mod.Gaussian_noise = noise  
    dpgp = dp4gp.DPGP_pseudo_prior(mod,sens,eps,delta)
    preds, mu, cov = dpgp.draw_prediction_samples(test_inputs,1)
    return preds, cov

#import os
#os.system("aws s3 cp working.txt s3://dpresults/working.txt")
    
import sys
steps = int(sys.argv[1])
eps = float(sys.argv[2])
lengthscale = float(sys.argv[3])
gaussian_noise = float(sys.argv[4])
filename = sys.argv[5]


#inputs, ys = get_house_prices()
inputs, ys = get_citibike_data()

#FOR CITIBIKE
#squash data into 0-2000 seconds range
ys[ys>2000] = 2000
ys[ys<0] = 0
sens = 2000-0
kernvar = 10.0
kern_ls = np.array([lengthscale,lengthscale,lengthscale,lengthscale]) #0.05

#FOR HOUSE PRICES
#squash data into gbp10k-500k range
#ys[ys>5e5] = 5e5
#ys[ys<1e4] = 1e4
#sens = 5e5-1e4
#kernvar = 1.0
#kern_ls = [15e3,15e3,50*31536000,5.0]

ys_mean = np.mean(ys)
ys_std = np.std(ys)
ys = ys - ys_mean
ys = ys / ys_std
sens = sens / ys_std

if (eps>10000.0):
    eps = 1.0
    sens = 0.0001

training_inputs = inputs[0:-100,:]
training_ys = ys[0:-100][:,None]
test_inputs = inputs[-100:,:]
test_ys = ys[-100:][:,None]

#fns = [get_noDP_prediction, get_integral_prediction, get_cloaking_prediction]#,get_pseudo_prediction,get_standard_prediction]
#labels = ["No DP","Integral","Cloaking"]#,"Pseudo","Standard"]

#fns = [get_integral_prediction]#, 
fns = [get_cloaking_prediction]
labels = ["Cloaking"]#["Integral"]#,"Cloaking"]

results = []
for fn,label in zip(fns,labels):
    preds, cov = fn(training_inputs,training_ys,test_inputs,sens,eps,0.01,gaussian_noise,1.0, kernvar, kern_ls, steps)
    RMSE = np.sqrt(np.mean((test_ys-preds)**2))
    results.append({'label':label, 'preds':preds, 'cov':cov, 'RMSE':RMSE})

textfile = open(filename, "a")
resstring = "%0.5f, %0.5f, %0.5f, %0.5f, %d, %0.5f, %0.5f " % (sens,ys_mean,ys_std,eps,steps,lengthscale,gaussian_noise)
for r in results:
    resstring+= "%s, %0.5f, " % (r['label'],r['RMSE'])
resstring+='\n';
textfile.write(resstring)
textfile.close()
import os
os.system("aws s3 cp %s s3://dpresults/%s" % (filename, filename))
#how to detect number of instances: ps a | grep 'python\ gen_paper_results.py' | wc -l
