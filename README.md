# Differential Privacy for GPs

This repo covers my recent experimentation with Differential Privacy.

The key module is https://github.com/lionfish0/dp/blob/master/dp4gp.py

## Demo notebooks

Demonstrates using the dp4gp module
https://github.com/lionfish0/dp/blob/master/Demonstrating%20dp4gp.ipynb

## Other

Simple example to test the integral kernel
https://github.com/lionfish0/dp/blob/master/Building%20Histogram%20Class.ipynb

## Paper results
Used for plotting, etc
https://github.com/lionfish0/dp/blob/master/Paper%20Results%201d%20Kung%20dataset.ipynb

Handling results from AWS computations
https://github.com/lionfish0/dp/blob/master/Analyse%20DP%20results%20from%20AWS.ipynb

House price map creation
https://github.com/lionfish0/dp/blob/master/Paper%20results%20Houseprices.ipynb

Old version of citibike analysis (as this requires multiple runs and is quite large I've moved it to run on AWS)
https://github.com/lionfish0/dp/blob/master/Paper%20results%20Citibike.ipynb

## Useful notebooks

Originally I was going to use features from the census as additional inputs to the houseprice dataset, this allows access to the census API and also uses a geolocation database of postcodes.
https://github.com/lionfish0/dp/blob/master/Census%20code.ipynb

Earlier version tried to create the noise covariance matrix by considering various rotations etc between the values of $\mathbf{c}_i$, before we solved the lagrange system.
https://github.com/lionfish0/dp/blob/master/Rotate%20between%20two%20n-dimensional%20matrices.ipynb

Demo to self about how one can add noise to individual data points, making the noise hetroscedastic
https://github.com/lionfish0/dp/blob/master/WhiteHeteroscedastic%20experimentation.ipynb

Code demonstrating the method works
https://github.com/lionfish0/dp/blob/master/Testing_dp4gp.ipynb

## Exploring ideas/Junk

Exploring the citibike data a little
https://github.com/lionfish0/dp/blob/master/Citibike%20experimentation.ipynb

First look at the DP cloaking idea
https://github.com/lionfish0/dp/blob/master/Explaining%20Vector%20Alternative.ipynb

Working out how to calculate $\lambda_i$
https://github.com/lionfish0/dp/blob/master/Numerical%20Solution%20For%20Finding%20Langrange%20Multipliers-Corrected%20Gradient.ipynb

Earlier messing about with DP/GP/ideas (both full of wrongness)
https://github.com/lionfish0/dp/blob/master/Improved%20Bound%20Constraint%20differing%20across%20domain.ipynb
https://github.com/lionfish0/dp/blob/master/Fix%20to%20DP%20method.ipynb


House prices
https://github.com/lionfish0/dp/blob/master/House%20price%20example.ipynb




# Future Work

Future work will involve fitting the hyperparameters in a private way and more importantly in a way which maximises the accuracy of the predictions given the DP (the best lengthscale for a non-DP GP isn't the same as for one with DP noise added, as short lengthscales lead to increased DP noise, so the optimum with DP noise may be longer).
