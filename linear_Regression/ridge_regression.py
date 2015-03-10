#!/usr/bin/env python

import numpy as np
import itertools

import matplotlib.pyplot as plt 

import reading_prostate_data as read

# compute ridge regression model parameter \beta
def compute_Beta_ridge(X,y,lambda_):
	inv_ = np.linalg.pinv( np.dot(X.T, X) + np.identity(len(X[0])) * lambda_ ) 
	Xy = np.dot(X.T,y) 
	beta = np.dot(inv_ , Xy)
	beta = beta.flatten()	# make 1D

	return beta

# predict takes 1D array. flatten ndarrays
def predictY(beta,x):
	yhat = np.dot(x,beta)
	return yhat

# read data
trainX, trainY, testX, testY = read.read_data()

X = trainX
X = [ x_.extend([1]) for x_ in X]   # Add one to feature vector, compensate for constant 
X = np.array(trainX)                # N x (n+1) matrix

Y = np.array(trainY)[np.newaxis] # slice the array
Y = Y.T

RSS = []

lambda_ = np.linspace(0,1000,num=100)

for l in lambda_:
  # Ridge regression model
  beta = compute_Beta_ridge(X, Y, l)
  # Predict y for all points 
  yhat = np.array([  predictY( beta , x[np.newaxis].flatten() )  for x in X])
  err_ = np.subtract(trainY, yhat)    
  least_sq_err =  np.sum( np.square( err_) )
  RSS.append( least_sq_err )

plt.plot(lambda_,RSS,'ro')   
plt.xlabel(r'$ \lambda $')
plt.ylabel('Residual sums of square')
plt.show()  