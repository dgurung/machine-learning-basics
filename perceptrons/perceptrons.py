# Binary classification using logistic Regression

# Compute weights of network using gradient descent

import numpy as np
from sklearn.metrics import (confusion_matrix,accuracy_score )

import read_zip_data_2class as read

def iteration_paramters():
	MAXITER = 50			# fix for larger iteration 
	eps = 10^-5
	learningRate = 0.01

	return MAXITER, eps, learningRate


def compute_Beta(X,y):
	""" Computes coefficients for linear model: yhat = X * beta
	Args: 
		X : N x n+1 ndarray of x_i (sample) values
		y : N x 1 ndarray of y_i (labels of samples)

	Returns:
		beta : coefficients of linear model   
	"""
	inv_ = np.linalg.pinv( np.dot(X.T, X) ) 
	Xy = np.dot(X.T,y) 
	beta = np.dot(inv_ , Xy)

	return beta

def  compute_gradient(soft_y, Y, trainX ):
	""" Computes gradient of sum of squared errors
	Args:
		soft_y : N x 1 ndarray of conditional probabilities or simply squashing function
		Y : N x 1 ndarray of 0/1 coded labels for y_i, see ESL book section 4.4.1
		trainX : N x n+1 ndarray of x_i (sample) values		

	Returns:
		E_gradient : N x 1 ndarray of gradient of sum of squared errors 
	"""
	
	factor = ( soft_y - Y ) *  soft_y * ( 1 - soft_y) 			
	
	E_gradient = [] 	# list of gradient of error

	for j in range(0, trainX.shape[1]):
		x_j = np.asarray([ x_[j]  for x_ in trainX ]) 				# extract jth element from each trainX
		E_j_i = factor * x_j
		E_gradient.append(np.sum(E_j_i))

	
	return np.asarray(E_gradient)


def gradient_descent(trainX, Y, beta, learningRate):
	""" Computes one iteration of gradient descent and updates beta 
	Args:
		trainX : N x n+1 ndarray of x_i (sample) values
		Y : N x 1 ndarray of 0/1 coded labels for y_i, see ESL book section 4.4.1
		beta : coefficients of linear model
		learningRate : learning rate 

	Returns:
		newBeta : updated coefficients of linear model

	"""

	y_hat = np.asarray([np.dot(trainX_ , beta) for trainX_ in trainX ])
	soft_y =  np.asarray([ np.exp(yhat_) / (1 + np.exp(yhat_) ) for yhat_ in y_hat ])	
	soft_y = np.ravel(soft_y)

	Y = np.ravel(Y)

	E_gradient = compute_gradient(soft_y, Y, trainX)

	betaNew = beta - learningRate * E_gradient 

	return betaNew		
		
def compute_error_metrics(X, Y, beta , label_):
	""" Computes confusion matrix and accuracy score of binary classifier
	Args:
		X : N x n+1 ndarray of x_i values
		Y : N x1 ndarray of 0/1 coded labels for y_i
		beta : coefficients of linear model
		label_ : labels for two classes

	Returns:
		confusion_matrix : confusion matrix for two classes 
		accuracy_score : accuracy score for two classes
	"""
	# Calculate predicted probabilities
	yhat = np.asarray([np.dot(X_ , beta) for X_ in X ])
	p = np.asarray([ np.exp(yhat_) / (1 + np.exp(yhat_) ) for yhat_ in yhat ])

	# Predict labels based on probabilities. For binary classifier probabilities 
	# for each sample sum up to 1. That is,  p(label_[0]) + p(label_[1]) = 1
	p[p>=0.5] = label_[0]
	p[p<0.5] = label_[1]

	p = p.astype(int)

	return confusion_matrix(Y,list(p)), accuracy_score(Y,list(p))


def main():

	# Array of pair of labels (or decimal digits) that are to be 
	# classified using binary logistic regression
	labelTest = [[2,3] , [4,5], [7,9]]


	# perform classification for each pair of labels
	for label_ in labelTest:

		print 'Classifying labels: ', label_

		# Read list of training samples
		trainX, trainY = read.read_training_data(label_[0],label_[1])		
		
		# Create a tall ndarray and augment column vector of value 1
		# This augmentation of 1 is for including bias in the linear model
		trainX = np.asarray(trainX)								# N x n ndarray
		trainX = np.c_[np.ones(trainX.shape[0]), trainX]		# N x (n+1) ndarray

		temptrainY = trainY
		trainY = np.array(trainY)[np.newaxis] 	# slice the array
		trainY = trainY.T 						# N x 1 vector

		# Code the training labels y_i to 0/1
		Y = trainY
		Y[ Y == label_[0] ] = 1
		Y[ Y == label_[1] ] = 0

		# Initialize model
		beta = np.zeros(trainX.shape[1])  	# Initial value taken from ESL book	section 4.4.1

		# Get iteration parameters
		MAXITER, eps, learningRate = iteration_paramters()		

		for iter in range(0, MAXITER):

			betaNew = gradient_descent(trainX,Y,beta, learningRate)
			beta = betaNew
	
		# Compute confusion matirx for training data
		cm, accuracy = compute_error_metrics(trainX, temptrainY, beta, label_)

		print cm
		print 'Accuracy', accuracy

		# Compute Confusion matrix for testing data
		# Read testing data
		testX, testY = read.read_testing_data(label_[0],label_[1])
		testX = np.asarray(testX)								# N x n matrix
		testX = np.c_[np.ones(testX.shape[0]), testX]			# N x (n+1) matrix

		cm, accuracy = compute_error_metrics(testX, testY, beta, label_)

		print cm
		print 'Accuracy', accuracy

if __name__ == '__main__':
	main()
