# Binary classification using logistic Regression

# Iteratively compute model coefficients, beta

# Two fixes required: 
# 1. Higher iteration number causes pinv to raise SVD did not converge error
# 2. Model paramter, beta, is initialized to 0 for iteration. Local minimum is possible.

import numpy as np
from sklearn.metrics import (confusion_matrix,accuracy_score )

import read_zip_data_2class as read

def iteration_paramters():
	MAXITER = 5 			# fix for larger iteration 
	eps = 10^-5

	return MAXITER, eps


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


def newton_raphson(trainX, Y, beta):
	""" Runs single iteration of newton_raphson and computes updated model coefficients beta
	Args:
		trainX : N x n+1 ndarray of x_i (sample) values
		Y : N x 1 ndarray of 0/1 coded labels for y_i, see ESL book section 4.4.1
			for 0/1 coding of labels 
		beta : Current coefficients of linear model
	Returns: 
			betaNew: an array of updated beta 
	"""

	# compute conditional probabilities	
	yhat = np.asarray([np.dot(trainX_ , beta) for trainX_ in trainX ])
	cond_prob_1 =  np.asarray([ np.exp(yhat_) / (1 + np.exp(yhat_) ) 
		for yhat_ in yhat ])
	p1 = np.ravel(cond_prob_1)

	# Compute the Weight matrix (diagonal)
	weight_elements = p1 * ( 1 - p1 )
	W = np.diag( weight_elements ) 

	trainXT = trainX.transpose()
	inv_ = np.linalg.pinv( np.dot( trainXT,  np.dot(W, trainX) ) )
	y_p = np.ravel(Y) - cond_prob_1

	# Update beta (model paramter) Newton Raphson method 
	update_ = np.dot( inv_, np.dot( trainXT, y_p ) )
	betaNew = beta + np.ravel(update_)  				# update beta
	
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
		MAXITER, eps = iteration_paramters()

		for iter in range(0, MAXITER):
			# Run one iteration of newton_raphson 
			betaNew = newton_raphson(trainX, Y, beta)

			# Test for convergence
			relativeChange = np.linalg.norm( betaNew - beta  ) 
			if relativeChange < np.linalg.norm(beta) * eps:
				print 'no convergence'
				break;

			beta = betaNew
			#print relativeChange

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
