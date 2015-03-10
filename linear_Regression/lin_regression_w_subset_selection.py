
import numpy as np
import itertools

import matplotlib.pyplot as plt 

import reading_prostate_data as read


# model: yhat = X * beta
# compute rigression coefficients
def compute_Beta(X,y):
	inv_ = np.linalg.pinv( np.dot(X.T, X) ) 
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

# compute RSS 
RSS = np.zeros(8)

# Subset selection for linear regression models
features = [0, 1, 2, 3, 4, 5, 6, 7]

RSS = []
noFeatures = []

for L in range(0, len(features)+1):
  for subset in itertools.combinations(features, L):
        lenSub = len(subset) ; lenFeat = len(features)
        if lenSub <= lenFeat:
            Xsubset = np.delete(X,subset, axis=1)    # delete column: col of X
            # Linear regression model
            beta = compute_Beta(Xsubset,Y)
            # Predict y for all points 
            yhat = np.array([  predictY( beta , x[np.newaxis].flatten() )  for x in Xsubset])
            err_ = np.subtract(trainY, yhat)  
            least_sq_err = np.sum( np.square( err_) )
            RSS.append( least_sq_err )
            noFeatures.append( lenFeat - lenSub )
# plot
plt.plot(noFeatures,RSS,'ro') 
plt.axis([-1,10,0,100])
plt.xlabel('# features')
plt.ylabel('Residual sum of squares')
plt.show()
