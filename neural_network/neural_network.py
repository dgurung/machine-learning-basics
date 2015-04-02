import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix,accuracy_score )

import read_zip_data as read

class NeuralNetwork_1H(object):
    def __init__(self, learning_rate=0.1, n_hidden_units=40,maxiter=200):
        # intialize neightwork parameters
        self.gamma = learning_rate
        self.n_hidden_units = n_hidden_units
        self.maxiter = maxiter
    
    def rand_init_weights(self):
        # Initialize weights randomly
        # this weights also includes weight associated with bias
        self.weights_1 = np.random.normal(0,0.001,(self.n_input_units, self.n_hidden_units))
        self.weights_2 = np.random.normal(0,0.001,(self.n_hidden_units + 1, self.n_output_units))   
        
    def feed_forward(self,X):  
        
        # hidden layer
        z1 = np.dot(X,self.weights_1)
        o1 = self.sigmoid(z1)
        
        o1_biased = np.hstack((np.ones(1),o1))    # append 1 to introduce the bias term
        
        # output layer
        z2 = np.dot(o1_biased, self.weights_2)
        o2 = self.sigmoid(z2)

        return o1, o1_biased, o2
                
    def back_propagation(self, o1, o2, y):
        
        # Error in the output layer
        err = o2 - y                  
        
        # The derivative (dE/dw) in feed-forward step at the output layer
        op_prime_2 = o2 * (1 - o2)    
        D2 = np.diag(op_prime_2)
        
        # The derivative (dE/dw) in feed-forward step at the hidden layer
        op_prime_1 = o1 * (1 - o1)    
        D1 = np.diag(op_prime_1)
        
        # The back propagated error up to the output unit
        delta_2 = np.dot(D2 , err)
        # The back propagated error up to the hidden layer
        delta_1 = np.dot(D1, np.dot(self.weights_2[:-1,:], delta_2 ) )
        
        return delta_2, delta_1
    
    def weight_update(self, o1_biased, delta_2, delta_1, x ):
        # The correction for weight matrix joining hidden and output layers
        delta_weightsT_2 = - self.gamma * np.dot(delta_2[np.newaxis].T, o1_biased[np.newaxis])
        # The correction for weight matrix joining input and hidden layers
        delta_weightsT_1 = - self.gamma * np.dot(delta_1[np.newaxis].T, x[np.newaxis])
        
        # Corrected weight
        self.weights_2 = self.weights_2 + delta_weightsT_2.T
        self.weights_1 = self.weights_1 + delta_weightsT_1.T    
        
    def train(self,X,y):                              
        # Train the network by randomly shuffling all the input data.
        # Do above step for multiple iteration (upto maxiter)
        for i in range(self.maxiter):
            # Randomly shufffle the input data in each iteration
            index=list(range(self.num_samples))
            np.random.shuffle(index)        
            
            # Train the network (weights) for all data
            for row in index:
                # Compute the output error 
                o1, o1_biased, o2 = self.feed_forward(X[row])
                # Compute the back propagated error
                delta_2, delta_1 = self.back_propagation(o1,o2,y[row])
                # Correct the weight matrix
                self.weight_update(o1_biased, delta_2, delta_1, X[row])

        
    def fit(self, X, y):
        self.n_input_units = X.shape[1]
        self.n_output_units = 10
        self.num_samples = X.shape[0]          
                
        # initialize weights
        self.rand_init_weights()        
        # train the nerual network
        self.train(X,y)    
    
    def predict(self,X):
        n_samples = X.shape[0]
        pred = np.zeros((n_samples,self.n_output_units))
        # Predict the output using the trained network
        for i in range(n_samples):
            _,_,pred[i,:] = self.feed_forward(X[i])        
        return pred
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


def preprocess_data():
    origtrainX, trainY = read.read_training_data()
    testX, testY = read.read_testing_data()

    # Reduce dimensionality of input X data
    pca = PCA(n_components=20)
    pca.fit_transform(origtrainX)
    trainX = pca.transform(origtrainX)

    testX = pca.transform(testX)

    # list to array
    trainX = np.asarray(trainX)             # N x n ndarray
    Xmean = np.mean(trainX, axis=0)    
    trainX = trainX - Xmean   
    Xstd = np.std(trainX,axis=0)
    trainX = trainX /  Xstd 
    trainX = np.c_[np.ones(trainX.shape[0]), trainX]    # N x (1+n) ndarray      


    # Binary coding 10 labels
    trainY_ = np.zeros((trainX.shape[0],10))
    for i,j in enumerate(trainY):
        trainY_[i,j] = 1

    # read testX
    testX = np.asarray(testX)                                  # N x n matrix
    testX = np.c_[np.ones(testX.shape[0]), testX]             # N x (n+1) matrix

    print 'train: ', trainX.shape, 'test: ', testX.shape 

    return trainX,trainY_, testX, testY

def main():
    # read data that has been preprocessed using pca
    trainX,trainY_, testX, testY = preprocess_data()

    # create an instance of NN
    nn = NeuralNetwork_1H()
    # train the NN 
    nn.fit(trainX,trainY_)
    # predict using NN
    prob = nn.predict(testX)

    predY = [np.argmax(p,axis=0) for p in prob]
    # compute comparision matrix
    cm = confusion_matrix(testY,predY)
    ac = accuracy_score(testY,predY)
    print cm, ac


if __name__ == '__main__':
    main()
