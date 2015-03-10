import numpy as np
from sklearn.metrics import (confusion_matrix,accuracy_score )

import reading_zip_data as read

# Compute mu and covariance of training set
def compute_model_parameter(trainX,trainY):
    labels = list(set(trainY))
    noLabels = len(labels)
    muX = []
    covX = []
    for i in range(0,noLabels):
        tempX = np.asarray( [valX_ for valX_, labelY_ in zip(trainX,trainY) if labelY_ == labels[i] ] )
        muX.append( np.mean(tempX,axis=0) )
        covX.append( np.cov(tempX.T,bias=1) )
    
    muX = np.asarray(muX)                                                                           
    covX = np.asarray(covX)
    
    return labels,muX, covX

# compute fisher vector using training set
def fisher_vector(muX,covX):
    cov_sum = np.add(covX[0],covX[1])
    inv_sum = np.linalg.pinv( np.add(covX[0],covX[1]) )
    mu_diff =  muX[0] - muX[1]
    u = np.dot(inv_sum, mu_diff)[np.newaxis]
    u = u / np.linalg.norm(u)                       # normalize
    return u

# Compute 1D Fisher Component
def projection(u, muX, covX):
    project_mu0 = np.dot(u,muX[0])
    project_mu1 = np.dot(u,muX[1])
    
    project_var0 = np.dot(u, np.dot(covX[0], u.T) )[0]
    project_var1 = np.dot(u, np.dot(covX[1], u.T) )[0]
    
    return project_mu0, project_mu1, project_var0, project_var1

# Find Gaussian pdf of each testing set
def pdf_gaussian(x, mu0, mu1, var0,var1):
    denom = (2*np.pi* float(var0))**.5
    num = np.exp(-(float(x)-float(mu0))**2/(2*var0))
    p0 = num/denom
    
    denom = (2*np.pi* float(var1))**.5
    num = np.exp(-(float(x)-float(mu1))**2/(2*var1))
    p1 = num/denom
    
    return p0, p1

def main():        
    labelTest = [[2,3] , [4,5], [7,9]]
    
    for label_ in labelTest:        
        trainX, trainY = read.read_training_data(label_[0],label_[1])
        testX, testY = read.read_testing_data(label_[0],label_[1])

        trainX = np.asarray(trainX)
        testX = np.asarray(testX)

        print 'Compared lables: ', label_[0], label_[1]
        print trainX.shape, testX.shape

        # Compute mean and Cov of each class for training set
        labels, muX, covX = compute_model_parameter(trainX,trainY)      
        # Compute fisher discriminant vector of unit length
        u = fisher_vector(muX,covX)                                     
        # project mean and Cov of each class to 1D fisher vector
        mu0_u, mu1_u, var0_u, var1_u = projection(u, muX,covX)          

        predictY = []
        for x in testX:
            # Project each test point along the fisher vector
            x_ = np.dot(u,x)                                             
            # compute the gaussian pdf of x for each class  
            px0, px1 = pdf_gaussian(x_, mu0_u, mu1_u, var0_u, var1_u)    
            # Gaussian Classifer 
            if px0 > px1:                                                
                predictY.append(labels[0])
            else:
                predictY.append(labels[1])

        print 'Confusion matrix: '
        print confusion_matrix(testY,predictY)
        print 'Accuracy', accuracy_score(testY,predictY)
        

if __name__ == '__main__':
	main()
