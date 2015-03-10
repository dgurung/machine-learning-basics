#!/usr/bin/env python
import numpy as np
from sklearn.decomposition import PCA

def read_training_data(label0,label1):
	with open('../datasets/zip.train') as f:		
		trainX = []	
		trainY = []
		for line in f:
			line = line.split()
			if line:
				if int( float(line[0])) == label0 or int( float( line[0]) )== label1 :
					line = [float(i) for i in line]
					trainY.append(int(line[0]))
					trainX.append(line[1:])	

	return trainX,trainY

def read_testing_data(label0,label1):
	with open('../datasets/zip.test') as f:
		trainX = []
		trainY = []
		for line in f:
			line = line.split()
			if line:
				if int( float(line[0])) == label0 or int( float( line[0]) )== label1 :
					line = [float(i) for i in line]			
					trainY.append(int(line[0]))
					trainX.append(line[1:])

	return trainX, trainY

if __name__ == '__main__':
	label1 = 2;	label2 = 3;
	trainX, trainY = read_training_data(label1, label2)
	trainX, trainY = read_testing_data(label1, label2)
	print 'number of labels: ', len(set(trainY))
