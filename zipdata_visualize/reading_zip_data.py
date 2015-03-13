import numpy as np

def read_training_data():
	with open('../datasets/zip.train') as f:		
		trainX = []	
		trainY = []
		for line in f:
			line = line.split()
			if line:
				line = [float(i) for i in line]
				trainY.append(int(line[0]))
				trainX.append(line[1:])	

	return trainX,trainY

def read_testing_data():
	with open('../datasets/zip.test') as f:
		trainX = []
		trainY = []
		for line in f:
			line = line.split()
			if line:
				line = [float(i) for i in line]			
				trainY.append(int(line[0]))
				trainX.append(line[1:])

	return trainX, trainY

if __name__ == '__main__':
	trainX, trainY = read_training_data()
	trainX, trainY = read_testing_data()
	print 'number of labels: ', len(set(trainY))
