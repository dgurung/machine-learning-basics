# Read (x,y) data from prostate dataset. (Both training and testing data set)

def read_data():
	print 'Reading data ...'

	with open('../datasets/prostate.data') as f:		# Read lines [1]
		trainX = []; trainY = []
		testX = []; testY = []
		for line in f:
			line = line.split()
			if line and len(line) == 11:				
				X = [float(i) for i in line[1:9] ]					
				Y = float(line[9])
				if line[10] == 'T':					
					trainX.append(X)
					trainY.append(Y)
				else :					
					testX.append(X)
					testY.append(Y)
	print 'Training data: ', len(trainX)	
	print 'Testing data: ', len(testX)	

	return trainX, trainY, testX, testY		

if __name__ == '__main__':
	dim_Reduced = 10
	read_data()
