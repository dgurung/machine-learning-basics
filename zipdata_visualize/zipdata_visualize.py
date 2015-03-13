import numpy as np
from sklearn.manifold import TSNE

import reading_zip_data as read

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, label, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()    
    ax = axes(frameon=False)
    setp(ax, xticks=(), yticks=())
    scatter(X[:,0], X[:,1], c=label)
    if title is not None:
        plt.title(title)

def main():	        
	trainX, trainY = read.read_training_data()
	testX, testY = read.read_testing_data()

	trainX = np.asarray(trainX)
	testX = np.asarray(testX)

	tsne =  TSNE(n_components=2, random_state=0)
	X_tsne = tsne.fit_transform(trainX)
	plot_embedding(X_tsne, trainY)
	plt.show()

	X_tsne = tsne.fit_transform(testX)
	plot_embedding(X_tsne, testY)
	plt.show()


if __name__ == '__main__':
	main()
