import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys, os

if __name__ == '__main__':
	argv = sys.argv

	train_data = np.load(argv[1]) / 255
	train_data = np.reshape(train_data, (len(train_data), -1))
	pca = PCA(n_components=280, whiten=True, svd_solver='randomized')
	pca_features = pca.fit_transform(train_data)

	# kmeans
	kmeans = KMeans(init='k-means++', n_clusters=2, random_state=0).fit(pca_features)

	f = pd.read_csv(argv[2])
	IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
    #
	o = open(argv[3], 'w')
	o.write('ID,Ans\n')
    #
	for idx, i1, i2 in zip(IDs, idx1, idx2):
		p1 = kmeans.labels_[i1]
		p2 = kmeans.labels_[i2]
		if p1 == p2:
			pred = 1
		else:
			pred = 0
		o.write('{},{}\n'.format(idx,pred))
	o.close()
