# do the sam thing as MeanShiftModel.py but using spectral clustering instead of mean shift
# this is a good example of how to use spectral clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs

# generate some random data
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# plot the data
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# do the spectral clustering
model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='discretize')
model.fit(X)

# plot the results
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()
