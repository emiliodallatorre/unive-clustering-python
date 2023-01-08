# do the sam thing as MeanShiftModel.py but using spectral clustering instead of mean shift
# this is a good example of how to use spectral clustering
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from randind import rand_index_

# import dataset semeion.csv

df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256), names=range(256))
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = control.idxmax(axis=1)

# standardize the data
X_std = StandardScaler().fit_transform(df)

# doing pca

pca = PCA(n_components=10)
pca.fit(df)
df = pca.transform(df)
print(df.shape)

# do spectral clustering
# n_clusters = 10
# affinity = 'nearest_neighbors'
# n_neighbors = 10
# eigen_solver = 'arpack'

model = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=10, eigen_solver='arpack',
                           random_state=1)
model.fit(df)
labels = model.labels_

# want to calculate the rand index
ra: float = rand_index_(control, labels)
print(f'rand index = {ra}')
# i want to see the mean of the first cluster in a plot
import matplotlib.pyplot as plt

# plot the data in 3d
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[:, 0], df[:, 1], df[:, 2], c=labels, cmap='rainbow')
ax.grid(True, linestyle='-.', color='0.75')
plt.tight_layout()
plt.show()
