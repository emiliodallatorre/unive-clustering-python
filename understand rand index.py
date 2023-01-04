# this is a file to open and read the data from the csv file and understand the data
# Semeion Handwritten Digit. (2008). UCI Machine Learning Repository.
# Retrieved from https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit

import pandas as pd

# read the data from the csv file
data = pd.read_csv('semeion.csv', sep=' ', usecols=range(256), names=range(256))
target = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
target = target.idxmax(axis=1)

# standardize the data using the standard scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# do pca
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(data)
data_pca = pca.transform(data)

# do mean shift
from sklearn.cluster import MeanShift

ms = MeanShift(bandwidth=4.3)
ms.fit(data_pca)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print('Number of clusters: ', len(cluster_centers))
print(*labels)
print(*target)

clusters: dict = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = [target[i]]
    else:
        clusters[label].append(target[i])

from randind import rand_index_

print(rand_index_(clusters))

# want to plot the mean shift
import matplotlib.pyplot as plt

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, s=40, cmap='tab20')
plt.show()
