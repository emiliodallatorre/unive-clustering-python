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

# i want to rename the labels after the mode of the label in every cluster
new_labels = []
for i in range(len(cluster_centers)):
    new_labels.append(target[labels == i].mode()[0])
print(*new_labels)

# pca inverse transform
data_pca_inverse = pca.inverse_transform(data_pca)
data = data_pca_inverse

# plot the mean of the image for every cluster
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 6, figsize=(10, 4))
for i in range(len(cluster_centers)):
    ax[i // 6, i % 6].imshow(data[labels == i].mean(axis=0).reshape(16, 16))

plt.tight_layout()
plt.show()

# do the rand index


# print(adjusted_rand_score(target, new_labels))

# i want to reneme the old labels to the new labels
for i in range(len(new_labels)):
    labels[labels == i] = new_labels[i]

print(*labels)

from sklearn.metrics import adjusted_rand_score

print(adjusted_rand_score(target, labels))
