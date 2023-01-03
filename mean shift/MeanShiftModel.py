# do clustering using mean shift and various kernels width
import time as time
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

# how to load semeion.csv file in the previous folder


# load data
df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256), names=range(256))
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(256, 266))
control = control.idxmax(axis=1)
pca_n: list = [2, 3, 4, 5, 6, 8, 10, 20]  # sono 8 righe
bandwidth: list = [1.0, 1.25, 1.5, 1.75, 2, 2.25]  # sono 6 colonne

# plot the data
fig, ax = plt.subplots(6, 8, figsize=(40, 30))

for i in range(8):
    for j in range(6):
        pca = PCA(n_components=pca_n[i])
        pca.fit(df)
        df_pca = pca.transform(df)
        # estimate bandwidth for mean shift
        start: float = time.time()
        ms = MeanShift(bandwidth=bandwidth[j], bin_seeding=True)
        ms.fit(df_pca)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            ax[j, i].plot(df_pca[my_members, 0], df_pca[my_members, 1], col + '.')
            ax[j, i].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                          markeredgecolor='k', markersize=14)
            ax[j, i].set_title('n_components = %d,\n bandwidth = %.3f' % (pca_n[i], bandwidth[j]), fontsize=20)
            ax[j, i].set_xticks(())
            ax[j, i].set_yticks(())
            ax[j, i].text(0.99, 0.01, f'time = {round(time.time() - start, 2)} s',  # tempo di esecuzione
                          transform=ax[j, i].transAxes, size=15,
                          horizontalalignment='right')
            ax[j, i].text(0.01, 0.01, f'clusters = {n_clusters_}',
                          transform=ax[j, i].transAxes, size=15,
                          horizontalalignment='left')

plt.tight_layout()
plt.savefig('images/MeanShiftParamAnalysis.png')
plt.show()

"""
# do pca
from sklearn.decomposition import PCA

pca = PCA(n_components=20)
pca.fit(df)
df = pca.transform(df)

# estimate bandwidth for mean shift
bandwidth = estimate_bandwidth(df, quantile=0.01, n_samples=1000)
print(bandwidth)

MS = MeanShift(bandwidth=bandwidth, bin_seeding=True)
MS.fit(df)
labels = MS.labels_
cluster_centers = MS.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
print("number of estimated clusters : %d" % len(cluster_centers))

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(df[my_members, 0], df[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.tight_layout()
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel([])
plt.ylabel([])
#plt.show()

# pca inverse transform
cluster_centers = pca.inverse_transform(cluster_centers)
# plot the cluster centers
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(cluster_centers[i].reshape(16, 16), cmap='gray')
    plt.title('cluster center %d' % i)
    plt.axis('off')
plt.tight_layout()
plt.show()


# see the mean of the first cluster in a plot 16x16
plt.imshow(cluster_centers[0].reshape(16, 16), cmap='gray')
#plt.show()
"""
