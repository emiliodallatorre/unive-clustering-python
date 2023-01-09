# do clustering using mean shift and various kernels width
import time as time
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

from randind import rand_index_

# load data
df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256), names=range(256))
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(256, 266))
control = control.idxmax(axis=1)

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
        end: float = time.time()
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        # print("number of estimated clusters : %d" % n_clusters_)
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
            ax[j, i].text(0.99, 0.01, f't = {round(end - start, 2)} s',  # tempo di esecuzione
                          transform=ax[j, i].transAxes, size=12,
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          color='black',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax[j, i].text(0.01, 0.01, f'clusters = {n_clusters_}',
                          transform=ax[j, i].transAxes, size=12,
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          color='black',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            # calcolo rand index
            rand_indexx = rand_index_(control, labels)
            # want to add the rand index to every subplot in corner top right inside the plot
            ax[j, i].text(0.99, 0.99, f'rand index = {round(rand_indexx, 2)}',
                          transform=ax[j, i].transAxes, size=12,
                          horizontalalignment='right',
                          verticalalignment='top',
                          color='black',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('../images/MeanShiftParamAnalysis.png')
plt.show()
