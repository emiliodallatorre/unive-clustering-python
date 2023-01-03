# want to see the mean of the cluster centers in a plot 16x16
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

# load data
df = df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256), names=range(256))
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = np.array(control.idxmax(axis=1))

pca = PCA(n_components=5)
pca.fit(df)
df_pca = pca.transform(df)

ms = MeanShift(bandwidth=2, bin_seeding=True)
ms.fit(df_pca)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# do the pca inverse transform
cluster_centers = pca.inverse_transform(cluster_centers)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
fig, ax = plt.subplots(2, 6, figsize=(6, 2))
# in every subplot plot the mean of the image of the cluster center
for i in range(11):
    ax[i // 6, i % 6].imshow(cluster_centers[i].reshape(16, 16), cmap='gray')
    ax[i // 6, i % 6].set_xticks(())
    ax[i // 6, i % 6].set_yticks(())

# dont show the last subplot
ax[1, 5].set_visible(False)

# plt.subplots_adjust(wspace=0.9, hspace=0.4, left=0.1, right=0.4, top=0.9, bottom=0.1)


plt.tight_layout()
# plt.savefig('images/MeanShiftClusterCenters.png')
plt.show()
from sklearn.metrics.cluster import adjusted_rand_score

rand_index = adjusted_rand_score(control, labels)
print(*labels)

import numpy as np
from scipy.special import comb


# define Rand index function
def rand_index(actual, pred):
    tp_plus_fp = comb(np.bincount(actual), 2).sum()
    tp_plus_fn = comb(np.bincount(pred), 2).sum()
    A = np.c_[(actual, pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(actual))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


print(rand_index(control, labels))

# calculate Rand index
# rand_index([1, 1, 1, 2, 2], [1, 1, 2, 2, 3])
#
# 0.6


print(rand_index)
