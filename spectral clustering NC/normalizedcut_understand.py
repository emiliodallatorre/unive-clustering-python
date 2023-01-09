# do the sam thing as MeanShiftModel.py but using spectral clustering instead of mean shift
# this is a good example of how to use spectral clustering
import time as time

import matplotlib.pyplot as plt
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

number_of_pca: list = [2, 3, 4, 5, 6, 8, 10, 20]  # sono 8 righe
number_of_k: list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # sono 10 componenti

fig, ax = plt.subplots(8, 10, figsize=(60, 50))
# do pca
for pca_n in number_of_pca:
    for k in number_of_k:
        pca = PCA(n_components=int(pca_n))
        X_pca = pca.fit_transform(X_std)
        # do gaussian mixture model and take the time
        start: float = time.time()
        SC: SpectralClustering = SpectralClustering(n_clusters=int(k), random_state=1)
        SC.fit(X_pca)
        y_pred = SC.labels_
        # plot the result
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, s=40,
                                                                     cmap='tab20')
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].set_title('pca: ' + str(pca_n) + '\n k: ' + str(k),
                                                                       fontsize=30)
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].set_xticks([])
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].set_yticks([])
        # want to insert a text in every subplot in bottomright corner
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].text(0.99, 0.01,
                                                                  't:' + str(round(time.time() - start, 3)),
                                                                  fontsize=20,
                                                                  transform=ax[number_of_pca.index(pca_n)][
                                                                      number_of_k.index(k)].transAxes,
                                                                  verticalalignment='bottom',
                                                                  horizontalalignment='right',
                                                                  color='black',
                                                                  bbox=dict(facecolor='white', alpha=0.5))
        # calculate rand index

        rand_inde = rand_index_(control, y_pred)
        # add rand index in corner botton left of every subplot inside the figure
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].text(0.01, 0.01, 'ri:' + str(round(rand_inde, 3)),
                                                                  fontsize=20,
                                                                  transform=ax[number_of_pca.index(pca_n)][
                                                                      number_of_k.index(k)].transAxes,
                                                                  verticalalignment='bottom',
                                                                  horizontalalignment='left',
                                                                  color='black',
                                                                  bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()
