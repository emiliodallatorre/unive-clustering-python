# file python per gaussian mixture model e rand index con vari valori di pca e di k

import time as time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# load the data
df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = control.idxmax(axis=1)

# standardize the data
X_std = StandardScaler().fit_transform(df)

number_of_pca = [3, 10, 15, 20, 25, 50, 256]  # sono 7
number_of_k = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # sono 10

fig, ax = plt.subplots(7, 10, figsize=(50, 40))
# do pca
for pca_n in number_of_pca:
    for k in number_of_k:
        pca = PCA(n_components=int(pca_n))
        X_pca = pca.fit_transform(X_std)
        # do gaussian mixture model and take the time
        start: float = time.time()
        GM: GaussianMixture = GaussianMixture(n_components=int(k),
                                              covariance_type='diag', random_state=1)
        GM.fit(X_pca)
        y_pred = GM.predict(X_pca)
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
        a, b, c, d = 0, 0, 0, 0
        for i in range(len(y_pred)):
            for j in range(i + 1, len(y_pred)):
                if y_pred[i] == y_pred[j] and control[i] == control[j]:
                    a += 1
                elif y_pred[i] == y_pred[j] and control[i] != control[j]:
                    b += 1
                elif y_pred[i] != y_pred[j] and control[i] == control[j]:
                    c += 1
                else:
                    d += 1
        rand_index = (a + d) / (a + b + c + d)
        ax[number_of_pca.index(pca_n)][number_of_k.index(k)].text(0.01, 0.01, 'RI:' + str(round(rand_index, 3)),
                                                                  fontsize=20,
                                                                  transform=ax[number_of_pca.index(pca_n)][
                                                                      number_of_k.index(k)].transAxes,
                                                                  verticalalignment='bottom',
                                                                  horizontalalignment='left',
                                                                  color='black',
                                                                  bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig("images/gmm.png")
plt.show()
