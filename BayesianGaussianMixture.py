import time as time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))

X_std = StandardScaler().fit_transform(df)

n_pca = [2, 4, 8, 16, 32, 40, 64, 90, 128, 256]
fig, ax = plt.subplots(10, 10, figsize=(35, 30))
for i in range(10):
    for j in range(10):
        pca = PCA(n_components=n_pca[i])
        X_pca = pca.fit_transform(X_std)
        start = time.time()
        bgm = BayesianGaussianMixture(n_components=n_pca[j], covariance_type='diag', max_iter=1000, random_state=0)
        bgm.fit(X_pca)
        ax[i][j].scatter(X_pca[:, 0], X_pca[:, 1], c=bgm.predict(X_pca), s=40, cmap='nipy_spectral')
        ax[i][j].set_title(
            'n_pca = ' + str(n_pca[i]) + ' n_components = ' + str(n_pca[j]) +
            '\n time = ' + str(time.time() - start) + '\n' + 'score = ' + str(bgm.score(X_pca)))
plt.tight_layout()
plt.savefig('gaussianPCA.png')
plt.show()
