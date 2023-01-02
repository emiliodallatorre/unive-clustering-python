import time as time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))

# the last 10 columns are the labels of the digits where 1 means the digit is the number of the column and 0 means it is not
control = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = control.idxmax(axis=1)
# add a column to the dataframe that contains the right answer
df['right_answer'] = control

X_std = StandardScaler().fit_transform(df[df.columns[:255]])

n_pca = [2, 4, 8, 16, 32, 40, 64, 90, 128, 256]
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for j in range(10):
    pca = PCA(n_components=14)
    X_pca = pca.fit_transform(X_std)
    ax[j // 5, j % 5].scatter(X_pca[:, 0], X_pca[:, 1], c=df['right_answer'], cmap='viridis')
    ax[j // 5, j % 5].set_title('n_components = {}'.format(n_pca[j]))
    ax[j // 5, j % 5].set_xlabel('n_components')
    ax[j // 5, j % 5].axis('off')

plt.tight_layout()
# plt.savefig('gaussianPCA.png')
plt.show()
