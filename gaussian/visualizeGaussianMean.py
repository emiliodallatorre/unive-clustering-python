import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from randind import rand_index_

# open a csv file but do not consider the last 10 columns
df = pd.read_csv('../semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))

# the last 10 columns are the labels of the digits where 1 means the digit is the number of the column and 0 means it
# is not
control = pd.read_csv('../semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
target = control.idxmax(axis=1)
# add a column to the dataframe that contains the right answer
# df['right_answer'] = control


X_std = StandardScaler().fit_transform(df)

# do pca
pca = PCA(n_components=25)
X_pca = pca.fit_transform(X_std)
print(pca.n_components_)

# from the result, we can see that n_components = 10 is the best choice
GM: GaussianMixture = GaussianMixture(n_components=15, covariance_type='diag', random_state=1)
GM.fit(X_pca)
y_pred = GM.predict(X_pca)

# calculate the rand index
print(rand_index_(target, y_pred))

# pca reconstruction
X_reconstructed = pca.inverse_transform(X_pca)
# i want to plot the mean of each gaussian
fig, ax = plt.subplots(3, 5, figsize=(15, 11))
for i in range(15):
    ax[i // 5, i % 5].imshow(X_reconstructed[y_pred == i].mean(axis=0).reshape(16, 16), cmap='Blues')
    ax[i // 5, i % 5].set_title('pca: ' + str(pca.n_components_) + '\ncluster n: ' + str(i), fontsize=15)
    ax[i // 5, i % 5].axis('off')
plt.tight_layout()
plt.show()

"""
# plot the result
fig, ax = plt.subplots(3, 5, figsize=(15, 11))
for i in range(15):
    ax[i // 5, i % 5].imshow(df[y_pred == i].mean(axis=0).values.reshape(16, 16), cmap='Blues')
    ax[i // 5, i % 5].set_title('pca: ' + str(pca.n_components_) + '\ncluster n: ' + str(i), fontsize=15)
    ax[i // 5, i % 5].axis('off')
plt.tight_layout()
plt.savefig('../images/gaussianpca25cluster15.png')
plt.show()
"""
