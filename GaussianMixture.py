from math import comb
from statistics import mode

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# open a csv file but do not consider the last 10 columns
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))

# the last 10 columns are the labels of the digits where 1 means the digit is the number of the column and 0 means it
# is not
control = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = control.idxmax(axis=1)
# add a column to the dataframe that contains the right answer
# df['right_answer'] = control


X_std = StandardScaler().fit_transform(df)

# do pca
pca = PCA(2)
X_pca = pca.fit_transform(X_std)
print(pca.n_components_)

# from the result, we can see that n_components = 10 is the best choice
GM: GaussianMixture = GaussianMixture(n_components=20, covariance_type='diag', random_state=1)
GM.fit(X_pca)
y_pred = GM.predict(X_pca)
print(list(set(y_pred)))
c0 = df[y_pred == 0]
c1 = df[y_pred == 1]
c2 = df[y_pred == 2]
c3 = df[y_pred == 3]
c4 = df[y_pred == 4]
c5 = df[y_pred == 5]
c6 = df[y_pred == 6]
c7 = df[y_pred == 7]
c8 = df[y_pred == 8]
c9 = df[y_pred == 9]
c10 = df[y_pred == 10]
c11 = df[y_pred == 11]
c12 = df[y_pred == 12]
c13 = df[y_pred == 13]
c14 = df[y_pred == 14]
c15 = df[y_pred == 15]
c16 = df[y_pred == 16]
c17 = df[y_pred == 17]
c18 = df[y_pred == 18]
c19 = df[y_pred == 19]

print(c0.shape, c1.shape, c2.shape, c3.shape,
      c4.shape, c5.shape, c6.shape, c7.shape,
      c8.shape, c9.shape, c10.shape, c11.shape,
      c12.shape, c13.shape, c14.shape, c15.shape,
      c16.shape, c17.shape, c18.shape, c19.shape)

# plot the result
fig, ax = plt.subplots(4, 5, figsize=(5, 4))
for i in range(20):
    ax[i // 5, i % 5].imshow(eval('c' + str(i)).mean(axis=0).values.reshape(16, 16), cmap='Blues')
    ax[i // 5, i % 5].set_title('cluster n:' + str(i))
    ax[i // 5, i % 5].axis('off')
plt.tight_layout()
# plt.savefig('GaussianMixture_with 20 cluster.png')
plt.show()

# take an image from the first cluster and extraxt his index in the dataframe
index = c12.index[8]
# plot the image
plt.imshow(df.loc[index].values.reshape(16, 16), cmap='Blues')
plt.axis('off')
plt.title('right answer: ' + str(control[index]))
plt.tight_layout()
# plt.show()

# for every cluster I want to know the number of the digit that is the most present cecking the control dataframe
mode_cluster: pd.DataFrame = pd.DataFrame(columns=['cluster', 'mode'])
for i in range(20):
    mode_cluster = pd.concat([mode_cluster, pd.DataFrame([[i, mode(control[eval('c' + str(i)).index])]],
                                                         columns=['cluster', 'mode'])])
    print('cluster n:', i, 'right answer:', mode(control[eval('c' + str(i)).index]))
    # work as I expected :)

# now i want the rand index of the cluster and need the binomial coefficient for the number of pairs of elements in
# the cluster I need to count the number of pairs of elements in the cluster and the number of pairs of elements in
# the cluster that have the same label

# for every cluster I want to know the number of the digit that is the most present cecking the control dataframe
rand_index: pd.DataFrame = pd.DataFrame(columns=['cluster', 'rand_index'])

for i in range(20):
    # count the number of pairs of elements in the cluster
    n = comb(eval('c' + str(i)).shape[0], 2)
    # count the number of pairs of elements in the cluster that have the same label
    a: int = 0
    for j in range(eval('c' + str(i)).shape[0]):
        for k in range(j + 1, eval('c' + str(i)).shape[0]):
            if control[eval('c' + str(i)).index[j]] == control[eval('c' + str(i)).index[k]]:
                a += 1
    # count the number of pairs of elements in the cluster that have different label
    b = n - a
    # count the number of pairs of elements in the cluster that have the same label
    c: int = 0
    for j in range(eval('c' + str(i)).shape[0]):
        for k in range(j + 1, eval('c' + str(i)).shape[0]):
            if control[eval('c' + str(i)).index[j]] != control[eval('c' + str(i)).index[k]]:
                c += 1
    # count the number of pairs of elements in the cluster that have different label
    d = n - c
    # calculate the rand index
    rand_index = pd.concat([rand_index, pd.DataFrame([[i, (a + d) / (a + b + c + d)]],
                                                     columns=['cluster', 'rand_index'])])
    print('cluster n:', i, 'rand index:', (a + d) / (a + b + c + d))

# i want only one rand index for the gaussian mixture
print('rand index for the gaussian mixture:', rand_index['rand_index'].mean())
# not like this :(


# For each value of k (or kernel width) provide the value of the Rand index:
# R=2(a+b)/(n(n-1))
# where
# • n is the number of images in the dataset.
# • a is the number of pairs of images that represent the same digit and that are clustered together.
# • b is the number of pairs of images that represent different digits and that are placed in different clusters.


# **********************************************************************************************************************
# I need to count the number of pairs of elements in the cluster and the number of pairs of elements in
# the cluster that have the same label
import time as time

start: float = time.time()
n = comb(df.shape[0], 2)

A: int = 0
B: int = 0

a = 0
for j in range(df.shape[0]):  # for every image
    for k in range(j + 1, df.shape[0]):  # for every image after the image j
        if control[df.index[j]] == mode_cluster[mode_cluster['cluster'] == y_pred[j]]['mode'].values[0]:
            a += 1
    A += a
# count the number of pairs of elements in the cluster that have different label
b = n - a
# count the number of pairs of elements in the cluster that have the same label
c = 0
for j in range(df.shape[0]):  # for every image
    for k in range(j + 1, df.shape[0]):  # for every image after the image j
        if control[df.index[j]] != mode_cluster[mode_cluster['cluster'] == y_pred[j]]['mode'].values[0]:
            c += 1
    B += c
# count the number of pairs of elements in the cluster that have different label
d = n - c
# calculate the rand index
print('rand index for the gaussian mixture:', (a + d) / (a + b + c + d), 'time:', time.time() - start)
print('rand index 2 for the gaussian mixture:', 2 * (b + d) / (1593 * 1592), 'time:', time.time() - start)
#**********************************************************************************************************************
