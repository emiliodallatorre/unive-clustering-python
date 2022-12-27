import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split

# open a csv file but do not consider the last 10 columns
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))


# from the result, we can see that n_components = 10 is the best choice
GM: GaussianMixture = GaussianMixture(n_components=20, covariance_type='diag', random_state=0)
GM.fit(df)
y_pred = GM.predict(df)
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
# c20 = df[y_pred == 20]

print(c0.shape, c1.shape, c2.shape, c3.shape, c4.shape, c5.shape, c6.shape, c7.shape, c8.shape, c9.shape)

# plot the result
fig, ax = plt.subplots(4, 5, figsize=(5, 4))
for i in range(20):
    ax[i // 5, i % 5].imshow(eval('c' + str(i)).mean(axis=0).values.reshape(16, 16), cmap='gray')
    ax[i // 5, i % 5].set_title('cluster ' + str(i))
    ax[i // 5, i % 5].axis('off')
plt.tight_layout()
plt.savefig('GaussianMixture_with 20 cluster.png')
plt.show()

# now i want to do the same thing but using PCA

