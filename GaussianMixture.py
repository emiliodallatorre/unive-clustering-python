import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# open a csv file but do not consider the last 10 columns
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))


# from the result, we can see that n_components = 10 is the best choice
GM: GaussianMixture = GaussianMixture(n_components=10, covariance_type='diag', random_state=0)
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
print(c0.shape, c1.shape, c2.shape, c3.shape, c4.shape, c5.shape, c6.shape, c7.shape, c8.shape, c9.shape)

# plot the result
fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax[i // 5, i % 5].imshow(eval('c' + str(i)).mean(axis=0).values.reshape(16, 16), cmap='gray')
    ax[i // 5, i % 5].set_title('cluster ' + str(i))
    ax[i // 5, i % 5].axis('off')
plt.tight_layout()
plt.show()



