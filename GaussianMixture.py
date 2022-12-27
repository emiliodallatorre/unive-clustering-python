# this is a file for the Gaussian Mixture Model (GMM) algorithm to classify the images of the dataset into 10 classes (0 to 9)
# the GMM algorithm is a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions
# the GMM algorithm is a generative model
# the GMM algorithm is a soft clustering algorithm
# we assum thar the covariance matrix is diagonal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# open a csv file but do not consider the last 10 columns
numbers = range(256)
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=numbers)

# this is the dataset that contains the right answer for each image in the dataset
control = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
# i want to trasform control from a matrix of 0 and 1 to a vector of 0 to 9
# for example, if the 10-dimensional vector is [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], then the number is 9
# if the 10-dimensional vector is [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], then the number is 8
# if the 10-dimensional vector is [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], then the number is 7

control = np.array(control.idxmax(axis=1))

# i want to see if the control dataset is correct
# plt.imshow(df.iloc[1300].values.reshape(16, 16), cmap='gray')
# plt.title('This is a ' + str(control[1300]))
# plt.show()
# the control dataset is correct


X_train, X_test, y_train, y_test = train_test_split(df, control, test_size=0.2, random_state=42)

GBB = GaussianMixture(n_components=10, covariance_type='diag', max_iter=1000, random_state=0)

GBB.fit(X_train, y_train)

y_pred = GBB.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))

# i want to see the first imagin predicted by the GMM algorithm
# the first image is a 16x16 matrix

# plt.imshow(X_test.iloc[3].values.reshape(16, 16), cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.title('Predicted: {}'.format(y_pred[3]))
# plt.show()

# i want to see the confusion matrix using seaborn
'''
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
'''
# now to improve perfomance a wanto to use PCA to reduce the dimensionality of the dataset

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

GBB = GaussianMixture(n_components=10, covariance_type='diag', max_iter=1000, random_state=0)

GBB.fit(X_train_pca, y_train)

y_pred = GBB.predict(X_test_pca)

print('Accuracy: ', accuracy_score(y_test, y_pred))
# and see the confusion matrix again

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
