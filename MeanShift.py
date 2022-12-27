# this is a file for the mean shift algorithm for the semeion handwritten digit dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# load the data
numbers = range(256)
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=numbers)
# print(df.shape)

# this is the dataset that contains the right answer for each image in the dataset
control = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
# print(control.head())

control = np.array(control.idxmax(axis=1))
print(control)

X_train, X_test, y_train, y_test = train_test_split(df, control, test_size=0.5, random_state=7)

mean_shift = MeanShift(bandwidth=10, bin_seeding=True)
mean_shift.fit(X_train)
y_pred = mean_shift.predict(X_test)

# see the confusion matrix using seaborn

mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=range(10),
            yticklabels=range(10))
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.show()
