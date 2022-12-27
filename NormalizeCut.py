# the same clustering but using normlized cut

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering

# open a csv file but do not consider the last 10 columns
numbers = range(256)
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=numbers)

# this is the dataset that contains the right answer for each image in the dataset
control = pd.read_csv('semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
# i want to trasform control from a matrix of 0 and 1 to a vector of 0 to 9
control = np.array(control.idxmax(axis=1))

# i want to see if the control dataset is correct
# plt.imshow(df.iloc[1300].values.reshape(16, 16), cmap='gray')
# plt.title('This is a ' + str(control[1300]))
# plt.show()
# the control dataset is correct

X_train, X_test, y_train, y_test = train_test_split(df, control, test_size=0.2, random_state=42)

normalized_cut = SpectralClustering(n_clusters=10, affinity='nearest_neighbors', random_state=0)

normalized_cut.fit(X_train, y_train)

y_pred = normalized_cut.fit_predict(X_test)

# see the confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette("rocket", as_cmap=True))
plt.title('Confusion matrix \n Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

