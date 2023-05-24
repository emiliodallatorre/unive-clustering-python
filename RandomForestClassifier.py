# i want to see if a random forest can do better

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# open a csv file but do not consider the last 10 columns
numbers = range(256)
df = pd.read_csv('data/semeion.csv', sep=' ', usecols=range(0, 256), names=numbers)

# this is the dataset that contains the right answer for each image in the dataset
control = pd.read_csv('data/semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
# i want to trasform control from a matrix of 0 and 1 to a vector of 0 to 9
control = np.array(control.idxmax(axis=1))

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(df, control, test_size=0.2, random_state=42)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# see the confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
plt.title('Confusion matrix \n Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
