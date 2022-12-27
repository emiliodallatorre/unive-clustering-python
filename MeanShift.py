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

bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(df)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print("number of estimated clusters : %d" % len(cluster_centers))