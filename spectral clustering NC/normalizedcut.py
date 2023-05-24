# do the sam thing as MeanShiftModel.py but using spectral clustering instead of mean shift
# this is a good example of how to use spectral clustering
import time as time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from randind import rand_index_

# import dataset semeion.csv

df = pd.read_csv('../data/semeion.csv', sep=' ', usecols=range(256), names=range(256))
control = pd.read_csv('../data/semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control = control.idxmax(axis=1)

# standardize the data
X_std = StandardScaler().fit_transform(df)

number_of_pca: list = [2, 3, 4, 5, 6]  # sono 5 righe
number_of_k: list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # sono 10 componenti

