# this is a file to open and read the data from the csv file and understand the data

import matplotlib.pyplot as plt
import pandas as pd

# open a csc file but do not consider the last 10 columns
numbers = range(256)
df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=numbers)
print(df.shape)

# this is a dataset of 1593 images of 16x16 pixels
# each image is a handwritten digit from 0 to 9
# each image is represented by a 256-dimensional vector

# the first 256 columns are the pixels of the image
# the last 10 columns are the labels of the image
# the labels are 10-dimensional vectors

# see the first 10 images
# each image is a 16x16 matrix
# each pixel is a number between 0 and 1
# 0 means white and 1 means black

# I want to see the 80 images
fig, ax = plt.subplots(20, 10, figsize=(10, 20))
for i in range(20):
    for j in range(10):
        ax[i, j].imshow(df.iloc[i * 10 + j].values.reshape(16, 16), cmap='gray')
        ax[i, j].axis('off')
plt.tight_layout()
plt.show()
