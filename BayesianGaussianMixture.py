# want to see sameion.csv file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))
