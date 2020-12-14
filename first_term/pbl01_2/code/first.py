import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


answer =pd.read_csv('./data/answer.csv', header=None, index_col=0)
winner = pd.read_csv('./data/winner.csv', header=None, index_col=0)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
