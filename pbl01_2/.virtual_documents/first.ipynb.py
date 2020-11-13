import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


answer =pd.read_csv('./data/answer.csv', header=None, index_col=0)
winner = pd.read_csv('./data/winner.csv', header=None, index_col=0)



answer.head()


plt.scatter(answer, winner)



answer.shape


winner.shape


sa = answer.iloc[:] - winner.iloc[:]


np.sqrt((sa**2).sum())/len(answer)


sa.shape


answer.iloc[0]


len(answer)


np.sqrt((sa**2).sum()/3060)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



mean_squared_error(answer, winner)


mean_absolute_error(answer,winner)


sa


plt.scatter(answer.iloc[0:10],winner.iloc[0:10])



answer.iloc[0:100]


kk
