
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

data = load_breast_cancer()

X = data.data
y = data.target

X = pd.DataFrame(X, columns=data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

params = {'n_neighbors': range(1,20)}

gs = GridSearchCV(clf, params)
gs.fit(X_train, y_train)

plt.errorbar(gs.cv_results_['param_n_neighbors'].data,
             gs.cv_results_['mean_test_score'],
             yerr=gs.cv_results_['std_test_score'],
              label='test')


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=8, max_iter= 2000)

params = {
    'hidden_layer_sizes': [(10,),(50,),(100, ),
                           (10,10,), (50, 50, ), (100, 100, ),
                           (10, 5, ), (5,5, ), (30,20, 10), (100, 1000, 50,),
                           (1000, 100, 50),(10,10,10),(50, 50, 50), (100, 100, 100,)
                           ],
    'activation': ['identity', 'logistic', 'tanh','relu'],
    'beta_1': [0.9, 0.8, 0.7, 0.6, 0.5],
    'beta_2': [0.999, 0.9, 0.8, 0.7],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

}

gs = GridSearchCV(clf, params)
