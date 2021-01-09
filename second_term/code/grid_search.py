import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

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

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

C_range_exp = np.arange(-15.0, 21.0)
C_range = 10 ** C_range_exp

from sklearn.model_selection import GridSearchCV
param = {'C' : C_range}
gs = GridSearchCV(clf, param)
gs.fit(X_train, y_train)

gs.cv_results_

gs.best_params_,
gs.best_score_,
gs.best_estimator_

clf_best = gs.best_estimator_
clf_best.fit(X_train, y_train)
clf_best.score(X_test, y_test)

plt.errorbar(gs.cv_results_['param_C'].data,
             gs.cv_results_['mean_test_score'],
             yerr=gs.cv_results_['std_test_score'],
             label='test(val)')

plt.errorbar(gs.cv_results_['param_C'].data,
             gs.cv_results_['mean_fit_time'],
             yerr=gs.cv_results_['std_fit_time'],label='train')

plt.errorbar(gs.cv_results_['param_C'].data,
             gs.cv_results_['mean_score_time'],
             yerr=gs.cv_results_['std_score_time'],
             label='test')

plt.ylim(0,)
plt.xscale('log')


from sklearn.svm import SVC

clf = SVC()

C_range_exp = np.arange(-2.0, 5.0)
C_range = 10 ** C_range_exp

params = [
    {'C': C_range, 'kernel': ['linear','rbf']},
]

gs = GridSearchCV(clf, params, n_jobs=1, verbose=2)
gs.fit(X_train, y_train)

s_linear = [gs.cv_results_['param_kernel']=='linear']

plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_test_score'])
