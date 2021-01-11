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


from sklearn.decomposition import PCA

pca = PCA(whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

clf.fit(X_train_pca, y_train)
clf.score(X_test_pca, y_test)


from sklearn.pipeline import Pipeline

estimators = [('pca', PCA(whiten=True)),
            ('clf', LogisticRegression())]
pipe = Pipeline(estimators)

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

estimators = [('pca', MinMaxScaler()),
                ('clf', SVC(kernel='rbf', C=1e10))]

pipe = Pipeline(estimators)

from sklearn.model_selection import GridSearchCV

params = {'clf__C':[1e-5, 1e-3, 1e-2, 1, 1e2, 1e5, 1e10]}

gs = GridSearchCV(pipe, params)
gs.fit(X_train, y_train)


C_range = [1e-3, 1e-2, 1, 1e2, 1e3]

params = {'clf__C': C_range,
        'clf__keernel': ['linear', 'rbf'],
        'pca__whiten': [True, False],
        'pca__n_components': [30,20,10]}

estimators [('pca', PCA()),
            ('clf', SVC())]

pipe = Pipeline(estimators)

gs = GridSearchCV(pipe, params)

gs.fit(X_train, X_test)
