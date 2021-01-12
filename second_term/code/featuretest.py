import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import librosa

files_normaly = sorted(glob.glob(os.path.abspath('../input/train/*/normal/*.wav')))
files_anomaly = sorted(glob.glob(os.path.abspath('../input/train/*/anomaly/*.wav')))
files_test = sorted(glob.glob(os.path.abspath('../input/test/*.wav')))

normal = []
for file in files_normaly:
    y, sr = librosa.load(file, sr=None)
    normal.append(y)

normal = np.array(normal)

anomaly = []
for file in files_anomaly:
    y, sr = librosa.load(file, sr=None)
    anomaly.append(y)
anomaly = np.array(anomaly)

test = []
for file in files_test:
    y, sr = librosa.load(file, sr=None)
    test.append(y)
test = np.array(test)


train = np.concatenate([normal, anomaly])

train_Y = np.concatenate([np.zeros(len(normal)), np.ones(len(anomaly))])

from sklearn.metrics import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(train, train_Y, test_size=0.33, random_state=42)



import xgboost as xgb

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                            GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC
from sklearn.model_selection import KFold


train = train.reshape(train.shape[0],-1)

train_Y = np.concatenate([np.zeros(len(melspec_normal)), np.ones(len(melspec_anomaly))])
from sklearn.model_selection import KFold

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 4
kf = KFold(ntrain, n_fold=NFOLDS)

class SkearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)



def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain, ))
    oof_test = np.zeros((ntest, ))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)



rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.7,
}

gb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

rf = SkearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SkearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
gb = SkearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SkearnHelper(clf=SVC, seed=SEED, params=svc_params)
ada = sklearn(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test)
rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test)
ada_oof_train, ada_oof_test = get_oof(ada, X_train, y_train, X_test)
gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, X_test)
