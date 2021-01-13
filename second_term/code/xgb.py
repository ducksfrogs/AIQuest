import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import librosa
import lightgbm as lgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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


melspec_normal = []
for n in normal:
    m = librosa.feature.melspectrogram(n, n_mels=256)
    m = librosa.power_to_db(m).astype(np.float32)
    melspec_normal.append(m)
melspec_normal = np.array(melspec_normal)

melspec_anomaly = []
for a in anomaly:
    m = librosa.feature.melspectrogram(a, n_mels=256)
    m = librosa.power_to_db(m).astype(np.float32)
    melspec_anomaly.append(m)
melspec_anomaly = np.array(melspec_anomaly)

melspec_test = []
for t in test:
    m = librosa.feature.melspectrogram(t, n_mels=256)
    m = librosa.power_to_db(m).astype(np.float32)
    melspec_test.append(m)
melspec_test = np.array(melspec_test)


train = np.concatenate([melspec_normal, melspec_anomaly])
train = train.reshape(train.shape[0],-1)
test = melspec_test.reshape(melspec_test.shape[0], -1)

target = np.concatenate([np.zeros(len(melspec_normal)), np.ones(len(melspec_anomaly))])

from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(train, target, test_size=0.2, random_state=42)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
test = scaler.transform(test)


dtrain = xgb.DMatrix(train_X, label=train_Y)
deval = xgb.DMatrix(test_X, label=test_Y)
dtest = xgb.DMatrix(test)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

bst = xgb.train(xgb_params,
            dtrain,
            num_boost_round=100)


y_pred = bst.predict(dtest)
sub[1] = pred.astype('int')

sub.to_csv('submit.csv')
