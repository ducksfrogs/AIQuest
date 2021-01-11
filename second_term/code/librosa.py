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


plt.plot(normal[0])

plt.plot(anomaly[0])

melspec_normal = []
for n in normal:
    m = librosa.feature.melspectrogram(n, n_mels=256)
    m = librosa.power_to_db(m).astype(np.float32)
    melspec_normal.append(m)
melspec_normal = np.array(melspec_normal)

melspec_anomaly = []
for a in melspec_anomaly:
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

import librosa.display
librosa.display.specshow(melspec_normal[0])
