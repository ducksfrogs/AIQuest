import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed=42
tf.random.set_seed(seed)
np.random.seed(seed)



#filenames_normal = tf.io.gfile.glob('input/train/*/normal/*')
#filenames_anomaly = tf.io.gfile.glob('input/train/*/anomaly/*')

#num_normal = len(filenames_normal)
#num_anomaly = len(filenames_anomaly)

data_dir = pathlib.Path('input/train')
filenames = tf.io.gfile.glob(str(data_dir)+'/*/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print("Number of examples:", num_samples)

train_files = filenames[:2708]
val_files = filenames[2708: 3008]
test_files = filenames[-300:]


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols

fig, axes = plt.subplots(rows, cols, figsize=(10,12))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-0.04, 0.04, 0.01))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)


tf.strings.split('input/data/train/*/', os.path.sep)
