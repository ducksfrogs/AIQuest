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

def get_spectrogram(waveform):
    zero_padding = tf.zeros([160000] - tf.shape(waveform), dtype=tf.float64)

    waveform = tf.cast(waveform, tf.float64)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)

print("Label: ", label)
print("Waveform shape: ", waveform.shape)
print("Spectrogram shape :", spectrogram.shape)
display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    X = np.arange(160000, step=height+1)
    Y = range(height)
    ax.pcolormesh(X,Y, log_spec)

fig, axes = plt.subplots(2, figsize=(12,8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title("Wave form")
axes[0].set_xlim([0, 160000])
plot_spectrogram(spectrogram.numpy(), axes[1])
