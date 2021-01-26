import wave
import struct
import numpy as np
from pylab import *

def createSinWave(A, f0, f1, fs, length):
    data = []
    for n in arange(length*fs):
        s = A*(np.sin(2.np.pi*f0*n/fs) + np.sin(2*np.pi*f1*n/fs))
        if s > 1.0: s = 1.0
        if s< -1.0: s = -1.0

        data.append(s)
        data = struct.pack("h"* len(data), *data)
    return data


def play(data, fs, bit):
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,

    channels = 1
    rate = int(fs)
    ot
