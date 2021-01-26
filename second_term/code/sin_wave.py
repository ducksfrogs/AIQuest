import wave
import struct
import numpy as np
from pylab import *

import pyaudio

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
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
        channels = 1,
        rate = int(fs),
        output=True)
    chunk = 1024
    sp = 0
    buffer = data[sp:sp+chunk]
    stream.close()
    p.terminate()


def dtmf(number):
    freq_row = (697, 770, 852, 941)
    freq_col = (1209, 1336, 1477, 1633)

    if(number=='0'):
        row = 3
        col = 1
    elif (number=='#'):
        row = 3
        col =2
    elif (number=='*'):
        row = 3
        col = 0
    else:
        num = int(number)-1
        row = int(num/3)
        col = int(num/3)

    return ( freq_row[row], freq_col[col])
