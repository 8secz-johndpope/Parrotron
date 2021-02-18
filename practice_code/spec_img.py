import matplotlib.pyplot as plt
import librosa.display

import numpy as np
import pandas as pd
import librosa


filename = librosa.util.example_audio_file()
y, sr = librosa.load("/home/jhjeong/jiho_deep/Parrotron/test.wav")
y = y[:100000] # shorten audio a bit for speed

window_size = 1024
window = np.hanning(window_size)
stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
out = 2 * np.abs(stft) / np.sum(window)

# For plotting headlessly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

plt.rcParams["figure.figsize"] = (10,4)
fig = plt.Figure()
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
fig.savefig('spec.png')