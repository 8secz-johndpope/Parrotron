import matplotlib.pyplot as plt
import librosa.display

import numpy as np
import pandas as pd
import librosa


filename = librosa.util.example_audio_file()

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

spf = wave.open("/home/jhjeong/jiho_deep/Parrotron/19-198-0011.wav")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")


# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

plt.rcParams["figure.figsize"] = (10,4)
plt.figure(1)
plt.title("Signal Wave...")
plt.plot(signal)
plt.show()
plt.savefig('fig1.png', dpi=300)