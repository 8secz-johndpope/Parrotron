import torch
from glob import glob
import wave

import soundfile as sf
import librosa
import os



w_data, w_sr = sf.read("/home/jhjeong/jiho_deep/Parrotron/LibriSpeech/train-clean-100/19/198/19-198-0000.flac") # data, sampling rate divide

sf.write('test.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav


w, sr = librosa.load('test.wav', sr=16000)
'''
m, _ = librosa.load(save_dir+m_id+'.wav', sr=20000)
'''

w_resample = librosa.resample(w, sr, 8000)
sf.write('8000k.wav', w_resample, 8000, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav
'''
g_sr = 16000 # goal sampling rate


m_resample = librosa.resample(m, sr, g_sr)

sf.write(save_dir + w_id + '_resample16k.wav', w_resample, g_sr, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav
sf.write(save_dir + m_id + '_resample16k.wav', m_resample, g_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
'''

#pcm2wav("/home/jhjeong/jiho_deep/Parrotron/LibriSpeech/train-clean-100/19/198/19-198-0000.flac","test.wav")

