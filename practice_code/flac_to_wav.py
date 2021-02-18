import torch
from glob import glob
import wave

import soundfile as sf
import librosa
import os

def flac_to_wav(orinal_path, new_path):
    w_data, w_sr = sf.read(orinal_path) 

    sf.write(new_path, w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav


flac_list = glob("/home/jhjeong/librispeech_old/LibriSpeech/test-clean/*/*/*.flac")

for flac_path in flac_list:
    file_name = flac_path.split('/')[-1][:-5]
    
    new_path = "./wav/" + file_name + ".wav"
    
    flac_to_wav(flac_path, new_path)