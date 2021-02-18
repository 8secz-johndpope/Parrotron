from glob import glob
import os
import wave
import librosa
import soundfile as sf

def down_sample(input_wav, origin_sr, resample_sr, output_path):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    
    
    sf.write(output_path, resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')



wav_list = glob('/home/jhjeong/Librispeech_data/train/24000_TTS_wav/*.wav')
# 24000

for wav_path in wav_list:    
    output_path = "/home/jhjeong/Librispeech_data/train/TTS_wav/" + wav_path.split("/")[-1]
    
    down_sample(wav_path, 24000, 16000, output_path)
