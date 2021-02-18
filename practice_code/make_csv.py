from glob import glob
import os
import wave
import librosa
import soundfile as sf

txt_list = glob('/home/jhjeong/Librispeech_data/train/txt/*.txt')

wav_path = "/home/jhjeong/Librispeech_data/train/wav/"
txt_path_1 = "/home/jhjeong/Librispeech_data/train/txt/"
tts_path = "/home/jhjeong/Librispeech_data/train/TTS_wav/"


with open("train.csv", "w") as f:
    f.write("")

for txt_path in txt_list:
    file_name = txt_path.split("/")[-1][:-4]

    final_wav_path = wav_path + file_name + ".wav"
    final_txt_path = txt_path_1 + file_name + ".txt"
    final_tts_path = tts_path + file_name + ".wav"

    with open("train.csv", "a") as f:
        f.write(final_wav_path)
        f.write(",")
        f.write(final_txt_path)
        f.write(",")
        f.write(final_tts_path)
        f.write("\n")
    