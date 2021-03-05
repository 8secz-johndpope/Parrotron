from glob import glob
import os
import wave
import librosa
import soundfile as sf
import wave


def get_duration(audio_path):
    audio = wave.open(audio_path)
    frames = audio.getnframes()
    rate = audio.getframerate()
    duration = frames / float(rate)
    return duration


with open("/home/jhjeong/jiho_deep/Parrotron/label,csv/toy_test.csv", "r") as f:
    lines = f.readlines()

for line in lines:
    name = line.split(",")[0]
    time = get_duration(name)
    print(time)
    if time > 5:
        print(time)
        print(line)

'''

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
'''