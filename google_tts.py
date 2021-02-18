from google.cloud import texttospeech
from glob import glob

def list_languages():
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices().voices
    languages = unique_languages_from_voices(voices)

    print(f" Languages: {len(languages)} ".center(60, "-"))
    for i, language in enumerate(sorted(languages)):
        print(f"{language:>10}", end="" if i % 5 < 4 else "\n")


def unique_languages_from_voices(voices):
    language_set = set()
    for voice in voices:
        for language_code in voice.language_codes:
            language_set.add(language_code)
    return language_set

def text_to_wav(voice_name, text, file_name):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )

    filename = "./wav_train/"+ file_name + ".wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{filename}"')

'''
txt_list = glob("/home/jhjeong/Librispeech_data/dev/txt/*.txt")

for txt in txt_list: 
    with open(txt, "r") as f:
        line = f.readline().strip().lower()
    
    file_name = txt.split('/')[-1][:-4]
    #print("./wav/"+ file_name + ".wav")
    text_to_wav("en-US-Wavenet-G", line, file_name)
'''

train_txt_list = glob("/home/jhjeong/Librispeech_data/train/txt/*.txt")
original_txt = []

for txt in train_txt_list:
    original_txt.append(txt.split("/")[-1][:-4])

train_pre_txt_list = glob("/home/jhjeong/jiho_deep/Parrotron/wav_train/*.wav")
pre_txt_list = []

for pre_txt in train_pre_txt_list:
    pre_txt_list.append(pre_txt.split("/")[-1][:-4])

goal = list(set(original_txt) - set(pre_txt_list))

print(len(goal))

for file_name in goal: 
    final_txt = "/home/jhjeong/Librispeech_data/train/txt/"+file_name+".txt"
    
    with open(final_txt, "r") as f:
        line = f.readline().strip().lower()
    
    #print("./wav_train/"+ file_name + ".wav")
    text_to_wav("en-US-Wavenet-G", line, file_name)