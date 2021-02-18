import torch
from glob import glob

'''
txts = glob('/home/jhjeong/Librispeech_data/LibriSpeech/train-clean-100/*/*/*.txt') 

for txt in txts:
    with open(txt, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(" ")
            
            file_txt = " ".join(line[1:]) 
            
            file_name = line[0]

            with open("./train/" + file_name + ".txt", "w") as ff:
                ff.write(file_txt)
'''

'''
txts = glob('/home/jhjeong/Librispeech_data/LibriSpeech/dev-clean/*/*/*.txt')

for txt in txts:
    with open(txt, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(" ")
            
            file_txt = " ".join(line[1:]) 
            
            file_name = line[0]

            with open("./dev/" + file_name + ".txt", "w") as ff:
                ff.write(file_txt)
'''

txts = glob('/home/jhjeong/Librispeech_data/LibriSpeech/test-clean/*/*/*.txt')

for txt in txts:
    with open(txt, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.split(" ")
            
            file_txt = " ".join(line[1:]) 
            
            file_name = line[0]

            with open("./test/" + file_name + ".txt", "w") as ff:
                ff.write(file_txt)