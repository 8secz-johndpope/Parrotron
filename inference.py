import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import librosa
import editdistance
import scipy.signal
import numpy as np 
import soundfile as sf

# torch 관련
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import matplotlib

from models.encoder import Encoder
from models.decoder import Decoder
from models.asr_decoder import ASR_Decoder
from models.model import Parrotron
from models.eval_distance import eval_wer, eval_cer
from models.data_loader import SpectrogramDataset, AudioDataLoader, AttrDict

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue
            
            index, char = line.split('   ')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

# SOS_token, EOS_token, PAD_token 정의
char2index, index2char = load_label('./label,csv/english_unit.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

def compute_cer(preds, labels):
    total_wer = 0
    total_cer = 0

    total_wer_len = 0
    total_cer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            if a == EOS_token: # eos
                break
            units.append(index2char[a])
            
        for b in pred:
            if b == EOS_token: # eos
                break
            units_pred.append(index2char[b])

        label = ''.join(units)
        pred = ''.join(units_pred)

        wer = eval_wer(pred, label)
        cer = eval_cer(pred, label)
        
        wer_len = len(label.split())
        cer_len = len(label.replace(" ", ""))

        total_wer += wer
        total_cer += cer

        total_wer_len += wer_len
        total_cer_len += cer_len
        
    return total_wer, total_cer, total_wer_len, total_cer_len

def inference(model, val_loader, device):
    model.eval()

    total_asr_loss = 0
    total_spec_loss = 0
    total_num = 0

    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i % 10 == 0:
                print(i)
                
            seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data
            
            seqs = seqs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            tts_seqs = tts_seqs.to(device)
            
            mel_outputs_postnet, _ = model.inference(seqs, tts_seqs, targets)
            #mel_outputs_postnet, _ = model(seqs, tts_seqs, targets)

            spec = mel_outputs_postnet.squeeze().transpose(0,1).numpy()
            
            path = './test_wav'
            os.makedirs(path, exist_ok=True)
            y_inv = librosa.griffinlim(spec, hop_length=200, win_length=800, window='hann')
            sf.write('./test_wav/'+ str(i) +'.wav', y_inv, 16000)
            #print(y_inv.shape)

            path1 = './test_img'
            os.makedirs(path1, exist_ok=True)
            matplotlib.image.imsave('./test_img/'+ str(i) +'.png', spec)
            
    return 

def main():
    yaml_name = "./label,csv/Parrotron.yaml"
    
    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    device = torch.device('cpu')
    
    windows = { 'hamming': scipy.signal.hamming,
                'hann': scipy.signal.hann,
                'blackman': scipy.signal.blackman,
                'bartlett': scipy.signal.bartlett
                }

    SAMPLE_RATE = config.audio_data.sampling_rate
    WINDOW_SIZE = config.audio_data.window_size
    WINDOW_STRIDE = config.audio_data.window_stride
    WINDOW = config.audio_data.window

    audio_conf = dict(sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE,
                        window_stride=WINDOW_STRIDE,
                        window=WINDOW)

    hop_length = int(round(SAMPLE_RATE * 0.001 * WINDOW_STRIDE))
 
    #wow = torchaudio.transforms.GriffinLim(n_fft=2048, win_length=WINDOW_SIZE, hop_length=hop_length)

    #-------------------------- Model Initialize --------------------------
    enc = Encoder(rnn_hidden_size=256,
                  dropout=0.5, 
                  bidirectional=True)

    dec = Decoder(target_dim=1025,
                  pre_net_dim=256,
                  rnn_hidden_size=1024,
                  encoder_dim=256*2,
                  attention_dim=128,
                  attention_filter_n=32,
                  attention_filter_len=31,  
                  postnet_hidden_size=512,
                  postnet_filter=5,
                  dropout=0.5)
    
    asr_dec = ASR_Decoder(label_dim=31, 
                          Embedding_dim=64,
                          rnn_hidden_size=512, 
                          second_rnn_hidden_size=256, 
                          n_layer=3,
                          sos_id=SOS_token,
                          eos_id=EOS_token,
                          pad_id=PAD_token)
    
    model = Parrotron(enc, dec, asr_dec).to(device)

    #model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/Parrotron/plz_load/best_parrotron.pth"))
    
    #inference dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     "/home/jhjeong/jiho_deep/Parrotron/label,csv/toy_test.csv", 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=False,
                                 num_workers=config.data.num_workers,
                                 batch_size=1,
                                 drop_last=True)
    
    print(" ")
    print("Inferrence 합니다.")
    print(" ")

    print('{} 평가 시작'.format(datetime.datetime.now()))
    eval_time = time.time()
    eval_spec_loss = inference(model, val_loader, device)
    
    

if __name__ == '__main__':
    main()