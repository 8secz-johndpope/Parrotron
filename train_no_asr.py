import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import editdistance
import scipy.signal
import numpy as np 

# torch 관련
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

from models.encoder import Encoder
from models.decoder import Decoder
from models.asr_decoder import ASR_Decoder
from models.model import Parrotron, Parrotron_No_ASR
from models.eval_distance import eval_wer, eval_cer
from models.data_loader import SpectrogramDataset, AudioDataLoader, AttrDict
from models.loss_function import ParrotronLoss, ParrotronLossNoASR

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

def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        seqs, _, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data
        seqs = seqs.to(device) # (batch_size, time, freq)
        tts_seqs = tts_seqs.to(device)

        mel_outputs_postnet, mel_outputs, txt_outputs = model(seqs, tts_seqs, None, 0)
        
        loss = criterion(mel_outputs_postnet, mel_outputs, tts_seqs)

        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
      
        if i % 100 == 0:
            print('{} train_batch: {:4d}/{:4d}, train_spec_loss: {:.4f}, train_time: {:.2f}'
                  .format(datetime.datetime.now(), i, total_batch_num, loss.item(),  time.time() - start_time))
            start_time = time.time()
    
    train_loss = total_loss / total_batch_num

    return train_loss


def evaluation(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    total_num = 0

    start_time = time.time()
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            seqs, _, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths = data
            seqs = seqs.to(device) # (batch_size, time, freq)
            tts_seqs = tts_seqs.to(device)

            mel_outputs_postnet, mel_outputs, txt_outputs = model(seqs, tts_seqs, None, 0)
            
            loss = criterion(mel_outputs_postnet, mel_outputs, tts_seqs)
            total_loss += loss.item()
            
    eval_loss = total_loss / total_batch_num
    
    return eval_loss

def main():
    yaml_name = "./label,csv/Parrotron.yaml"
    
    with open("./parrotron_no_asr.txt", "w") as f:
        f.write(yaml_name)
        f.write('\n')
        f.write('\n')
        f.write("학습 시작")
        f.write('\n')

    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    
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
    
    model = Parrotron_No_ASR(enc, dec).to(device)
    #model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/Parrotron/plz_load/parrotron_no_asr.pth"))
    
    model = nn.DataParallel(model)
    #-------------------------- Loss Initialize --------------------------
    spec_criterion = nn.MSELoss()    
    criterion = ParrotronLossNoASR(spec_criterion)
    #-------------------- Model Pararllel & Optimizer --------------------
        
    optimizer = optim.Adam(model.module.parameters(), 
                                lr=config.optim.lr,
                                betas=(0.9, 0.999), 
                                eps=1e-06, 
                                weight_decay=1e-06)
    
    #-------------------------- Data load --------------------------
    #train dataset
    train_dataset = SpectrogramDataset(audio_conf, 
                                       "/home/jhjeong/jiho_deep/Parrotron/label,csv/train.csv",
                                       feature_type=config.audio_data.type, 
                                       normalize=True, 
                                       spec_augment=True)

    train_loader = AudioDataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=44,
                                    drop_last=True)
    
    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     "/home/jhjeong/jiho_deep/Parrotron/label,csv/test.csv", 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                 shuffle=True,
                                 num_workers=config.data.num_workers,
                                 batch_size=44,
                                 drop_last=True)
    
    print(" ")
    print("parrotron 를 학습합니다.")
    print(" ")

    pre_test_loss = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            print("lr = ", lr)

        print('{} 학습 시작'.format(datetime.datetime.now()))
        train_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_total_time = time.time() - train_time
        print('{} Epoch {} (Train) Loss {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, train_loss, train_total_time))
        
        print('{} 평가 시작'.format(datetime.datetime.now()))
        eval_time = time.time()
        eval_loss = evaluation(model, val_loader, criterion, device)
        eval_total_time = time.time() - eval_time
        print('{} Epoch {} (Eval) Loss {:.4f}, time: {:.2f}'.format(datetime.datetime.now(), epoch+1, eval_loss, eval_total_time))
        
        with open("./parrotron_no_asr.txt", "a") as f:
            f.write("lr = " + str(lr))
            f.write('\n')
            f.write('Epoch %d (Train) Loss %0.4f time %0.4f' % (epoch+1, train_loss, train_total_time))
            f.write('\n')
            f.write('Epoch %d (Eval) Loss %0.4f time %0.4f' % (epoch+1, eval_loss, eval_total_time))
            f.write('\n')
        
        if pre_test_loss > eval_loss:
            print("best model을 저장하였습니다.")
            torch.save(model.module.state_dict(), "./plz_load/best_parrotron_no_asr.pth")
            pre_test_loss = eval_loss
        
        torch.save(model.module.state_dict(), "./plz_load/parrotron_no_asr.pth")               

if __name__ == '__main__':
    main()