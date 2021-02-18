import math
import os
import time
from matplotlib import pyplot as plt
import pandas as pd
import librosa.display, librosa
import numpy as np
import scipy.signal
import soundfile as sf
#import sox
import torch
import csv
from .spec_augment import spec_augment
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
import matplotlib
import torchaudio


windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

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

char2index, index2char = load_label('./label,csv/english_unit.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    def parse_audio(self, audio_path):
        raise NotImplementedError

class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, feature_type, normalize, spec_augment):
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.feature_type = feature_type
        self.spec_augment = spec_augment

        self.hop_length = int(round(self.sample_rate * 0.001 * self.window_stride))

        mel_spec = dict(win_length = self.window_size, hop_length=self.hop_length, n_fft=1024)
        
        self.transforms = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate, n_mfcc=80,
                log_mels=True, melkwargs=mel_spec)

        self.tts_transforms = torchaudio.transforms.Spectrogram(
                n_fft=2048, win_length=self.window_size, hop_length=self.hop_length)

    def parse_audio(self, audio_path):

        signal, _ = torchaudio.load(audio_path)
        spect = self.transforms(signal)

        return spect

    def parse_audio_tts(self, audio_path):
       
        signal, _ = torchaudio.load(audio_path)
        spect = self.tts_transforms(signal)
        
        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, feature_type, normalize, spec_augment):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        
        self.ids = ids
        self.size = len(ids)
               
        super(SpectrogramDataset, self).__init__(audio_conf, feature_type, normalize, spec_augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path, tts_path = sample[0], sample[1], sample[2]
        
        spect = self.parse_audio(audio_path)
        tts_spect = self.parse_audio_tts(tts_path)
        transcript = self.parse_transcript(transcript_path)
                
        return spect, transcript, tts_spect

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as f:
            
            transcript_list = []
            transcript = f.read()
            transcript = transcript.strip()
            
            for char in transcript:
                transcript_list.append(char2index[char])
            
            transcript_list.append(EOS_token)           
            
        return transcript_list

    def __len__(self):
        return self.size

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    def seq_length_(p):
        #p[0] [1, feq_dim, spec_len]
        return p[0].size(2)
    
    def tts_seq_length_(p):
        #p[2] [1, feq_dim, spec_len]
        return p[2].size(2)

    def target_length_(p):
        return len(p[1])
    
    seq_lengths = [s[0].size(2) for s in batch] 
    target_lengths = [len(s[1])-1 for s in batch] # eos 제거한거임 -1로
    tts_seq_lengths = [s[2].size(2) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]
    max_tts_seq_sample = max(batch, key=tts_seq_length_)[2]
    
    
    max_seq_size = max_seq_sample.size(2)
    max_tts_seq_size = max_tts_seq_sample.size(2)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    tts_feat_size = max_tts_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    tts_seqs = torch.zeros(batch_size, max_tts_seq_size, tts_feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long) 
   
    targets.fill_(PAD_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0].squeeze().transpose(0,1)
        target = sample[1]
        tts_tensor = sample[2].squeeze().transpose(0,1)
        
        seq_length = tensor.size(0)
        tts_seq_length = tts_tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        tts_seqs[x].narrow(0, 0, tts_seq_length).copy_(tts_tensor)
        
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    # seqs (B, S_L, S_dim)
    # targets (B, T_L)
    # tts_seqs (B, TS_L, TS_dim)

    return seqs, targets, tts_seqs, seq_lengths, target_lengths, tts_seq_lengths



