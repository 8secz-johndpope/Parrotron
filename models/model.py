import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, autograd
import math
from torch.autograd import Variable

class Parrotron(nn.Module):
    def __init__(self, encoder, spectrogram_decoder, asr_decoder):
        super(Parrotron, self).__init__()

        self.encoder = encoder
        self.spectrogram_decoder = spectrogram_decoder
        self.asr_decoder = asr_decoder

    def forward(self, inputs, tts_inputs, targets):
        
        txt_outputs = None

        encoder_outputs = self.encoder(inputs)

        mel_outputs_postnet = self.spectrogram_decoder(encoder_outputs, tts_inputs)
        
        #txt_outputs = self.asr_decoder(encoder_outputs, targets)

        return mel_outputs_postnet, txt_outputs

    def inference(self, inputs, tts_inputs, targets):
        
        txt_outputs = None

        encoder_outputs = self.encoder(inputs)

        mel_outputs_postnet = self.spectrogram_decoder.inference(encoder_outputs, tts_inputs)
        
        #txt_outputs = self.asr_decoder(encoder_outputs, targets)

        return mel_outputs_postnet, txt_outputs


    