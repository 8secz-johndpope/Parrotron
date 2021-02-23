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

    def flatten_parameters(self):
        self.encoder.lstm_1.flatten_parameters()
        self.encoder.lstm_2.flatten_parameters()
        self.encoder.lstm_3.flatten_parameters()

        self.spectrogram_decoder.attention_rnn.flatten_parameters()

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


if __name__ == '__main__':

    rnn_hidden_size = 256
    
    n_layers = 2
    
    dropout = 0
    
    enc = Encoder(rnn_hidden_size, n_layers, dropout, True)

    dec = Decoder(512, 256, 0)
    
    model = Parrotron(enc, dec)
    
    aaa = Variable(torch.randn(1, 16, 80))
    bbb = Variable(torch.randn(1, 20, 80))
    
    answer = Variable(torch.randn(1, 20, 80))

    loss = nn.MSELoss()

    context = model(aaa, bbb)

    output = loss(context, answer)

    print(output)

    