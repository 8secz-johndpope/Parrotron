import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Prenet(nn.Module):
    def __init__(self, in_dim, out_dim, drop_out_p):
        super(Prenet, self).__init__()
        
        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_out_p),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_out_p)
        )

    def forward(self, x):
        
        output = self.pre_net(x)

        return output

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, target_dim, filter_num, dropout, padding):
        super(Postnet, self).__init__()

        self.conv = nn.Conv1d(target_dim, filter_num,
                            kernel_size=5, stride=1,
                            padding=padding, dilation=1,
                            bias=True)

        self.conv_1 = nn.Conv1d(filter_num, filter_num,
                            kernel_size=5, stride=1,
                            padding=padding, dilation=1,
                            bias=True)
        
        self.conv_2 = nn.Conv1d(filter_num, filter_num,
                            kernel_size=5, stride=1,
                            padding=padding, dilation=1,
                            bias=True)

        self.conv_3 = nn.Conv1d(filter_num, filter_num,
                            kernel_size=5, stride=1,
                            padding=padding, dilation=1,
                            bias=True)

        self.conv_4 = nn.Conv1d(filter_num, target_dim,
                            kernel_size=5, stride=1,
                            padding=padding, dilation=1,
                            bias=True)
        
        self.total_conv = nn.Sequential(
            self.conv,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            nn.Dropout(dropout),
            self.conv_1,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            nn.Dropout(dropout),
            self.conv_2,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            nn.Dropout(dropout),
            self.conv_3,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            nn.Dropout(dropout),
            self.conv_4,
            nn.BatchNorm1d(target_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        
        x = x.transpose(1, 2).contiguous()
        x = self.total_conv(x)
        
        return x

# dot product -> 나중에 location attention으로 바꾸기
class DotProductAttention(nn.Module):
    """
    Dot-Product Attention
    Inputs: decoder_inputs, encoder_inputs
        - **decoder_inputs** (batch, q_len, d_model)
        - **encoder_inputs** (batch, k_len, d_model)
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()
      
    def forward(self, query, key, value):

        score = torch.bmm(query, key.transpose(1, 2))
        
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)

        return context, attn

class Decoder(nn.Module):
    def __init__(self, target_dim, pre_net_dim, rnn_hidden_size, second_rnn_hidden_size, postnet_hidden_size, n_layers, dropout, attention_type):
        super(Decoder, self).__init__()

        self.pre_net = Prenet(target_dim, pre_net_dim, dropout)

        self.attention_layer = DotProductAttention()
        
        self.projection_layer = nn.Linear(second_rnn_hidden_size + rnn_hidden_size, target_dim, False)

        self.lstm_1 = nn.LSTM(
            input_size=pre_net_dim,
            hidden_size=rnn_hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0
        )

        self.lstm_2 = nn.LSTM(
            input_size=rnn_hidden_size+pre_net_dim,
            hidden_size=second_rnn_hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0
        )

        self.postnet = Postnet(target_dim, postnet_hidden_size, dropout, int((5 - 1) / 2)) # 5 = kernel size

    def forward_step(self, encoder_inputs, decoder_inputs):     
        decoder_inputs, _ = self.lstm_1(decoder_inputs) # (batch, seq_len, rnn_hidden_size)
        
        context, attn = self.attention_layer(decoder_inputs, encoder_inputs, encoder_inputs)      

        return context, attn

    
    def forward(self, encoder_inputs, decoder_inputs):
        '''
        encoder_inputs = [batch, seq_len, feature] 
        decoder_inputs = [batch, feature, seq_len] ex) torch.Size([2, 440, 1025])
        '''
        go_frame = torch.zeros(decoder_inputs.shape[0], 1, decoder_inputs.shape[2])

        if encoder_inputs.is_cuda: go_frame = go_frame.cuda()
        
        decoder_inputs = torch.cat((go_frame, decoder_inputs), dim=1)
        decoder_inputs = self.pre_net(decoder_inputs)
        
        #decoder_inputs = [2, 101, 256]

        context, attn = self.forward_step(encoder_inputs, decoder_inputs)
       
        attention_hidden = torch.cat((decoder_inputs, context), dim=-1)
        decoder_hidden, _ = self.lstm_2(attention_hidden)

        attention_hidden = torch.cat((context, decoder_hidden), dim=-1)
        spectogram = self.projection_layer(attention_hidden)

        spectogram_postnet = self.postnet(spectogram)
        spectogram_postnet = spectogram + spectogram_postnet.transpose(1, 2)
        
        return spectogram_postnet[:, :-1 ,:]

if __name__ == '__main__':
    # dec test 
    rnn_hidden_size = 256
    
    n_layers = 2
    dropout = 0
    
    #class Decoder(nn.Module):
    #def __init__(self, target_dim, pre_net_dim, rnn_hidden_size, n_layers, dropout, attention_type):

    aaa = Decoder(1025, 256, 512, 1024, 512, 2, 0.5, "wow")
    
    enc = Variable(torch.randn(2, 49, 512))
    dec = Variable(torch.randn(2, 100, 1025))
    
    answer = Variable(torch.randn(2, 100, 1025))

    loss = nn.MSELoss()

    context = aaa(enc, dec)
    
    output = loss(context, answer)

   

    