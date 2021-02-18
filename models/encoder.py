import torch
import torch.nn as nn
from torch.autograd import Variable
from 

#convolution rnn은 추후에 추가할 예정
class Encoder(nn.Module):
    """
    n_layer는 2 이상이여야 함
    """
    def __init__(self, rnn_hidden_size, n_layers, dropout, bidirectional):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        embedding_size = 608

        self.lstm_1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )

        self.lstm_2 = nn.LSTM(
            input_size=rnn_hidden_size * 2 if bidirectional else rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=n_layers-1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def flatten_parameters(self):
        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()

    def forward(self, x):
        '''
        x =  (B, S_L, S_dim)
        '''
        x = x.transpose(1, 2) # ->[batch, feature, seq_len] 
        x = x.unsqueeze(1)
    
        x = self.conv(x) # [batch, chanel, feature, seq_len] 
        sizes = x.size()

        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) # [batch, chanel * feature, seq_len]  
        x = x.transpose(1, 2).contiguous() # [batch, seq_len, chanel * feature]  

        x, _ = self.lstm_1(x)
        output, _ = self.lstm_2(x) # [batch, seq_len, feature] 
        
        return output


if __name__ == '__main__':
    
    # enc test 
    rnn_hidden_size = 256
    
    n_layers = 5
    
    dropout = 0
    
    aaa = Encoder(rnn_hidden_size=256,
                      n_layers=5, 
                      dropout=0.3, 
                      bidirectional=True).cuda()
    
    wow = Variable(torch.randn(4, 200, 80)).cuda()

    a = aaa(wow)
    
    print(a.shape)