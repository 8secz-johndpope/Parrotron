import torch
import torch.nn as nn
from torch.autograd import Variable
from models.ConvLSTM import ConvBLSTM

#convolution rnn은 추후에 추가할 예정
class Encoder(nn.Module):
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
        
        self.Clstm = ConvBLSTM(
            in_channels=32, 
            hidden_channels=32, 
            kernel_size=(3, 1), 
            num_layers=1, 
            bias=True, 
            batch_first=True
        )

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
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )

        self.lstm_3 = nn.LSTM(
            input_size=rnn_hidden_size * 2 if bidirectional else rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        self.projection_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU()
        )

        self.projection_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU()
        )
        self.projection_3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU()
        )
        
        self.batchnorm_1 = nn.BatchNorm1d(512)
        self.batchnorm_2 = nn.BatchNorm1d(512)
        self.batchnorm_3 = nn.BatchNorm1d(512)

    def flatten_parameters(self):
        self.lstm_1.flatten_parameters()
        self.lstm_2.flatten_parameters()

    def forward(self, x):
        '''
        x =  (B, S_L, S_dim)
        '''
        x = x.transpose(1, 2) # ->[batch, feature, seq_len] 
        x = x.unsqueeze(1)
    
        x = self.conv(x) # [batch, chanel, feature, seq_len] -> [b, c, h, t]

        x = x.permute(0, 3, 1, 2)
        x = x.unsqueeze(4) # B, T, C, H, W
        
        x_inverse = x.flip(1) # T에 대해서 flip -> x inverse 생성 (bidirection을 위해)

        x = self.Clstm(x, x_inverse) # B, T, C, H, W 
        x = x.permute(0, 4, 2, 3, 1) # B, W, C, H, T
        sizes = x.size()

        x = x.view(sizes[0], sizes[1] * sizes[2] * sizes[3], sizes[4]) # [batch, chanel * feature, seq_len]  
        x = x.transpose(1, 2).contiguous() # [batch, seq_len, chanel * feature]  

        #first layer
        x, _ = self.lstm_1(x)
        x = self.projection_1(x)
        x = x.transpose(1, 2).contiguous() # [batch, feature, seq_len] 
        x = self.batchnorm_1(x)
        x = x.transpose(1, 2).contiguous() # [batch, seq_len, feature] 

        #second layer
        x, _ = self.lstm_2(x)
        x = self.projection_2(x)
        x = x.transpose(1, 2).contiguous() # [batch, feature, seq_len] 
        x = self.batchnorm_2(x)
        x = x.transpose(1, 2).contiguous() # [batch, seq_len, feature] 

        #third layer
        x, _ = self.lstm_3(x)
        x = self.projection_3(x)
        x = x.transpose(1, 2).contiguous() # [batch, feature, seq_len] 
        x = self.batchnorm_3(x)
        output = x.transpose(1, 2).contiguous() # [batch, seq_len, feature] 

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