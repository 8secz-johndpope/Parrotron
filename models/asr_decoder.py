import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

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

class ASR_Decoder(nn.Module):
    def __init__(self, label_dim, Embedding_dim, rnn_hidden_size, second_rnn_hidden_size, n_layer, sos_id, eos_id, pad_id):
        super(ASR_Decoder, self).__init__()

        self.embedding = nn.Embedding(label_dim, Embedding_dim)

        self.lstm_1 = nn.LSTM(
            input_size=Embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=n_layer,
            batch_first=True,
            dropout=0
        )  

        
        self.attention_layer = DotProductAttention()

        self.lstm_2 = nn.LSTM(
            input_size=rnn_hidden_size+Embedding_dim,
            hidden_size=second_rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )  

        self.projection_layer = nn.Linear(second_rnn_hidden_size + rnn_hidden_size, label_dim, False)
        self.softmax = nn.Softmax(dim=-1)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    
    def forward(self, encoder_inputs, decoder_inputs):
        '''
        encoder_inputs = [batch, seq_len, feature] 
        decoder_inputs = [batch, feature, seq_len] 
        '''
        # sos를 붙이고 eos를 제거 
        batch_size = encoder_inputs.size(0)
        inputs_add_sos = torch.LongTensor([self.sos_id]*batch_size).view(batch_size, 1)
        
        if encoder_inputs.is_cuda: inputs_add_sos = inputs_add_sos.cuda()

        decoder_inputs = torch.cat((inputs_add_sos, decoder_inputs), dim=1)
        decoder_inputs[decoder_inputs==self.eos_id] = self.pad_id

        embedding_output = self.embedding(decoder_inputs[:, :-1])

        outputs, _ = self.lstm_1(embedding_output)
        context, attn = self.attention_layer(outputs, encoder_inputs, encoder_inputs)   

        cat_context = torch.cat((context, embedding_output), dim=-1)
        total_output, _ = self.lstm_2(cat_context)
        
        cat_total_output = torch.cat((context, total_output), dim=-1)
        asr_decoder_outputs = self.projection_layer(cat_total_output)

        # CE 사용할껀데 softmax 필요없지>?        
        #asr_decoder_outputs = self.softmax(asr_decoder_outputs)

        return asr_decoder_outputs

if __name__ == '__main__':
    # dec test 
    rnn_hidden_size = 256
    
    n_layers = 2
    dropout = 0
    loss = nn.CrossEntropyLoss()
    aaa = ASR_Decoder(label_dim=31, 
                          Embedding_dim=64,
                          rnn_hidden_size=512, 
                          second_rnn_hidden_size=256, 
                          n_layer=3,
                          sos_id=29,
                          eos_id=30,
                          pad_id=0)
    
    
    enc = Variable(torch.randn(2, 49, 512))

    dec = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,30],[1,2,3,4,5,6,7,8,9,10,11,12,30]]).long()

    context = aaa(enc, dec)

    print(context.shape)
    print(dec.shape)

    output = loss(context.contiguous().view(-1, context.size(-1)), dec.contiguous().view(-1))

    print(output)


    