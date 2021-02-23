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

        self.attention_rnn = nn.LSTM(
            input_size=576,
            hidden_size=second_rnn_hidden_size,
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
        self.rnn_projection = nn.Sequential(
            nn.Linear(in_features=256, out_features=512, bias=True),
            nn.ReLU()
        )
        self.projection_layer = nn.Linear(1024, label_dim, False)
        self.softmax = nn.Softmax(dim=-1)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def forward_step(self, encoder_inputs, decoder_input, attention_context):
        '''
        output : (mel_output, new attention context vector)
        '''
        attention_context = attention_context.squeeze(1)
       
        cell_input_1 = torch.cat((decoder_input, attention_context), -1) # pre-net and attention context vector concatenated
        cell_input_1 = cell_input_1.unsqueeze(1)
        
        decoder_input, _ = self.attention_rnn(cell_input_1) # (batch, seq_len, rnn_hidden_size) # passed through a stack of 2 unidirectional         
        attention_context = attention_context.unsqueeze(1)

        decoder_input = self.rnn_projection(decoder_input)
        context, attn = self.attention_layer(decoder_input, encoder_inputs, encoder_inputs) 

        attention_context = context 
        cell_input_2 = torch.cat((decoder_input, attention_context), -1) # concatenation of the LSTM output and attention context vector
        
        decoder_output = self.projection_layer(cell_input_2) # projected through a linear transform
        
        return decoder_output, context

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

        decoder_inputs = self.embedding(decoder_inputs[:, :-1]) # 64-dim embedding
        
        txt_outputs, gate_outputs, alignments = [], [], []
        decoder_inputs = decoder_inputs.transpose(0, 1).contiguous() # T, B, F
        attention_context = torch.zeros(batch_size, 512)

        if encoder_inputs.is_cuda: attention_context = attention_context.cuda()
        
        while len(txt_outputs) < decoder_inputs.size(0):
            decoder_input = decoder_inputs[len(txt_outputs)]
            txt_output, new_attention_context = self.forward_step(encoder_inputs, decoder_input, attention_context)

            attention_context = new_attention_context
            txt_outputs += [txt_output.squeeze(1)]


        txt_outputs = torch.stack(txt_outputs).transpose(0, 1).contiguous()
        
        # CE 사용할껀데 softmax 필요없지>?        
        #asr_decoder_outputs = self.softmax(asr_decoder_outputs)

        return txt_outputs

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


    