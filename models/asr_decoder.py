import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.attention import LocationSensitiveAttention

class ASR_Decoder(nn.Module):
    def __init__(self,label_dim, embedding_dim, encoder_dim, rnn_hidden_size, second_rnn_hidden_size, 
                 attention_dim, attention_filter_n, attention_filter_len, sos_id, eos_id, pad_id):
        super(ASR_Decoder, self).__init__()

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.second_rnn_hidden_size = second_rnn_hidden_size
        self.attention_dim = attention_dim

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(self.label_dim, self.embedding_dim)

        self.attention_rnn = nn.LSTMCell(self.embedding_dim + self.rnn_hidden_size, self.rnn_hidden_size)
        self.attention_layer = LocationSensitiveAttention(self.rnn_hidden_size, self.encoder_dim, self.attention_dim, attention_filter_n, attention_filter_len)
        self.decoder_rnn = nn.LSTMCell(self.rnn_hidden_size, self.second_rnn_hidden_size)

        self.memory_layer = nn.Linear(in_features=self.encoder_dim, out_features=self.attention_dim, bias=True)
        self.projection_layer = nn.Linear(self.second_rnn_hidden_size, self.label_dim, False)


    def forward_step(self, encoder_inputs, decoder_input):
        
        cell_input_1 = torch.cat((decoder_input, self.attention_context), -1) # pre-net and attention context vector concatenated
        
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input_1, (self.attention_hidden, self.attention_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, encoder_inputs, self.memory, attention_weights_cat)
    
        self.attention_weights_cum += self.attention_weights
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(self.attention_context, (self.decoder_hidden, self.decoder_cell))
        
        decoder_output = self.projection_layer(self.decoder_hidden) # projected through a linear transform
        
        return decoder_output, self.attention_weights

    def forward(self, encoder_inputs, decoder_inputs):
        """
        ASR Decoder forward pass for training
        PARAMS
        ------
        encoder_inputs: Encoder outputs (B, T, e_F)
        decoder_inputs: Decoder inputs for teacher forcing. (B, decoder_T)
        """
        #init part
        batch_size, max_len = decoder_inputs.size()
        _, encoder_max_len, _ = encoder_inputs.size()

        sos_frame = torch.LongTensor([self.sos_id]*batch_size).view(batch_size, 1)     
        if encoder_inputs.is_cuda: sos_frame = sos_frame.cuda()

        self.attention_hidden = torch.zeros(batch_size, self.rnn_hidden_size) # (n_layer, bathch sise, hidden_size)
        self.attention_cell = torch.zeros(batch_size, self.rnn_hidden_size)

        self.decoder_hidden = torch.zeros(batch_size, self.second_rnn_hidden_size)
        self.decoder_cell = torch.zeros(batch_size, self.second_rnn_hidden_size)

        self.attention_context = torch.zeros(batch_size, self.rnn_hidden_size)

        self.attention_weights_cum = torch.zeros(batch_size, encoder_max_len)
        self.attention_weights = torch.zeros(batch_size, encoder_max_len)

        self.memory = self.memory_layer(encoder_inputs)

        if encoder_inputs.is_cuda:
            self.attention_context = self.attention_context.cuda()
            self.attention_cell = self.attention_cell.cuda()
            self.attention_hidden = self.attention_hidden.cuda()

            self.decoder_hidden = self.decoder_hidden.cuda()
            self.decoder_cell = self.decoder_cell.cuda()

            self.attention_weights_cum = self.attention_weights_cum.cuda()
            self.attention_weights = self.attention_weights.cuda()

        decoder_inputs = torch.cat((sos_frame, decoder_inputs), dim=1)
        decoder_inputs = self.embedding(decoder_inputs) # (batch, T, embedding dim)
        decoder_inputs = decoder_inputs.transpose(0, 1).contiguous() # (T, batch, embedding dim)

        txt_outputs, alignments = [], []
        
        while len(txt_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(txt_outputs)]
            txt_output, attention_weights = self.forward_step(encoder_inputs, decoder_input)
            
            txt_outputs += [txt_output.squeeze(1)]
            alignments += [attention_weights]

        txt_outputs = torch.stack(txt_outputs).transpose(0, 1).contiguous()
        
        # CE 사용할껀데 softmax 필요없지>?        
        #asr_decoder_outputs = self.softmax(asr_decoder_outputs)

        return txt_outputs
    