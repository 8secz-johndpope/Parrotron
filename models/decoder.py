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


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()

        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim)

        self.v = nn.Linear(attention_dim, 1)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Decoder(nn.Module):
    def __init__(self, target_dim, pre_net_dim, rnn_hidden_size, second_rnn_hidden_size, postnet_hidden_size, n_layers, dropout, attention_type):
        super(Decoder, self).__init__()

        self.pre_net = Prenet(target_dim, pre_net_dim, dropout)

        self.attention_layer = DotProductAttention()

        '''
        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        '''

        self.projection_layer = nn.Linear(1024, target_dim, False)

        self.attention_rnn = nn.LSTM(
            input_size=pre_net_dim+512,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            dropout=0
        )

        self.rnn_projection = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU()
        )

        self.postnet = Postnet(target_dim, postnet_hidden_size, dropout, int((5 - 1) / 2)) # 5 = kernel size

    def forward_step(self, encoder_inputs, decoder_input, attention_context):
        '''
        output : (mel_output, new attention context vector)
        '''
        
        attention_context = attention_context.squeeze()
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
        decoder_inputs = [batch, feature, seq_len] ex) torch.Size([2, 440, 1025])
        '''

        #print(decoder_inputs.shape)
        batch_size, _, _ = decoder_inputs.size()
        
        go_frame = torch.zeros(decoder_inputs.shape[0], 1, decoder_inputs.shape[2])

        if encoder_inputs.is_cuda: go_frame = go_frame.cuda()
        
        decoder_inputs = torch.cat((go_frame, decoder_inputs), dim=1)
        decoder_inputs = self.pre_net(decoder_inputs) # B, T, F (prenet output)

        mel_outputs, gate_outputs, alignments = [], [], []

        decoder_inputs = decoder_inputs.transpose(0, 1).contiguous() # T, B, F

        attention_context = torch.zeros(batch_size, 512)
        if encoder_inputs.is_cuda: attention_context = attention_context.cuda()
        
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
 
            mel_output, new_attention_context = self.forward_step(encoder_inputs, decoder_input, attention_context)
            
            attention_context = new_attention_context
            mel_outputs += [mel_output.squeeze(1)]

        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        final_mel_outputs = mel_outputs + mel_outputs_postnet.transpose(1, 2)
        
        return final_mel_outputs

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

   

    