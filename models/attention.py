import os
import torch
import matplotlib
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import random

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        
        self.location_conv = nn.Conv1d(2, attention_n_filters,
                                    kernel_size=attention_kernel_size, stride=1,
                                    padding=padding, dilation=1,
                                    bias=False)
        
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)

        return processed_attention

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


class LocationSensitiveAttention(nn.Module):
    """
    Location-Sensitive Attention
    (Location-Sensitive attention from "Attention-Based Models for speech Recognition", 
    which extends the additive attention mechanism "Neural machine translation by jointly learning to align and Translate")
    Inputs: decoder_inputs, encoder_inputs
        - **decoder_inputs** (batch, q_len, d_model)
        - **encoder_inputs** (batch, k_len, d_model)
    """
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(LocationSensitiveAttention, self).__init__()

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

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output (batch, T, a_F)
        memory: encoder outputs (batch, T, e_F)
        processed_memory: processed encoder outputs (batch, T, e_F)
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """        
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
