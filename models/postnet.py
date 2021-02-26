import os
import torch
import matplotlib
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import random

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
            self.conv_1,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            self.conv_2,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            self.conv_3,
            nn.BatchNorm1d(filter_num),
            nn.Tanh(),
            self.conv_4,
            nn.BatchNorm1d(target_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        
        x = x.transpose(1, 2).contiguous()
        x = self.total_conv(x)
        
        return x
