import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, batch_norm_momentum=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d( n_outputs, momentum=batch_norm_momentum )
        #self.dropout1 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout2d(dropout)  # turn off entire channel

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d( n_outputs, momentum=batch_norm_momentum )
        #self.dropout2 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout2d(dropout)  # turn off entire channel

        self.net = nn.Sequential(self.conv1, self.chomp1, self.batchnorm1, self.dropout1, self.relu1,
                                 self.conv2, self.chomp2, self.batchnorm2, self.dropout2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        #self.conv1.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)
        #if self.downsample is not None:
        #    self.downsample.weight.data.normal_(0, 0.01)

        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, batch_norm_momentum=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []

        dilation_size = 1
        in_channels = num_inputs
        out_channels = num_channels[0]
        layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                 padding=(kernel_size - 1) * dilation_size, dropout=0.0, batch_norm_momentum=batch_norm_momentum)]

        num_levels = len(num_channels)
        for i in range(1, num_levels):
            dilation_size = 2 ** int(0.34 * i)
            in_channels = num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, batch_norm_momentum=batch_norm_momentum)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNDenseOutput(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, batch_norm_momentum):
        super(TCNDenseOutput, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, batch_norm_momentum=batch_norm_momentum)
        self.linear = nn.Linear(num_channels[-1], output_size)

        self.latent_size = num_channels[-1]

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(x)  # input should have dimension (N, C, L)
        out = self.linear(out[:, :, -1])
        #proba = F.log_softmax(o, dim=1)
        return out

    def get_embeddings(self, x):
        out = self.tcn(x)
        out = out[:, :, -1]
        return out