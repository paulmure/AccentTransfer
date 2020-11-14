import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavenet_vocoder import WaveNet
from functools import reduce


num_time_samples = 16384 * 2


class GatedConv1dTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1dTranspose, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.pad(x, (self.dilation, 0))
        return torch.mul(self.tanh(self.conv_f(x)), self.sig(self.conv_g(x)))


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedBlock, self).__init__()
        self.gatedconv = GatedConv1dTranspose(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        return self.conv_1(self.gatedconv(x))


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = WaveNet(
                        out_channels=num_time_samples // 256,
                        scalar_input=True)
        
        hs = []
        batch_norms = []

        n_hidden = 128
        for i in range(7):
            rate = 2**i
            hs.append(GatedBlock(int(n_hidden), int(n_hidden // 2), 2, dilation=rate, stride=2))
            batch_norms.append(nn.BatchNorm1d(int(n_hidden // 2)))
            n_hidden /= 2

        self.last_conv_layers = nn.ModuleList(hs)
        self.last_batch_norms = nn.ModuleList(batch_norms)

        self.final_fc = nn.Linear(33985, num_time_samples, bias=False)

        self.relu_1 = nn.ReLU()
        self.conv_1_1 = nn.Conv1d(1, 1, 1)
        self.relu_2 = nn.ReLU()
        self.conv_1_2 = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        x = self.decoder(x)

        for layer, batch_norm in zip(self.last_conv_layers, self.last_batch_norms):
            x = layer(x)
            x = batch_norm(x)
        
        x = self.final_fc(x)
        x = self.relu_1(self.conv_1_1(x))
        x = self.relu_2(self.conv_1_2(x))

        return x


if __name__ == "__main__":
    # random data
    x = torch.randn(1, 1, 256)

    # test decoder
    decoder = Decoder()
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
    print('Sample window:', num_time_samples)
