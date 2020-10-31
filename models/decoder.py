import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UpsamplingResidualBlock(nn.Module):
    def __init__(self, dilations):
        super(UpsamplingResidualBlock, self).__init__()

        self.pre_stack_conv = nn.Conv1d(1, 1, kernel_size=1, stride=1, bias=False)

        self.causal_conv = nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=1, bias=False)

        self.filter_convs = nn.ModuleList()
        self.gated_convs = nn.ModuleList()

        for d in dilations:
            self.filter_convs.append(nn.Conv1d(1, 1, kernel_size=2,
                                        stride=1, padding=d//2, dilation=d))
            self.gated_convs.append(nn.Conv1d(1, 1, kernel_size=2,
                                        stride=1, padding=d//2, dilation=d))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.pre_stack_conv(x)
        x = self.causal_conv(x)
        x = x[:, :, :-1]  # causal conv
        residual = x
        filtered = x
        gated = x

        for filter_layer, gated_layer in zip(self.filter_convs, self.gated_convs):
            filtered = filter_layer(filtered)
            gated = gated_layer(gated)

        filtered = torch.tanh(filtered)
        gated = torch.sigmoid(gated)

        x = filtered * gated
        return x + residual

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.res_stack = nn.ModuleList()
        for _ in range(7):
            self.res_stack.append(UpsamplingResidualBlock([2, 4, 8]))

    def forward(self, x):
        for block in self.res_stack:
            x = block(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 1, 256))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder()
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
