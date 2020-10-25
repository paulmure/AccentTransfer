import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResLayer


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=11, stride=5, padding=2, dilation=2, output_padding=1),
            nn.ReLU()
        )
        self.res_layesrs = nn.ModuleList([ResLayer()] * 3)
        self.final_conv_transpose = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=7, stride=3, padding=2, dilation=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.inverse_conv_stack(x)
        for layer in self.res_layesrs:
            x = layer(x)
        x = self.final_conv_transpose(x)
        return x.view(x.shape[0], 1, -1)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 65, 128))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder()
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
