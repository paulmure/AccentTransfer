import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ConvStack, ResLayer


class RecurrentLayer(nn.Module):
    def __init__(self):
        super(RecurrentLayer, self).__init__()
        self.gru = nn.GRU(input_size=65, hidden_size=65, num_layers=10)

    def forward(self, x):
        x = self.gru(x)[0]
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_stack = ConvStack()
        self.res_layers = nn.ModuleList([ResLayer()] * 3)
        self.final_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11,
                        stride=5, padding=1),
            nn.ReLU(True),
            nn.Conv1d(32, 64,kernel_size=7,
                        stride=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=7,
                        stride=5, padding=1),
            nn.ReLU(True),
        )
        self.recurrent = RecurrentLayer()

    def forward(self, x):
        # Convolution part
        x = self.conv_stack(x)
        for layer in self.res_layers:
            x = layer(x)
        x = self.final_conv(x)    # 8320

        # Recurrent part
        x = x.permute(1, 0, 2)
        x = self.recurrent(x)
        x = x.permute(1, 0, 2)
        
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 1, 44100))
    x = torch.tensor(x).float()
    # test encoder shape
    encoder = Encoder()
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
