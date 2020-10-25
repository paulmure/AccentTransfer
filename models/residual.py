import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvStack(nn.Module):
    """
    ResNet used in encoder
    input shape = (44100,)
    """

    def __init__(self):
        super(ConvStack, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7,
                      stride=3, padding=1),      # 14700, 8
            nn.ReLU(True),
            nn.Conv1d(8, 16, kernel_size=7,
                      stride=3),                 # 4898, 16
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_stack(x)


class ResLayer(nn.Module):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.res_stack = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,
                        stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(32, 16, kernel_size=3,
                        stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x + self.res_stack(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 1, 44100))
    x = torch.tensor(x).float()
    # test conv layer
    conv = ConvStack()
    conv_out = conv(x)
    print('Conv Layer out shape:', conv_out.shape)
    # test res layer
    res = ResLayer()
    res_out = res(conv_out)
    print('Res Layer out shape:', res_out.shape)
