import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(30, 32, kernel_size=2, stride=1, padding=1),
            nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1),
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4480, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 30, 65))
    x = torch.tensor(x).float()
    # test encoder shape
    encoder = Encoder()
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
