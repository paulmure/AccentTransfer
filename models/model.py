import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .quantizer import VQEmbedding
from .decoder import Decoder

num_time_samples = 16384 * 2


class Multitask(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Multitask, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, n_embeddings, num_classes, device, quantize=True, decoder_final_block=True, parallel=False):
        super(Model, self).__init__()
        self.encoder = Encoder(num_time_samples, device)

        self.quantize = quantize
        if quantize:
            self.codebook = VQEmbedding(
                n_embeddings, 256)
        
        self.multitask = Multitask(32, num_classes)

        self.decoder = Decoder(decoder_final_block)

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.multitask = nn.DataParallel(self.multitask)
            self.decoder = nn.DataParallel(self.decoder)
            if quantize:
                self.codebook = nn.DataParallel(self.codebook)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_e_x = z_e_x.squeeze(2)

        if self.quantize:
            z_q_x_st, z_q_x = self.codebook(z_e_x)
            z_q_x_st = z_q_x_st.unsqueeze(1)
        else:
            z_q_x_st = z_e_x.unsqueeze(1)

        multitask = self.multitask(z_q_x_st[:, :, :32])

        x_hat = self.decoder(z_q_x_st)

        if self.quantize:
            return x_hat, z_e_x, z_q_x, multitask
        else:
            return x_hat, multitask


if __name__ == "__main__":
    x = torch.randn(1, 1, num_time_samples)
    model = Model(150, 44, torch.device('cpu'), quantize=False)
    x_hat, multitask = model(x)
    print('x_hat shape:', x_hat.shape)
    # print('z_e_x shape:', z_e_x.shape)
    # print('z_q_x shape:', z_q_x.shape)
    print('Multitask shape:', multitask.shape)
