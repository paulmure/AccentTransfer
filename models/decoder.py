import torch
import torch.nn as nn
from .wavenet_vocoder import WaveNet


num_time_samples = 16384 * 2

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = WaveNet(
                        out_channels=num_time_samples // 256,
                        scalar_input=True)

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    # random data
    x = torch.randn(1, 1, 256)

    # test decoder
    decoder = Decoder()
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
    reshaped = decoder_out.view(-1, 1, num_time_samples)
    print(reshaped.shape)
