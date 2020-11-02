import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .quantizer import VectorQuantizer
from .decoder import Decoder


class Multitask(torch.nn.Module):
    def __init__(self, num_classes):
        super(Multitask, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
<<<<<<< HEAD
            nn.Linear(128, num_classes),
=======
            nn.Linear(4160, num_classes),
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985
        )

    def forward(self, x):
        return self.model(x)


class Adversary(nn.Module):
    def __init__(self, num_classes):
        super(Adversary, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
<<<<<<< HEAD
            nn.Linear(128, 256),
=======
            nn.Linear(4160, 256),
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, n_embeddings, num_classes, beta, save_embedding_map=False):
        super(Model, self).__init__()
        self.encoder = Encoder()
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, 256, beta)
        
<<<<<<< HEAD
        self.multitask = Multitask(num_classes)
        self.adversary = Adversary(num_classes)
=======
        # self.multitask = Multitask(num_classes)
        # self.adversary = Adversary(num_classes)
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985

        # decode the discrete latent representation
        self.decoder = Decoder()

        if save_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        z_q = z_q.permute(0, 2, 1)
<<<<<<< HEAD

        multitask = self.multitask(z_q[:, :, :128])
        adversary = self.adversary(z_q[:, :, 128:])

=======
        
        # multitask = self.multitask(z_e[:, :64, :])
        # adversary = self.adversary(z_e[:, 64:, :])
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

<<<<<<< HEAD
        return embedding_loss, x_hat, perplexity, multitask, adversary
=======
        return embedding_loss, x_hat, perplexity
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 30, 65))
<<<<<<< HEAD
    x = torch.cuda.FloatTensor(x)
    x.to('cuda')

    # test decoder
    model = Model(150, 44, 1)
    model.to('cuda')
=======
    x = torch.tensor(x).float()

    # test decoder
    model = Model(150, 44, 1)
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985
    model_out = model(x)
    print('Dncoder out shape:', model_out[1].shape)

