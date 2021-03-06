import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import time
import os
import pickle

from tqdm import tqdm
from math import floor

from models.model import Model
from streaming_dataset import StreamingAccentDataset

device = torch.device('cuda')

training_params = {
    'n_embeddings': 128,
    'learning_rate': 0.001,
    'epochs': 20,
    'batch_size': 4,
    'commitment_cost': 0.25,
    'multitask_scale': 0.25,
    'decoder_final_block': True,
    'quantize': False,
    'device': device,
    'parallel': True,
    'test': True,
    'load_pretrained': False
}


class Trainer():
    def __init__(self,
                 n_embeddings,
                 learning_rate,
                 epochs,
                 batch_size,
                 commitment_cost,
                 multitask_scale,
                 decoder_final_block,
                 quantize,
                 device,
                 parallel,
                 test,
                 load_pretrained):
        self.epochs = epochs
        self.commitment_cost = commitment_cost
        self.multitask_scale = multitask_scale
        self.device = device
        self.quantize = quantize

        dataset = StreamingAccentDataset()
        num_classes = dataset.num_classes
        if test:
            self.epochs = 2
            dataset = torch.utils.data.Subset(dataset, range(2))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        self.model = Model(n_embeddings, num_classes, device, quantize, decoder_final_block, parallel)
        if load_pretrained:
            self.model.load_state_dict(torch.load('saved'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)

        self.multitask_criterion = nn.CrossEntropyLoss()

        self.model.train()
        self.model.to(device)

        self.logs = {f'epoch_{i}': {'loss_recons': [], 'loss_vq': [], 'loss_commit': [], 'loss_multitask': [], 'total_loss': []} \
                        for i in range(1, epochs+1)}

        self.timer = 0

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.time_epoch_start()

            self.optimizer.zero_grad()
            for i, (audios, labels) in enumerate(tqdm(self.dataloader)):

                labels = labels.to(device)
                audios = audios.to(device)

                if self.quantize:
                    x_hat, z_e_x, z_q_x, multitask = self.model(audios)
                else:
                    x_hat, multitask = self.model(audios)

                loss_recons = F.mse_loss(x_hat, audios)
                loss_multitask = self.multitask_criterion(multitask, labels)
                if self.quantize:
                    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

                if self.quantize:
                    total_loss = loss_recons + loss_vq + self.commitment_cost * loss_commit + self.multitask_scale * loss_multitask
                else:
                    total_loss = loss_recons + self.multitask_scale * loss_multitask

                total_loss.backward()

                if i % 10 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if self.quantize:
                    self.log_training_step(loss_recons, loss_vq, loss_commit, loss_multitask, total_loss, epoch)
                else:
                    self.log_training_step_no_quantize(loss_recons, loss_multitask, total_loss, epoch)

            self.optimizer.step()

            self.time_epoch_end(epoch)
            self.log_epoch(epoch)
            self.save_model(epoch)

        self.save_training_log()

    def time_epoch_start(self):
        self.timer = time.time()

    def time_epoch_end(self, epoch):
        runtime = time.time() - self.timer
        print(f'Epoch #{epoch} runtime: {(floor(runtime) // 60):02}:{(floor(runtime) % 60):02}')
    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join('trained_models', f'saved_model_epoch_{epoch}'))

    def log_training_step_no_quantize(self, loss_recons, loss_multitask, total_loss, epoch):
        logger = self.logs[f'epoch_{epoch}']
        logger['loss_recons'].append(loss_recons.detach().cpu())
        logger['loss_multitask'].append(loss_multitask.detach().cpu())
        logger['total_loss'].append(total_loss.detach().cpu())

    def log_training_step(self, loss_recons, loss_vq, loss_commit, loss_multitask, total_loss, epoch):
        logger = self.logs[f'epoch_{epoch}']
        logger['loss_recons'].append(loss_recons.detach().cpu())
        logger['loss_vq'].append(loss_vq.detach().cpu())
        logger['loss_commit'].append(loss_commit.detach().cpu())
        logger['loss_multitask'].append(loss_multitask.detach().cpu())
        logger['total_loss'].append(total_loss.detach().cpu())
    
    def log_epoch(self, epoch):
        logger = self.logs[f'epoch_{epoch}']
        epoch_recon_error = np.mean(logger['loss_recons'])
        epoch_total_loss = np.mean(logger['total_loss'])
        print(f'Epoch #{epoch}, Recon Error:{epoch_recon_error}, Total Loss: {epoch_total_loss}')

    def save_training_log(self):
        with open('training_log', 'wb') as f:
            pickle.dump(self.logs, f)


if __name__ == "__main__":
    trainer = Trainer(**training_params)
    trainer.train()
