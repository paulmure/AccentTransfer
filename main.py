import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.model import Model
from streaming_dataset import StreamingAccentDataset
import sys
import matplotlib.pyplot as plt
import time
import os
from math import floor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

"""
Load data and define batch data loaders
"""
dataset = StreamingAccentDataset()
# x_var will not be available if using the streaming data_loader
# x_train_var = torch.tensor(dataset.xvar)
num_classes = dataset.num_classes 

# subset = torch.utils.data.Subset(dataset, list(range(30)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

# x_train_var.to(device)

"""
Hyperparameters
"""
model_hyperparams = {
    'n_embeddings': 256,
    'num_classes': num_classes,
    'device': device
}

learning_rate = 0.001
epochs = 2
commitment_cost = 0.25
multitask_scale = 0.5

num_time_samples = 16384 * 2

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = Model(**model_hyperparams)

"""
Set up optimizer and training loop
"""

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

model.train()
model.to(device)

def train():
    all_losses = []
    all_recon_errors = []

    for i in range(epochs):
        start_time = time.time()
        recon_errors = []
        loss_vals = []
        multitask_losses = []
        # adversary_losses = []
        perplexities = []
        for audio, mfcc, labels in tqdm(dataloader):
            labels = labels.to(device)
            audio = audio.to(device)
            audio = audio.unsqueeze(1)
            x = audio.to(device)
            optimizer.zero_grad()

            x_hat, z_e_x, z_q_x, multitask = model(x)
            x_hat = x_hat.view(-1, 1, num_time_samples)
            loss_recons = F.mse_loss(x_hat, audio)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss_multitask = nn.CrossEntropyLoss()(multitask, labels)
            # adversary_loss = nn.CrossEntropyLoss()(adversary, labels)

            loss = loss_recons + loss_vq + commitment_cost * loss_commit + multitask_scale * loss_multitask
                #lambda_mul * multitask_loss - lambda_adv * adversary_loss

            loss.backward()
            optimizer.step()

            recon_errors.append(loss_recons.cpu().detach().numpy())
            all_recon_errors.append(loss_recons.cpu().detach().numpy())
            loss_vals.append(loss.cpu().detach().numpy())
            all_losses.append(loss.cpu().detach().numpy())
            multitask_losses.append(loss_multitask.cpu().detach().numpy())
            # adversary_losses.append(adversary_loss.cpu().detach().numpy())

        # logging
        epoch_recon_error = np.mean(recon_errors)
        epoch_loss_vals = np.mean(loss_vals)
        epoch_multitask_loss = np.mean(multitask_losses)
        # epoch_adversary_loss = np.mean(adversary_losses)
        epoch_perplexities = np.mean(perplexities)
        runtime = time.time() - start_time
        print(f'Epoch #{i+1} runtime: {(floor(runtime) // 60):02}:{(floor(runtime) % 60):02}')
        print(f'Epoch #{i+1}, Recon Error:{epoch_recon_error}, Loss: {epoch_loss_vals}, Perplexity: {epoch_perplexities}, \
                    Multitask Loss: {epoch_multitask_loss}') #, Adversary Loss: {epoch_adversary_loss}')
        torch.save(model.state_dict(), os.path.join('trained_models', f'saved_model_epoch_{i+1}'))
    
    plt.plot(all_losses, label='loss')
    plt.legend()
    plt.savefig("training_loss.png")
    plt.clf()
    plt.plot(all_recon_errors, label='recon loss')
    plt.legend()
    plt.savefig("training_recon_error.png")


if __name__ == "__main__":
    train()
