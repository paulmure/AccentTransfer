import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models.model import Model
from data_set import AccentDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

"""
Load data and define batch data loaders
"""
dataset = AccentDataset('audio_data', 'class_label_names')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

x_train_var = torch.tensor(dataset.xvar)
x_train_var.to(device)

"""
Hyperparameters
"""
num_embeddings = 150
num_classes = len(dataset.idx_to_labels)
beta = 1
learning_rate = 0.001
epochs = 10

lambda_mul = 1
lambda_adv = 1


"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = Model(num_embeddings, num_classes, beta)
"""
Set up optimizer and training loop
"""

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

model.train()
model.to(device)


def train():

    for i in range(epochs):
        recon_errors = []
        loss_vals = []
        multitask_losses = []
        adversary_losses = []
        perplexities = []
        for x, labels in tqdm(dataloader):
            labels = labels.to(device)
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, multitask, adversary, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var

            multitask_loss = nn.CrossEntropyLoss()(multitask, labels)
            adversary_loss = nn.CrossEntropyLoss()(adversary, labels)

            loss = recon_loss + embedding_loss + \
                lambda_mul * multitask_loss - lambda_adv * adversary_loss

            loss.backward()
            optimizer.step()

            recon_errors.append(recon_loss.cpu().detach().numpy())
            perplexities.append(perplexity.cpu().detach().numpy())
            loss_vals.append(loss.cpu().detach().numpy())
            multitask_losses.append(multitask_loss.cpu().detach().numpy())
            adversary_losses.append(adversary_loss.cpu().detach().numpy())

        # logging
        epoch_recon_error = np.mean(recon_errors)
        epoch_loss_vals = np.mean(loss_vals)
        epoch_multitask_loss = np.mean(multitask_losses)
        epoch_adversary_loss = np.mean(adversary_losses)
        epoch_perplexities = np.mean(perplexities)
        print(f'Epoch #{i}, Recon Error:{epoch_recon_error}, Loss: {epoch_loss_vals}, Perplexity: {epoch_perplexities}, \
                    Multitask Loss: {epoch_multitask_loss}, Adversary Loss: {epoch_adversary_loss}')


if __name__ == "__main__":
    train()
