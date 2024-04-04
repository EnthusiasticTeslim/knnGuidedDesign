import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def model(input_dim, latent_dim):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (nn.Sequential): The encoder neural network.
        decoder (nn.Sequential): The decoder neural network.
    ref:"""

    encoder = nn.Sequential(
                nn.Linear(input_dim, 100),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(100, 70),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(70, latent_dim*2)) # Two times latent_dim for mean and log_var
    
    decoder = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, 40),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(40, 70),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(70, input_dim)) # Sigmoid for output in [0, 1] range
    
    return encoder, decoder


class lightiningVAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (nn.Sequential): The encoder neural network.
        decoder (nn.Sequential): The decoder neural network.
    ref:
        https://github.com/Imfinethankyou1/TADF-likeness/blob/master/model.py
    """
    def __init__(self, input_dim, latent_dim):
        super(lightiningVAE, self).__init__()
        self.encoder, self.decoder = model(input_dim, latent_dim)
        

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for the VAE.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decoded output, mean, and log variance of the latent space.
        """
        # Encoding
        enc_output = self.encoder(x)
        mu, log_var = enc_output.chunk(2, dim=-1)
        # Reparameterization
        z = self.reparameterize(mu, log_var)
        # Decoding
        dec_output = self.decoder(z)
        return dec_output, mu, log_var

    def loss(self, recon_x, x, mu, log_var):
        """
        Calculate the VAE loss.

        Args:
            recon_x (torch.Tensor): Reconstructed data.
            x (torch.Tensor): Original input data.
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: VAE loss.
        """
        reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence
    

    def training_step(self, batch, batch_idx):
        x = batch[0] # Get the input data
        recon_x, mu, log_var = self(x) # Forward pass: Get reconstruction, mean, and log variance
        loss = self.loss(recon_x, x, mu, log_var) # compute loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0] # Get the input data
        recon_x, mu, log_var = self(x)
        loss = self.loss(recon_x, x, mu, log_var)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0] # Get the input data
        recon_x, mu, log_var = self(x)
        loss = self.loss(recon_x, x, mu, log_var)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


