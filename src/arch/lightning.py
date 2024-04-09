# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
# Ligthining
import pytorch_lightning as pl
# DGL
import dgl
from dgl.nn.pytorch import GraphConv
# newly built
from .model import VAE, GNN


# ************************* Variational Autoencoder *************************
class trainerVAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (nn.Sequential): The encoder neural network.
        decoder (nn.Sequential): The decoder neural network.
    """

    def __init__(self, input_dim, latent_dim, enc_hidden_dim_1=100, dec_hidden_dim_1=40,
                 enc_hidden_dim_2=70, dec_hidden_dim_2=40, dropout=0.2, learning_rate=1e-3):
        super().__init__()
        self.encoder, self.decoder = VAE(input_dim, latent_dim,
                                         enc_hidden_dim_1, dec_hidden_dim_1,
                                         enc_hidden_dim_2, dec_hidden_dim_2,
                                         dropout)
        self.learning_rate = learning_rate

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
        x = batch[0]  # Get the input data
        recon_x, mu, log_var = self(x)  # Forward pass: Get reconstruction, mean, and log variance
        loss = self.loss(recon_x, x, mu, log_var)  # compute loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]  # Get the input data
        recon_x, mu, log_var = self(x)
        loss = self.loss(recon_x, x, mu, log_var)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]  # Get the input data
        recon_x, mu, log_var = self(x)
        loss = self.loss(recon_x, x, mu, log_var)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# ************************* Graph Neural Network *************************
class trainerGNN(pl.LightningModule):
    """
    Graph Neural Network class.
    """

    def __init__(self, in_dim=74, gcn_hidden_dim=256, fcn_hidden_dim=256, out_dim=1,
                 n_gcn_layers=2, n_fcn_layers=2, learning_rate=1e-3):
        super().__init__()
        self.model = GNN(in_dim, gcn_hidden_dim, fcn_hidden_dim, out_dim,
                         n_gcn_layers, n_fcn_layers)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''Training step'''
        # the input and pred data
        graph, label = batch
        out = self(graph)
        # loss and accuracy
        loss = self.criterion(out, label.view(-1, 1))
        acc = r2_score(label.view(-1, 1).cpu().detach().numpy(), out.cpu().detach().numpy())
        # log the results
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        '''Validation step'''
        # input and pred data
        graph, label = batch
        out = self(graph)
        # loss and accuracy
        loss = self.criterion(out, label.view(-1, 1))
        acc = r2_score(label.view(-1, 1).cpu().detach().numpy(), out.cpu().detach().numpy())
        # log the results
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        '''Test step'''
        # get the input and pred data
        graph, label = batch
        out = self(graph)
        # compute loss and accuracy
        loss = self.criterion(out, label.view(-1, 1))
        acc = r2_score(label.view(-1, 1).cpu().detach().numpy(), out.cpu().detach().numpy())
        # log the results
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


