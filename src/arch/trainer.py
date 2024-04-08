import argparse
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ..data.loader import load_data, GraphDataModule
from ..data.utils import flatten
from .lightning import trainerVAE, trainerGNN

class BaseTrainer:
    def __init__(self, seed=2020, split=0.3):
        self.seed = seed
        self.split = split

    def trainGNN(self, hidden_gcn_dim=32, hidden_fcn_dim= 32,
                 n_gcn_layers=32, n_fcn_layers= 32,
                 num_epochs=1000, learning_rate=1e-3, batch_size=16, data_path='../data/data.csv'):
        # set seed for reproducibility
        pl.seed_everything(seed=self.seed)
        # Load the data
        data_module = GraphDataModule(csv_path=data_path, test_size=self.split, batch_size=batch_size)
        # Initialize the model
        model = trainerGNN(in_dim=74, gcn_hidden_dim=hidden_gcn_dim, fcn_hidden_dim=hidden_fcn_dim, out_dim=1,
                           n_gcn_layers=n_gcn_layers, n_fcn_layers=n_fcn_layers, learning_rate=learning_rate)
        # Define the model callbacks
        checkpoint_call_back = ModelCheckpoint(
            dirpath=f"../reports/GNNcheckpoints_{self.seed}", filename="best-chckpt",
            save_top_k=1, verbose=True, monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=f"../reports/GNNlogs_{self.seed}", name="GNN")
        # constraint to cpu/cuda due to dgl compatibility (https://discuss.dgl.ai/t/dgl-with-mps-device/4238)
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(logger=logger,
                             callbacks=checkpoint_call_back,
                             max_epochs=num_epochs, accelerator=device,
                             deterministic=True,
                             enable_progress_bar=True)

        trainer.fit(model, data_module)

        return model

    def trainVAE(self, enc_hidden_dim_1=32, dec_hidden_dim_1= 32,
                 enc_hidden_dim_2=32, dec_hidden_dim_2= 32, dropout=0.5,
                 num_epochs=1000, learning_rate=1e-3, batch_size=128, data_path='../data/data.csv'):
        # set seed for reproducibility
        pl.seed_everything(seed=self.seed)
        # Load the data
        x_train, x_val = load_data(seed=self.seed, test_size=self.split, csv_path=data_path)
        # Flatten the input data using the custom 'flatten' function
        width_train, height_train, input_dim_train, flattened_dataset_train = flatten(x_train)
        width_val, height_val, input_dim_val, flattened_dataset_val = flatten(x_val)
        # ensure that the input dimensions are the same
        assert height_train == height_val, "Height dimensions are not the same"
        assert width_train == width_val, "Width dimensions are not the same"
        assert input_dim_train == input_dim_val, "Input dimensions are not the same"
        # Define hyperparameters
        train_loader = DataLoader(TensorDataset(flattened_dataset_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(flattened_dataset_val), batch_size=batch_size, shuffle=True)
        # Initialize the model
        latent_dim = 32  # Dimensionality of the latent space
        model = trainerVAE(input_dim_train, latent_dim,
                              enc_hidden_dim_1=enc_hidden_dim_1,
                              dec_hidden_dim_1=dec_hidden_dim_1,
                              enc_hidden_dim_2=enc_hidden_dim_2,
                              dec_hidden_dim_2=dec_hidden_dim_2,
                              dropout=dropout, learning_rate=learning_rate)
        # Define the model callbacks
        checkpoint_call_back = ModelCheckpoint(dirpath=f"../reports/VAEcheckpoints_{self.seed}",
                                               filename="best-chckpt", save_top_k=1,
                                               verbose=True, monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=f"../reports/VAElogs_{self.seed}", name="knnMoleculeVAE")

        trainer = pl.Trainer(logger=logger,
                             callbacks=checkpoint_call_back,
                             max_epochs=num_epochs,
                             deterministic=True,
                             enable_progress_bar=True)

        trainer.fit(model, train_loader, val_loader)

        return model


if __name__ == "__main__":
    pass
