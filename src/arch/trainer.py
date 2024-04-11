import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ..data.loader import load_vae_data, GraphDataModule
from ..data.utils import flatten
from .lightning import trainerVAE, trainerGNN

class BaseTrainer:
    def __init__(self, seed=2020, split=0.3, save_path='../reports'):
        self.seed = seed
        self.split = split
        self.report_path = save_path

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
            dirpath=f"{self.report_path}/GNN/checkpts_{self.seed}", filename="best-chckpt",
            save_top_k=1, verbose=True, monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=f"{self.report_path}/GNN/logs_{self.seed}", name="GNN")
        # constraint to cpu/cuda due to dgl compatibility (https://discuss.dgl.ai/t/dgl-with-mps-device/4238)
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(logger=logger,
                             callbacks=checkpoint_call_back,
                             max_epochs=num_epochs, accelerator=device,
                             deterministic=True,
                             enable_progress_bar=True)

        trainer.fit(model, data_module)

        return model

    def pretrainVAE(self, enc_hidden_dim_1=32, dec_hidden_dim_1= 32,
                 enc_hidden_dim_2=32, dec_hidden_dim_2= 32, dropout=0.5,
                 num_epochs=1000, learning_rate=1e-3, batch_size=128, data_path='../data/data.csv'):
        # set seed for reproducibility
        pl.seed_everything(seed=self.seed)
        # Load the data
        x_train, x_val, x_test, selfies_alphabet, \
                    largest_selfies_len, vocab_stoi, vocab_itos = load_vae_data(seed=self.seed, test_size=self.split, csv_path=data_path)
        
        width_train, height_train, input_dim_train, flattened_dataset_train = flatten(x_train)
        width_val, height_val, input_dim_val, flattened_dataset_val = flatten(x_val)
        width_test, height_test, input_dim_test, flattened_dataset_test = flatten(x_test)
        
        assert height_train == height_val == height_test, "Height dimensions are not the same"
        assert width_train == width_val == width_test, "Width dimensions are not the same"
        assert input_dim_train == input_dim_val == input_dim_test, "Input dimensions are not the same"
        
        self.pretrain_loader = DataLoader(TensorDataset(flattened_dataset_train), batch_size=batch_size, shuffle=True)
        self.preval_loader = DataLoader(TensorDataset(flattened_dataset_val), batch_size=batch_size, shuffle=True)
        self.pretest_loader = DataLoader(TensorDataset(flattened_dataset_test), batch_size=batch_size, shuffle=True)
        self.selfies = {'dim': input_dim_train, 'max_len': largest_selfies_len, 'alphabet': selfies_alphabet}
        
        model = trainerVAE(input_dim=self.selfies['dim'], latent_dim=32,
                              enc_hidden_dim_1=enc_hidden_dim_1,
                              dec_hidden_dim_1=dec_hidden_dim_1,
                              enc_hidden_dim_2=enc_hidden_dim_2,
                              dec_hidden_dim_2=dec_hidden_dim_2,
                              dropout=dropout, learning_rate=learning_rate)
        # Define the model callbacks
        checkpoint_call_back = ModelCheckpoint(dirpath=f"{self.report_path}/VAE/pt_checkpts_{self.seed}",
                                               filename="best-chckpt", save_top_k=1,
                                               verbose=True, monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=f"{self.report_path}/VAE/pt_logs_{self.seed}", name="knnMoleculeVAE")

        trainer = pl.Trainer(logger=logger,
                             callbacks=checkpoint_call_back,
                             max_epochs=num_epochs,
                             deterministic=True,
                             enable_progress_bar=True)

        trainer.fit(model, self.pretrain_loader, self.preval_loader)

        return model

    def load_model(self, model_path):
        model = torch.load(model_path)
        return model
    
    def fineTuneVAE(self, model_info:dict, random_weight=False, model=None,
                    num_epochs=1000, batch_size=128, 
                    data_path='./data/data.csv'):
        # set seed for reproducibility
        pl.seed_everything(seed=self.seed)

        # set model
        if random_weight:
            model = trainerVAE(**model_info)
            pretrained_model = model.load_state_dict(torch.load(f'/Users/gbemidebe/Documents/GitHub/knnGuidedDesign/reports/VAE/pt_checkpts_{self.seed}/best-chckpt.ckpt')['state_dict'])
        else:
            pretrained_model = model

        # freeze encoder layers
        for param in pretrained_model.encoder.parameters():
            param.requires_grad = False

        # Load the data
        x_train, x_val, x_test, _, _ = load_vae_data(seed=self.seed, 
                                        test_size=self.split, csv_path=data_path,
                                        existing_selfies=True, max_len = self.selfies['max_len'], 
                                        alphabet = self.selfies['alphabet'])
        
        width_train, height_train, input_dim_train, flattened_dataset_train = flatten(x_train)
        width_val, height_val, input_dim_val, flattened_dataset_val = flatten(x_val)
        width_test, height_test, input_dim_test, flattened_dataset_test = flatten(x_test)
        
        assert height_train == height_val == height_test, "Height dimensions are not the same"
        assert width_train == width_val == width_test, "Width dimensions are not the same"
        assert input_dim_train == input_dim_val == input_dim_test, "Input dimensions are not the same"
        
        self.finetune_train_loader = DataLoader(TensorDataset(flattened_dataset_train), batch_size=batch_size, shuffle=True)
        self.finetune_val_loader = DataLoader(TensorDataset(flattened_dataset_val), batch_size=batch_size, shuffle=True)
        self.finetune_test_loader = DataLoader(TensorDataset(flattened_dataset_test), batch_size=batch_size, shuffle=True)
        

        # Define the model callbacks
        checkpoint_call_back = ModelCheckpoint(dirpath=f"{self.report_path}/ft_checkpts_{self.seed}",
                                               filename="best-chckpt", save_top_k=1,
                                               verbose=True, monitor="val_loss", mode="min")

        logger = TensorBoardLogger(save_dir=f"{self.report_path}/ft_logs_{self.seed}", name="knnMoleculeVAE")

        trainer = pl.Trainer(logger=logger,
                             callbacks=checkpoint_call_back,
                             max_epochs=num_epochs,
                             deterministic=True,
                             enable_progress_bar=True)

        trainer.fit(model, self.finetune_train_loader, self.finetune_val_loader)

        return model

if __name__ == "__main__":
    args = argparse.ArgumentParser('description=Train the VAE and GNN models')
    args.add_argument('-seed', type=int, default=2020, help='Seed for reproducibility')
    args.add_argument('-split', type=float, default=0.3, help='Train-Test split')
    args.add_argument('-save_path', type=str, default='../reports', help='Path to save the model')

    args = args.parse_args()

    trainer = BaseTrainer(seed=args.seed, split=args.split, save_path=args.save_path)

