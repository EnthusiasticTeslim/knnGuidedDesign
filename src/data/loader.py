# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch, yaml
from glob import glob
from tbparse import SummaryReader
from pytorch_lightning import LightningDataModule
from multiprocessing import cpu_count
from dgl.dataloading import GraphDataLoader
try:
    from utils import get_selfies_and_smiles_encodings, multiple_selfies_to_hot, get_smiles_selfies  # selfies and smiles functions
    from utils import graph_dataset
except:
    from .utils import get_selfies_and_smiles_encodings, multiple_selfies_to_hot, get_smiles_selfies  # selfies and smiles functions
    from .utils import graph_dataset


#*********************** Load the data into selfies ***********************
def load_vae_data(seed, test_size=0.2, csv_path='../data/data.csv', 
              existing_selfies=False, max_len: int = 0, alphabet: list = [],
              vocab_stoi: dict = {}, vocab_itos: dict = {}):
    # Load the data
    data = pd.read_csv(csv_path)
    # Transform the data into a list of SELFIES and SMILES
    if existing_selfies is not True:
        selfies_list, selfies_alphabet, largest_selfies_len, \
            smiles_list, smiles_alphabet, largest_smiles_len, \
                    vocab_stoi, vocab_itos = get_selfies_and_smiles_encodings(data)
        # One-Hot-Encode the SELFIES into a tensor
        input_one_hot_arr = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)

    else:
        selfies_list, smiles_list = get_smiles_selfies(data)
        input_one_hot_arr = multiple_selfies_to_hot(selfies_list, max_len, alphabet)
        selfies_alphabet = alphabet
        largest_selfies_len = max_len
        vocab_itos = vocab_itos
        vocab_stoi = vocab_stoi

    # split the data into training and validation sets
    x_train, x_test = train_test_split(input_one_hot_arr, test_size=test_size, random_state=seed)
    # split test into test and validation
    x_test, x_val = train_test_split(x_test, test_size=0.5, random_state=seed)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    return x_train, x_val, x_test, selfies_alphabet, largest_selfies_len, vocab_stoi, vocab_itos


# class VAEDataModule(LightningDataModule):
#     '''LightningDataModule for the VAE model
#     refs:
#         https://docs.dgl.ai/en/2.0.x/guide/training-graph.html'''
#     def __init__(self, csv_path, test_size=0.2, batch_size=16):
#         super().__init__()
#         self.train_data, self.test_data, self.val_data = load_graph_data(seed=42, test_size=test_size, csv_path=csv_path)
#         self.batch_size = batch_size

#     def train_dataloader(self):
        
#         return GraphDataLoader(self.train_data, batch_size=self.batch_size,
#                                shuffle=True, num_workers=cpu_count(), persistent_workers=True)
    
#     def val_dataloader(self):
#         return GraphDataLoader(self.val_data, batch_size=self.batch_size, 
#                                  shuffle=False, num_workers=cpu_count(), persistent_workers=True)
    
#     def test_dataloader(self):
#         return GraphDataLoader(self.test_data, batch_size=self.batch_size,
#                                     shuffle=False, num_workers=cpu_count(), persistent_workers=True)


#*********************** Load the data into Graph ***********************
def load_graph_data(seed, test_size=0.2, csv_path='../data/data.csv'):
    # Load the data
    data = pd.read_csv(csv_path)
    smiles = np.asarray(data.smiles, dtype="str")
    prop = np.asarray(data.redox_potential, dtype="float")

    # split into training and testing
    smlstr_train, smlstr_test, \
        property_train, property_test = train_test_split(smiles, prop,
                                                     test_size=test_size,
                                                     random_state=seed)
    # split the test set into test and validation
    smlstr_test, smlstr_val, \
        property_test, property_val = train_test_split(smlstr_test, property_test,
                                                     test_size=0.5,
                                                     random_state=seed)
    # smiles to graph
    train_dataset = graph_dataset(smlstr_train, property_train)
    test_dataset = graph_dataset(smlstr_test, property_test)
    val_dataset = graph_dataset(smlstr_val, property_val)

    return train_dataset, test_dataset, val_dataset


# Define the LightningDataModule for the GNN model
class GraphDataModule(LightningDataModule):
    '''LightningDataModule for the GNN model
    refs:
        https://docs.dgl.ai/en/2.0.x/guide/training-graph.html'''
    def __init__(self, csv_path, test_size=0.2, batch_size=16, seed=42):
        super().__init__()
        self.train_data, self.test_data, self.val_data = load_graph_data(seed=seed, test_size=test_size, csv_path=csv_path)
        self.batch_size = batch_size

    def train_dataloader(self):
        
        return GraphDataLoader(self.train_data, batch_size=self.batch_size,
                               shuffle=True, num_workers=cpu_count(), persistent_workers=True)
    
    def val_dataloader(self):
        return GraphDataLoader(self.val_data, batch_size=self.batch_size, 
                                 shuffle=False, num_workers=cpu_count(), persistent_workers=True)
    
    def test_dataloader(self):
        return GraphDataLoader(self.test_data, batch_size=self.batch_size,
                                    shuffle=False, num_workers=cpu_count(), persistent_workers=True)




#*********************** Load the hyperparameters from logs ***********************
def load_vae_hp_params(path,
                        params=['batch_size', 'dec_hidden_dim_1', 'dec_hidden_dim_2',
                                'dropout', 'enc_hidden_dim_1', 'enc_hidden_dim_2', 'height_dim',
                                'input_dim', 'latent_dim', 'learning_rate', 'max_len',
                                'num_epochs', 'seed', 'split_ratio', 'width_dim',
                                'trn_loss', 'val_loss'],
                        ignore_list=['input_dim', 'latent_dim', 'split_ratio']):
    log_dirs = sorted(glob(path))
    hp_params = {'run': []}
    for pms in params:
        hp_params[pms] = []

    for i, v in enumerate(log_dirs):
        print(f"Processing {v}")
        reader = SummaryReader(log_dirs[i])
        df = reader.scalars

        if len(df) > 2:
            hp_params["run"].append(f"run_{i}")
            with open(f"{log_dirs[i]}/hparams.yaml", "r") as file:
                hparams = yaml.safe_load(file)

            for k, v in hparams.items():
                if k not in ignore_list:
                    hp_params[k].append(v)

            train_loss = df[df.tag == "train_loss_epoch"].value.to_numpy()
            val_loss = df[df.tag == "val_loss_epoch"].value.to_numpy()

            hp_params["trn_loss"].append(np.nanmin(train_loss))
            hp_params["val_loss"].append(np.nanmin(val_loss))
            #hp_params["val_loss"].append(val_loss[np.nanargmin(train_loss)])

    return pd.DataFrame(hp_params)

def load_gnn_hp_params(path,
                        params=['batch_size', 'trn_loss', 'val_loss'],
                        ignore_list=['input_dim', 'latent_dim', 'split_ratio']):
    log_dirs = sorted(glob(path))
    hp_params = {'run': []}
    for pms in params:
        hp_params[pms] = []

    for i, v in enumerate(log_dirs):
        print(f"Processing {v}")
        reader = SummaryReader(log_dirs[i])
        df = reader.scalars

        if len(df) > 2:
            hp_params["run"].append(f"run_{i}")

            with open(f"{log_dirs[i]}/hparams.yaml", "r") as file:
                hparams = yaml.safe_load(file)

            for k, v in hparams.items():
                if k not in ignore_list:
                    hp_params[k].append(v)

            train_loss = df[df.tag == "train_loss"].value.to_numpy()
            val_loss = df[df.tag == "val_loss"].value.to_numpy()
            train_acc = df[df.tag == "train_acc"].value.to_numpy()
            val_acc = df[df.tag == "val_acc"].value.to_numpy()

            hp_params["trn_loss"].append(np.nanmin(train_loss))
            hp_params["val_loss"].append(np.nanmin(val_loss))
            hp_params["trn_acc"].append(np.nanmax(train_acc))
            hp_params['val_acc'].append(np.nanmax(val_acc))
            # hp_params["val_loss"].append(val_loss[np.nanargmin(train_loss)])
            # hp_params["trn_acc"].append(train_acc[np.nanargmin(train_loss)])
            # hp_params['val_acc'].append(val_acc[np.nanargmin(train_loss)])

    return pd.DataFrame(hp_params)