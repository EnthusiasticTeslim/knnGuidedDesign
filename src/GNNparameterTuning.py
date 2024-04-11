# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""

import argparse
import optuna
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data.loader import GraphDataModule
from arch.lightning import trainerGNN


def objective(trial, split: int, data_path: str = '../data/data.csv', save_dir: str='../reports'):
    # hyperparameters
    seed = trial.suggest_categorical('seed', [0, 42, 104, 4296])
    hidden_gcn_dim = trial.suggest_categorical('hidden_gcn_size', [64, 128, 256, 512])
    hidden_fcn_dim = trial.suggest_categorical('hidden_fcn_size', [64, 128, 256, 512])
    n_gcn_layers = trial.suggest_int('n_gcn_layers', 1, 10, step=1)
    n_fcn_layers = trial.suggest_int('n_fcn_layers', 1, 10, step=1)
    num_epochs = trial.suggest_categorical('num_epochs', [100, 200, 300, 500])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    # set seed for reproducibility
    pl.seed_everything(seed=seed)
    # Load the data
    data_module = GraphDataModule(csv_path=data_path, test_size=split, batch_size=batch_size, seed=seed)
    # Initialize the model
    model = trainerGNN(in_dim=74, gcn_hidden_dim=hidden_gcn_dim, fcn_hidden_dim=hidden_fcn_dim, out_dim=1,
                          n_gcn_layers=n_gcn_layers, n_fcn_layers=n_fcn_layers, learning_rate=learning_rate)
    # Define the model callbacks
    checkpoint_call_back = ModelCheckpoint(
                                            dirpath=f"{save_dir}/GNN/hyper_search_checkpoints/{trial.number}",
                                            filename="best-chckpt",
                                            save_top_k=1,
                                            verbose=True,
                                            monitor="val_loss",
                                            mode="min"
                                        )
    
    logger = TensorBoardLogger(save_dir=f"{save_dir}/GNN/hyper_search_logs", name="knnMoleculeGNN")
    # constraint to cpu/cuda due to dgl compatibility (https://discuss.dgl.ai/t/dgl-with-mps-device/4238)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(logger=logger,
                         callbacks=checkpoint_call_back,
                         max_epochs=num_epochs, accelerator=device,
                         deterministic=True,
                         enable_progress_bar=True)
    
    model.save_hyperparameters({"hidden_gcn_dim": hidden_gcn_dim,
                                "hidden_fcn_dim": hidden_fcn_dim,
                                "n_gcn_layers": n_gcn_layers,
                                "n_fcn_layers": n_fcn_layers,
                                "learning_rate": learning_rate,
                                "num_epochs": num_epochs,
                                "seed": seed,
                                "split": split,
                                "batch_size": batch_size})
    
    trainer.fit(model, data_module)
    
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for GNN model")
    parser.add_argument("--split", type=float, default=0.3, help="Split ratio")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--data_path", type=str, default='../data/data.csv', help="Path to data")
    parser.add_argument("--save_dir", type=str, default='../reports', help="path to save directory")
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial=trial, split=args.split, data_path=args.data_path, save_dir=args.save_dir), n_trials=args.n_trials)
    
    print("Best hyperparameters: ", study.best_params)
    print("Best value: ", study.best_value)
