import argparse
import optuna
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data_loader import load_graph_data, GraphDataModule
from model import lightiningGNN
from utils import collate


def objective(trial, seed, split):
    # hyperparameter
    hidden_gcn_dim = trial.suggest_categorical('hidden_gcn_size', [16, 32, 64, 128, 256])
    hidden_fcn_dim = trial.suggest_categorical('hidden_fcn_size', [16, 32, 64, 128, 256])
    n_gcn_layers = trial.suggest_int('n_gcn_layers', 1, 5, step=1)
    n_fcn_layers = trial.suggest_int('n_fcn_layers', 1, 5, step=1)
    num_epochs = trial.suggest_categorical('num_epochs', [150, 200, 250, 300, 500, 1000])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    # set seed for reproducibility
    pl.seed_everything(seed=seed)
    # Load the data
    train_data, test_data, val_data = load_graph_data(seed=seed, test_size=split, csv_path='../data/data.csv')
    # data loader
    data_module = GraphDataModule(train=train_data, test=test_data, val=val_data, batch_size=batch_size)
    # Initialize the model
    model = lightiningGNN(in_dim=74, gcn_hidden_dim=hidden_gcn_dim, fcn_hidden_dim=hidden_fcn_dim, out_dim=1,
                          n_gcn_layers=n_gcn_layers, n_fcn_layers=n_fcn_layers, learning_rate=learning_rate)
    print(f'model: \n{model}')
    # Define the model callbacks
    checkpoint_call_back = ModelCheckpoint(
                                            dirpath=f"../reports/GNNcheckpoints/{trial.number}",
                                            filename="best-chckpt",
                                            save_top_k=1,
                                            verbose=True,
                                            monitor="val_loss",
                                            mode="min"
                                        )
    
    logger = TensorBoardLogger(save_dir="../reports/GNNlightning_logs", name="GNN")

    trainer = pl.Trainer(logger=logger,
                         callbacks=checkpoint_call_back,
                         max_epochs=num_epochs,
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
                            "batch_size": batch_size,
                            "device":trainer.accelerator})
    
    trainer.fit(model, data_module)
    
    return trainer.callback_metrics["val_loss"].item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for GNN model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, default=0.2)
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.seed, args.split), n_trials=args.n_trials)
    
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)