import argparse
import optuna
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from data.loader import load_vae_data
from data.utils import flatten
from arch.lightning import trainerVAE


def objective(trial, split: float=0.3, data_path: str = '../data/data.csv', save_dir: str='../reports'):
    seed = trial.suggest_categorical('seed', [42, 104, 500, 1994, 2050])
    enc_hidden_dim_1 = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    dec_hidden_dim_1 = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    enc_hidden_dim_2 = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    dec_hidden_dim_2 = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    num_epochs = trial.suggest_categorical('num_epochs', [150, 200, 250, 300, 500, 1000])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    # set seed for reproducibility
    pl.seed_everything(seed=seed)
    # Load the data
    x_train, x_val, x_test = load_vae_data(seed=seed, test_size=split, csv_path=data_path)
    # Flatten the input data using the custom 'flatten' function
    # This function takes a 3D tensor 'x_train' and reshapes it into a 2D tensor
    width_train, height_train, input_dim_train, flattened_dataset_train = flatten(x_train)
    width_val, height_val, input_dim_val, flattened_dataset_val = flatten(x_val)
    width_test, height_test, input_dim_test, flattened_dataset_test = flatten(x_test)
    # ensure that the input dimensions are the same
    assert height_train == height_val == height_test, "Height dimensions are not the same"
    assert width_train == width_val == width_test, "Width dimensions are not the same"
    assert input_dim_train == input_dim_val == input_dim_test, "Input dimensions are not the same"
    # Define hyperparameters
    train_loader = DataLoader(TensorDataset(flattened_dataset_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(flattened_dataset_val), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(flattened_dataset_test), batch_size=batch_size, shuffle=True)
    # Initialize the model
    latent_dim = 32  # Dimensionality of the latent space
    model = trainerVAE(input_dim_train, latent_dim,
                          enc_hidden_dim_1=enc_hidden_dim_1,
                          dec_hidden_dim_1=dec_hidden_dim_1,
                          enc_hidden_dim_2=enc_hidden_dim_2,
                          dec_hidden_dim_2=dec_hidden_dim_2,
                          dropout=dropout, learning_rate=learning_rate)
    # Define the model callbacks
    checkpoint_call_back = ModelCheckpoint(dirpath=f"{save_dir}/VAE/hyper_search_checkpoints/{trial.number}",
                                           filename="best-chckpt", save_top_k=1,
                                           verbose=True, monitor="val_loss", mode="min")
    
    logger = TensorBoardLogger(save_dir=f"{save_dir}/VAE/hyper_search_logs", name="knnMoleculeVAE")

    trainer = pl.Trainer(logger=logger,
                         callbacks=checkpoint_call_back,
                         max_epochs=num_epochs,
                         deterministic=True,
                         enable_progress_bar=True)
    
    model.save_hyperparameters({"enc_hidden_dim_1": enc_hidden_dim_1,
                                "dec_hidden_dim_1": dec_hidden_dim_1,
                                "enc_hidden_dim_2": enc_hidden_dim_2,
                                "dec_hidden_dim_2": dec_hidden_dim_2,
                                "learning_rate": learning_rate,
                                "dropout": dropout, "num_epochs": num_epochs,
                                "seed": seed, "batch_size": batch_size})
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_dataloaders=test_loader)
    
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for VAE model")
    parser.add_argument("--split", type=float, default=0.2, help="split ratio")
    parser.add_argument("--n_trials", type=int, default=100, help="number of trials")
    parser.add_argument("--data_path", type=str, default='../data/data.csv', help="path to data")
    parser.add_argument("--id", type=str, default='VAE', help="name of model")
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.split,args.data_path, args.id), n_trials=args.n_trials)

    best_params = study.best_params
    print("Best hyperparameters: ", best_params)
    print("Best value: ", study.best_value)
