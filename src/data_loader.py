import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch, glob, yaml
from tbparse import SummaryReader
from pytorch_lightning import LightningDataModule
from multiprocessing import cpu_count
from dgl.dataloading import GraphDataLoader
try:
    from utils import get_selfies_and_smiles_encodings, multiple_selfies_to_hot  # selfies and smiles functions
    from utils import graph_dataset
except:
    from .utils import get_selfies_and_smiles_encodings, multiple_selfies_to_hot  # selfies and smiles functions
    from .utils import graph_dataset


#*********************** Load the data into selfies ***********************
def load_data(seed, test_size=0.2,
              csv_path='../data/data.csv'):
    # Load the data
    data = pd.read_csv(csv_path)

    # Transform the data into a list of SELFIES and SMILES
    selfies_list, selfies_alphabet, largest_selfies_len,_, _, _, vocab_stoi, vocab_itos = get_selfies_and_smiles_encodings(data)

    # One-Hot-Encode the SELFIES into a tensor
    input_one_hot_arr = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)

    # split the data into training and validation sets
    x_train, x_val = train_test_split(input_one_hot_arr, test_size=test_size, random_state=seed)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)

    return x_train, x_val


#*********************** Load the data into Graph ***********************
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
    def __init__(self, csv_path, test_size=0.2, batch_size=16):
        super().__init__()
        self.train_data, self.test_data, self.val_data = load_graph_data(seed=42, test_size=test_size, csv_path=csv_path)
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
def load_auto_hp_params(path):
    log_dirs = sorted(glob.glob(path))

    hp_params = {"run":[],
                    'enc_hidden_dim_1': [],
                    'dec_hidden_dim_1': [],
                    'enc_hidden_dim_2': [],
                    'dec_hidden_dim_2': [],
                    "dropout":[],
                    "lr": [],
                    "trn_loss":[],
                    "val_loss":[]}

    for i,v in enumerate(log_dirs):
        reader = SummaryReader(log_dirs[i])
        df = reader.scalars
        print(f"run_{i} has {len(df)} epochs")

        if len(df) > 2:
            hp_params["run"].append(f"run_{i}")
            with open(f"{log_dirs[i]}/hparams.yaml", "r") as file:
                hparams = yaml.safe_load(file)
                print(hparams)

            for k,v in hparams.items():
                if k not in ["n_features", "n_classes"]:
                    hp_params[k].append(v)

            train_loss = df[df.tag=="train_loss"].value.to_numpy()
            val_loss = df[df.tag=="val_loss"].value.to_numpy()

            hp_params["trn_loss"].append(np.nanmin(train_loss))
            hp_params["val_loss"].append(np.nanmin(val_loss))
    
    df = pd.DataFrame(hp_params)
    return df


def load_gnn_hp_params(path):
    log_dirs = sorted(glob.glob(path))
    hp_params = {"run":[],
                    'gc_hidden_dim': [], 'fcn_hidden_dim': [],
                    'n_gcn_layers': [], 'n_fcn_layers': [], "lr": [],
                    "trn_loss":[], "val_loss":[],
                    "trn_acc":[], "val_acc":[]}

    for i,v in enumerate(log_dirs):
        reader = SummaryReader(log_dirs[i])
        df = reader.scalars
        print(f"run_{i} has {len(df)} epochs")

        if len(df) > 2:
            hp_params["run"].append(f"run_{i}")
            with open(f"{log_dirs[i]}/hparams.yaml", "r") as file:
                hparams = yaml.safe_load(file)
                print(hparams)

            for k,v in hparams.items():
                if k not in ["n_features", "n_classes"]:
                    hp_params[k].append(v)

            train_loss = df[df.tag=="train_loss"].value.to_numpy()
            val_loss = df[df.tag=="val_loss"].value.to_numpy()
            train_acc = df[df.tag=="train_acc"].value.to_numpy()
            val_acc = df[df.tag=="val_acc"].value.to_numpy()

            hp_params["trn_loss"].append(np.nanmin(train_loss))
            hp_params["val_loss"].append(np.nanmin(val_loss))
            hp_params["trn_acc"].append(np.nanmax(train_acc))
            hp_params["val_acc"].append(np.nanmax(val_acc))
    
    df = pd.DataFrame(hp_params)
    return df