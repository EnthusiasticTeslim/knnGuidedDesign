# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""
import selfies as sf
import numpy as np
import torch
import dgl
from dgllife.utils import CanonicalAtomFeaturizer 
from dgllife.utils import mol_to_bigraph, smiles_to_complete_graph
from rdkit import Chem

def get_selfies_and_smiles_encodings(df):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        dataframe with 'smiles' as column's name.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    # get the smiles
    smiles_list = np.asanyarray(df.smiles)  # get the smiles list
    smiles_alphabet = list(set(''.join(smiles_list)))  # get the smiles alphabet
    smiles_alphabet.append(' ')  # for padding
    largest_smiles_len = len(max(smiles_list, key=len))  # get the largest smiles length

    print('--> Translating SMILES to SELFIES...0%')
    # translate smiles to selfies
    selfies_list = list(map(sf.encoder, smiles_list))  # translate smiles to selfies

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)  # get the selfies alphabet
    all_selfies_symbols.add('[nop]')  # add padding symbol
    all_selfies_symbols.add('.')  # add padding symbol
    selfies_alphabet = list(all_selfies_symbols)  # convert to list

    max_len = max(sf.len_selfies(s) for s in selfies_list)  # get the largest selfies length

    # get arrays from the selfies and smiles
    vocab_stoi = {symbol: idx for idx, symbol in enumerate(selfies_alphabet)}
    vocab_itos = {idx: symbol for symbol, idx in vocab_stoi.items()}

    print('--> Finished translating SMILES to SELFIES...100%')

    return selfies_list, selfies_alphabet, max_len, \
           smiles_list, smiles_alphabet, largest_smiles_len, vocab_stoi, vocab_itos


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Go from a single selfies string to a one-hot encoding.
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    print('--> Creating one-hot encoding...0%')
    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    print('--> Finished creating one-hot encoding...100%')
    return np.array(hot_list)



def generated_molecules_to_smiles(
                                    generated_molecules, 
                                    width, height, 
                                    vocab_itos):
    """
    Convert the generated molecules to SMILES representation
    Args:
        generated_molecules : PyTorch tensor
        width : int, width of the 2D tensor
        height : int, height of the 2D tensor
        vocab_itos : dict, vocabulary to index mapping
    Returns:
        output_smiles_list : list, SMILES representation of the generated molecules
    """
    print('--> Converting arrays to smiles...0%')
    # Reshape satisfying_molecules_tensor back to a 3D tensor
    generated_molecules_tensor_3d = generated_molecules.view(-1, width, height) 
    # Convert the PyTorch 3D tensor to a NumPy array
    generated_molecules_numpy = generated_molecules_tensor_3d.numpy()
    max_values = np.max(generated_molecules_numpy, axis=2, keepdims=True)
    generated_data = np.where(generated_molecules_numpy == max_values, 1, 0)
    ## Reproduce SMILES list and visualize the output images
    output_smiles_list = []
    for i in range (0,len(generated_data)):
        sm = sf.decoder(sf.encoding_to_selfies(generated_data[i].tolist(), vocab_itos, enc_type="one_hot"))
        output_smiles_list.append(sm)

    print('--> Finished converting arrays to smiles...100%')

    return output_smiles_list


# Define a function to flatten the input data
def flatten(x_train):
    """
    Flatten the input data.

    Args:
        x_train (torch.Tensor): The input dataset as a PyTorch tensor with shape (num_samples, width, height).

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing the input dimension (width * height) and
        the flattened dataset as a PyTorch tensor.
    """
    dataset_shape = x_train.shape
    num_samples = dataset_shape[0]
    width, height = dataset_shape[1:]

    # Calculate input_dim
    input_dim = width * height

    # Reshape the dataset into a 2D format
    flattened_dataset = x_train.reshape(num_samples, input_dim)

    # Return both input_dim and flattened_dataset
    return width, height, input_dim, flattened_dataset


def generate_new_molecules(vae, num_samples = 500, latent_dim = 32):
        """
        Generate new molecules from the trained VAE
        Args:
            num_samples : int, number of samples to generate
        Returns:
            encoded_generated : PyTorch tensor, encoded generated molecules
        """
        vae.eval()
        with torch.no_grad():
            latent_samples = torch.randn(num_samples, latent_dim)
            decoded_generated = vae.decoder(latent_samples)

        return decoded_generated

__all__ = ['graph_dataset']
class graph_dataset(object):

    '''A dataset class for molecular graphs. modified from Amy Qing's code at
    https://github.com/zavalab/ML/blob/master/CMC_GCN/code/generate_graph_dataset.py'''


    def __init__(self, smiles, y, 
                 node_enc = CanonicalAtomFeaturizer(), edge_enc = None,
                 graph_type = mol_to_bigraph, canonical_atom_order = False):
        super(graph_dataset, self).__init__()
        self.smiles = smiles
        self.y = y
        self.graph_type = graph_type
        self.node_enc = node_enc
        self.edge_enc = edge_enc
        self.canonical_atom_order = canonical_atom_order
        self.graphs = []
        self.labels = []
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample."""
        return self.graphs[idx], self.labels[idx]

    def getsmiles(self, idx):
        """Get the i^th smiles."""
        return self.smiles[idx]
    
    def node_to_atom(self, idx):
        """Get the i^th atom list."""
        g = self.graphs[idx]
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        node_feat = g.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list = []
        for i in range(g.number_of_nodes()):
            atom_list.append(allowable_set[np.where(node_feat[i]==1)[0][0]])
        return atom_list
    
    def _generate(self):
        '''Generate the graphs and labels.'''
        if self.graph_type==mol_to_bigraph:
            for i,j in enumerate(self.smiles):
                m = Chem.MolFromSmiles(j)
                g = self.graph_type(m,True,self.node_enc,self.edge_enc,
                                    self.canonical_atom_order)
                self.graphs.append(g)
                self.labels.append(torch.tensor(self.y[i], dtype=torch.float32))
        elif self.graph_type==smiles_to_complete_graph:
            for i,j in enumerate(self.smiles):
                g = self.graph_type(j,True,self.node_enc,self.edge_enc,
                                    self.canonical_atom_order)
                self.graphs.append(g)
                self.labels.append(torch.tensor(self.y[i], dtype=torch.float32))
                

def summarize_graph_data(g):
    node_data = g.ndata['h'].numpy()
    print("node data:\n",node_data)
    edge_data = g.edata
    print("edge data:",edge_data)
    adj_mat = g.adjacency_matrix_scipy(transpose=True,return_edge_ids=False)
    adj_mat = adj_mat.todense().astype(np.float32)
    print("adjacency matrix:\n",adj_mat)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return torch.tensor(batched_graph), torch.tensor(labels).unsqueeze(-1)