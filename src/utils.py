# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""
import selfies as sf
import numpy as np

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
    smiles_list = np.asanyarray(df.smiles) # get the smiles list
    smiles_alphabet = list(set(''.join(smiles_list))) # get the smiles alphabet
    smiles_alphabet.append(' ')  # for padding
    largest_smiles_len = len(max(smiles_list, key=len)) # get the largest smiles length

    print('--> Translating SMILES to SELFIES...0%')
    # translate smiles to selfies
    selfies_list = list(map(sf.encoder, smiles_list)) # translate smiles to selfies

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list) # get the selfies alphabet
    all_selfies_symbols.add('[nop]') # add padding symbol
    all_selfies_symbols.add('.') # add padding symbol
    selfies_alphabet = list(all_selfies_symbols) # convert to list

    max_len = max(sf.len_selfies(s) for s in selfies_list) # get the largest selfies length

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
            encoded_generated = vae.encoder(decoded_generated)

        return decoded_generated