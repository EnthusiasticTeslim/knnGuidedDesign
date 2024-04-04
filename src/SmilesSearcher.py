# -*- coding: utf-8 -*-
"""
Created on Tue April 2 15:26:24 2024

@author: Teslim Olayiwola
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.neighbors import NearestNeighbors
from rdkit.Contrib.SA_Score import sascorer
import selfies as sf

class MoleculeGuidedSearch:
    def __init__(self, 
                    latent_space: torch.tensor,
                    vae: nn.Module,
                    X: int, width: int, height: int, vocab_itos: dict, **kwargs):
        self.vae = vae # trained VAE model
        self.X = X # training data 
        self.width = width # width of the 2D tensor
        self.height = height # height of the 2D tensor
        self.latent_space = latent_space # latent_space = torch.randn(num_samples, latent_dim)
        self.vocab_itos = vocab_itos # vocabulary to index mapping

    def generate_new_molecules(self, vae, latent_space):
        """
        Generate new molecules from the trained VAE
        Args:
            num_samples : int, number of samples to generate
        Returns:
            encoded_generated : PyTorch tensor, encoded generated molecules
        """
        vae.eval()
        with torch.no_grad():
            decoded_generated = vae.decoder(latent_space)

        return decoded_generated
        
    def number2smiles(self, 
                        generated_encoder, 
                        width: int, height: int, vocab_itos):
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
        # Reshape satisfying_molecules_tensor back to a 3D tensor
        generated_molecules_tensor = generated_encoder.view(-1, width, height) 
        # Convert the PyTorch 3D tensor to a NumPy array
        generated_molecules_np = generated_molecules_tensor.numpy()
        max_val = np.max(generated_molecules_np, axis=2, keepdims=True)
        generated_data = np.where(generated_molecules_np == max_val, 1, 0)
        ## Reproduce SMILES list and visualize the output images
        output_smiles_list = []
        for i in range (0, len(generated_data)):
            sm = sf.decoder(sf.encoding_to_selfies(generated_data[i].tolist(), vocab_itos, enc_type="one_hot"))
            output_smiles_list.append(sm)

        return output_smiles_list

    
    def valid_smiles(self, smiles: list):
        '''return the valid smiles
        Args: smiles: list of smiles
        Returns: valid_smiles: list of valid smiles, their index & print the % of valid smiles
        '''
        valid_smiles = []
        index = []
        for idx, sm in enumerate(smiles):
            mol = Chem.MolFromSmiles(sm)
            if mol:
                valid_smiles.append(sm)
                index.append(idx)
        
        #print(f'% of valid smiles: {100*len(valid_smiles)/len(smiles)}')
        
        return np.array(valid_smiles)

    def knn_neighbor(self, X, neighbors: int = 5):

        nbrs = NearestNeighbors(n_neighbors = neighbors).fit(X)
        
        return nbrs

    def execute(self):
        # STEP 1. generate new molecules
        encoded_generated = self.generate_new_molecules(vae=self.vae, latent_space=self.latent_space)
        #print(encoded_generated.shape)
        #print(encoded_generated)
        # STEP 2. compute the distance
        knn = self.knn_neighbor(self.X.reshape(-1, self.width*self.height), neighbors=5)
        new_samples = encoded_generated.reshape(-1, self.width*self.height)
        distances, _ = knn.kneighbors(new_samples, return_distance=True)
        max_distance = np.max(distances, axis=1)
        # STEP 3. get smiles
        smiles = self.number2smiles(encoded_generated, self.width, self.height, self.vocab_itos)
        valid_smiles = self.valid_smiles(smiles)
        # STEP 4. compute the desired property and synthesizability score
        desired_property = np.array([Descriptors.MolLogP(Chem.MolFromSmiles(sm)) for sm in valid_smiles])
        synthesizability_score = np.array([sascorer.calculateScore(Chem.MolFromSmiles(sm)) for sm in valid_smiles])

        return desired_property, synthesizability_score, max_distance, valid_smiles
