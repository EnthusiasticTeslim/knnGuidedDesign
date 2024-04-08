# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# DGL
import dgl
from dgl.nn.pytorch import GraphConv

#************************* Variational Autoencoder *************************
def VAE(
        input_dim, latent_dim,
        enc_hidden_dim_1=100, dec_hidden_dim_1=40,
        enc_hidden_dim_2=70, dec_hidden_dim_2=40,
        dropout=0.2):
    """
    Variational Autoencoder (VAE) class.
    ref: base architecture from
            https://github.com/Imfinethankyou1/TADF-likeness/blob/master/model.py
    """

    encoder = nn.Sequential(
                nn.Linear(input_dim, enc_hidden_dim_1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(enc_hidden_dim_1, enc_hidden_dim_2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(enc_hidden_dim_2, latent_dim*2)) # Two times latent_dim for mean and log_var
    
    decoder = nn.Sequential(
                nn.Linear(latent_dim, dec_hidden_dim_1),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(dec_hidden_dim_1, dec_hidden_dim_2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(dec_hidden_dim_2, input_dim), nn.Sigmoid()) # Sigmoid for output in [0, 1] range
    
    return encoder, decoder

#************************* Graph Neural Network *************************

class GNN(nn.Module):
    def __init__(self, 
                 in_dim, gcn_hidden_dim, fcn_hidden_dim, out_dim,
                 n_gcn_layers, n_fcn_layers):
        super(GNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.classify_layers = nn.ModuleList()
        
        # Create GCN layers
        for i in range(n_gcn_layers):
            if i == 0:
                self.conv_layers.append(GraphConv(in_dim, gcn_hidden_dim))
            else:
                self.conv_layers.append(GraphConv(gcn_hidden_dim, gcn_hidden_dim))
        
        # Create FCN layers
        for i in range(n_fcn_layers):
            if i == 0:
                self.classify_layers.append(nn.Linear(gcn_hidden_dim, fcn_hidden_dim))
            else:
                self.classify_layers.append(nn.Linear(fcn_hidden_dim, fcn_hidden_dim))
        
        self.classify_layers.append(nn.Linear(fcn_hidden_dim, out_dim))
      
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        if torch.cuda.is_available():
            h = g.ndata['h'].float().cuda()
        else:
            h = g.ndata['h'].float()
        
        # Perform graph convolution and activation function.
        for conv_layer in self.conv_layers:
            h = F.relu(conv_layer(g, h))

        g.ndata['h'] = h
        
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        
        # Perform fully connected layers
        for classify_layer in self.classify_layers[:-1]:
            hg = F.relu(classify_layer(hg))
        
        output = self.classify_layers[-1](hg)
        
        return output


