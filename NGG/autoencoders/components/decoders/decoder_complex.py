import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder_base import Decoder
# This is not really normalized but with self loops !


class Decoder_complex(Decoder): #added by Yass mainly to accept (and output) normalized adjacency matrices
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__(latent_dim, hidden_dim, n_layers, n_nodes)
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        #reate a dicttionary of last layers
        last_layer_list={nn.Linear(hidden_dim, n_node*(n_node-1)) for n_node in range(10, n_nodes+1)}
        
        self.last_layer_selector = nn.ModuleDict(last_layer_list)
        self.mlp = nn.ModuleList(mlp_layers)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.relu(self.mlp[i](x))
        
        x = self.last_layer_selector[x]
        
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj