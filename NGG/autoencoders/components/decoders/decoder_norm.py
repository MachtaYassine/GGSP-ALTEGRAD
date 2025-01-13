import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder_base import Decoder
# This is not really normalized but with self loops !


class Decoder_normalized(Decoder): #added by Yass mainly to accept (and output) normalized adjacency matrices
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__(latent_dim, hidden_dim, n_layers, n_nodes)
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes+1)//2))

        self.mlp = nn.ModuleList(mlp_layers)

    def forward(self, x,mask):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
            
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 0)
        
        
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.tril(torch.transpose(adj, 1, 2),diagonal=-1)
        if mask is not None:
            adj = adj * mask
        return adj