
from ..deepsets import DeepSets
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GINConv,GATv2Conv #Need to look at this one
from torch_geometric.nn import global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2,attention=True):
        super().__init__()
        self.dropout = dropout
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.convs = torch.nn.ModuleList()
        self.load_convs()
        

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data, deepsets=None):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        # aggreagates all the nodes features in the batch
        if deepsets is not None:
            out = deepsets(x, data.batch)
        else:
            out = global_add_pool(x, data.batch)
        
        out = self.bn(out)
        out = self.fc(out)
        return out
    
    def load_convs(self):
        if self.attention:
            self.convs = torch.nn.ModuleList()
            self.convs.append(GATv2Conv(in_channels=self.input_dim, 
                                        out_channels=self.hidden_dim, 
                                        heads=4))  # Omit default parameters


            for layer in range(self.n_layers - 2):  # Apply concat for intermediate layers
                self.convs.append(GATv2Conv(in_channels=self.hidden_dim * 4,  # The output of each GATv2Conv layer is multiplied by the number of attention heads
                                            out_channels=self.hidden_dim, 
                                            heads=4))  # Omit default parameters

            # Last layer without concatenation
            self.convs.append(GATv2Conv(in_channels=self.hidden_dim * 4,  # Input is multiplied by the number of attention heads
                                        out_channels=self.hidden_dim,  # No concatenation in the last layer
                                        heads=1))

        else:
            # If attention is False, we use the GINConv layer
            self.convs = torch.nn.ModuleList()
            self.convs.append(GINConv(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),  
                                                    nn.LeakyReLU(0.2),
                                                    nn.BatchNorm1d(self.hidden_dim),
                                                    nn.Linear(self.hidden_dim, self.hidden_dim), 
                                                    nn.LeakyReLU(0.2))))
            
            for layer in range(self.n_layers - 1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),  
                                                        nn.LeakyReLU(0.2),
                                                        nn.BatchNorm1d(self.hidden_dim),
                                                        nn.Linear(self.hidden_dim, self.hidden_dim), 
                                                        nn.LeakyReLU(0.2))))
        print(f"Loaded {self.n_layers} GIN layers with attention={self.attention}")