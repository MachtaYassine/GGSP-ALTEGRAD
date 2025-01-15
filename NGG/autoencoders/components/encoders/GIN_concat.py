

import torch
import torch.nn as nn
import torch.nn.functional as F
from .GIN_base import GIN, global_add_pool, GINConv, DeepSets

class GIN_concat(GIN):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2, attention=False):
        super().__init__(input_dim, hidden_dim, latent_dim, n_layers, dropout, attention)
        self.bn = nn.BatchNorm1d(hidden_dim+7)
        self.fc = nn.Linear(hidden_dim+7, latent_dim)
        

    def forward(self, data, deepsets=None):
        #edge_attr = data.edge_features
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
        out= torch.cat((out, data.stats), dim=1)
        out = self.bn(out)
        out = self.fc(out)
        return out