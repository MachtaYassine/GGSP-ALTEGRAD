import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv #Need to look at this one
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter
import sys


from .autoencoder_concat import VariationalAutoEncoder_concat
from NGG.autoencoders.components.deepsets import DeepSets
from NGG.autoencoders.components.encoders.GIN_base import GIN
from NGG.autoencoders.components.encoders.GIN_concat import GIN_concat   
from NGG.autoencoders.components.decoders.decoder_base import Decoder
from NGG.autoencoders.components.decoders.decoder_norm import Decoder_normalized


class GraphStatisticsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=7):
        super(GraphStatisticsModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj_tensor):
        # Flatten the adjacency matrices
        batch_size, num_nodes, _ = adj_tensor.size()
        x = adj_tensor.view(batch_size, -1)
        # print(x.shape)
        encoded = self.encoder(x)
        statistics = self.decoder(encoded)
        return statistics

class GMVAE(VariationalAutoEncoder_concat):
    def __init__(
        self, 
        input_dim, 
        hidden_dim_enc, 
        hidden_dim_dec, 
        latent_dim, 
        n_layers_enc, 
        n_layers_dec, 
        n_max_nodes,
        to_labels, 
        label_dim=8
    ):
        super().__init__(input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes)
        # Label encoder
        self.label_dim = label_dim
        self.dropout = nn.Dropout(0.2)
        self.label_lookup = nn.Linear(label_dim, 64)
        self.lab = nn.Sequential(
            nn.Linear(64,  512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout,
        )
        self.lab_mu = nn.Linear(256, latent_dim)
        self.lab_logvar = nn.Linear(256, latent_dim)
        # Label decoder
        self.label_decoder = nn.Sequential(
            nn.Linear(latent_dim+label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU()
        )
        # Graph decoder
        # print(f"latent and label dim {latent_dim} {label_dim}")
        self.decoder = Decoder(latent_dim+label_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        # We need to go from adj matrix back to the labels to then compare
        # with whats expected so we need to give a function to do that
        self.to_labels = to_labels
        self.bce_loss = nn.BCELoss()
        
        self.label_forward=GraphStatisticsModel(n_max_nodes**2, 256, label_dim)

    # def label_encode(self, labels):
    #     """Encodes the graph descriptions with their cluster label."""
    #     h0 = self.dropout(F.relu(self.label_lookup(labels)))
    #     h = self.lab(h0)
    #     mu = self.lab_mu(h)
    #     log_var = self.lab_logvar(h)
    #     return mu, log_var
    
    def feat_encode(self, data):
        """Encodes the graphs."""
        x_g  = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        return mu, logvar
    
    # def label_forward(self, labels):
    #     """Encodes and decodes labels."""
        
    #     n_label = labels.shape[1]
    #     all_labels = torch.eye(n_label).to(labels.device)
    #     mu_label, log_var_label = self.label_encode(all_labels)
        
    #     z = torch.matmul(labels, mu_label) / labels.sum(1, keepdim=True)
    #     label_emb = self.label_decoder(torch.cat((z, labels), 1))

        
    #     return mu_label, log_var_label, label_emb

    def feat_forward(self, labels, data):
        """Encodes graphs and generate sample from the latent space"""
        mu, log_var = self.feat_encode(data)

        if not self.training:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        adj = self.decoder(torch.cat((z, labels), 1))

        
        stats_out = self.label_forward(adj)
        stats_out_og=self.label_forward(data.A)
        
        #check to labels does not break gradient

        return adj, stats_out, mu, log_var,stats_out_og

    def forward(self, data):
        # Encoding
        cluster_labels = torch.tensor([data[i].label for i in range(len(data))], device=data[0].x.device)
        labels = torch.cat((data.stats, cluster_labels.unsqueeze(1)), 1)
        adj, stats_out, mu, log_var,stats_out_og = self.feat_forward(labels, data)
        # mu_label, log_var_label, label_emb = self.label_forward(labels)

        
        

        output = {
            'adj': adj,
            'mu': mu,
            'log_var': log_var,
            'input_label': labels,
            'stats_out': stats_out,
            'stats_out_og': stats_out_og
        }
        return output
    
    def loss(
        self, 
        data,
        con_temprature = 1.0,
        alpha_mse = 0.1,
        mse_weight = 0.1,
        kl_weight = 1.0,
    ):
        output = self.forward(data)
        input_label = output['input_label']
        adj = output['adj']
        stats_out = output['stats_out']
        mu = output['mu']
        logvar = output['log_var']
        stats_out_og = output['stats_out_og']
       
        mse_recon = F.mse_loss(stats_out, input_label)
        mse_og = F.mse_loss(stats_out_og, input_label)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + kl_weight*kld+ alpha_mse*mse_recon + mse_weight*mse_og
        results = {
            "loss": loss,
            "recon": recon,
            "kld": kld,
            "mse_recon": mse_recon,
            "mse_og": mse_og
        }
        return results