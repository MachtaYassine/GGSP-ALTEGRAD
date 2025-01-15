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
        label_dim=8, 
        num_clusters=3,
    ):
        super().__init__(input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes)
        # Label encoder
        self.label_dim = label_dim
        self.num_clusters = num_clusters
        self.dropout = nn.Dropout(0.2)
        self.label_lookup = nn.Linear(num_clusters, 64)
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
            nn.Linear(latent_dim+num_clusters, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU()
        )
        # Graph decoder
        self.decoder = Decoder(latent_dim+label_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        # We need to go from adj matrix back to the labels to then compare
        # with whats expected so we need to give a function to do that
        self.to_labels = to_labels
        self.bce_loss = nn.BCELoss()

    def label_encode(self, labels):
        """Encodes the graph descriptions with their cluster label."""
        h0 = self.dropout(F.relu(self.label_lookup(labels)))
        h = self.lab(h0)
        mu = self.lab_mu(h)
        log_var = self.lab_logvar(h)
        return mu, log_var
    
    def feat_encode(self, data):
        """Encodes the graphs."""
        x_g  = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        return mu, logvar
    
    def label_forward(self, labels):
        """Encodes and decodes labels."""
        n_label = labels.shape[1]
        all_labels = torch.eye(n_label).to(labels.device)
        mu_label, log_var_label = self.label_encode(all_labels)
        
        z = torch.matmul(labels, mu_label) / labels.sum(1, keepdim=True)
        label_emb = self.label_decoder(torch.cat((z, labels), 1))

        return mu_label, log_var_label, label_emb

    def feat_forward(self, stats, data):
        """Encodes graphs and generate sample from the latent space"""
        mu, log_var = self.feat_encode(data)

        if not self.training:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        adj = self.decoder(torch.cat((z, stats), 1))

        stats = self.to_labels(adj)
        one_hot_labels = F.one_hot(torch.clamp(stats[:, -1].to(torch.int64), max=self.num_clusters - 1, min=0), num_classes=self.num_clusters).float()
        _ , _ , feat_emb = self.label_forward(one_hot_labels)

        return adj, feat_emb, mu, log_var, stats

    def forward(self, data):
        # Encoding
        cluster_labels = torch.tensor([data[i].label for i in range(len(data))], device="cuda")  # Shape: (batch_size,)
        one_hot_labels = F.one_hot(cluster_labels.to(torch.int64), num_classes=self.num_clusters).float()
        stats = torch.cat((data.stats, cluster_labels.unsqueeze(1)), 1)
        adj, feat_emb, mu, log_var, output_stats = self.feat_forward(stats, data)
        mu_label, log_var_label, label_emb = self.label_forward(one_hot_labels)

        # Align
        embs = self.label_lookup.weight
        label_out = torch.matmul(label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)

        output = {
            'adj': adj,
            'mu': mu,
            'log_var': log_var,
            'feat_emb': feat_emb,
            'mu_label': mu_label,
            'log_var_label': log_var_label,
            'label_emb': label_emb,
            'embs': embs,
            'label_out': label_out,
            'feat_out': feat_out,
            'input_label': one_hot_labels,
            'input_stats': stats,
            'output_stats': output_stats,
        }
        return output
    
    def loss(
        self, 
        data,
        con_temprature = 1.0,
        alpha_bce = 0.5,
        bce_weight = 1.0,
        kl_weight = 1.0,
        mse_weight = 0.001,
    ):
        output = self.forward(data)
        input_label = output['input_label']
        label_out, mu_label, log_var_label, label_emb = \
            output['label_out'], output['mu_label'], output['log_var_label'], output['label_emb']
        feat_out, mu, log_var, feat_emb, adj = \
            output['feat_out'], output['mu'], output['log_var'], output['feat_emb'], output['adj']
        embs = output['embs']
        inp_stats, out_stats = output['input_stats'], output['output_stats']

        fx_sample = self.reparameterize(mu, log_var)
        fx_var = torch.exp(log_var)
        fe_var = torch.exp(log_var_label)

        pred_label = torch.sigmoid(label_out)
        pred_graph_label = torch.sigmoid(feat_out)

        def compute_BCE_and_RL_loss(E):
            #compute negative log likelihood (BCE loss) for each sample point
            sample_nll = -(
                torch.log(E) * input_label + torch.log(1 - E) * (1 - input_label)
            )
            logprob = -torch.sum(sample_nll, dim=2)

            #the following computation is designed to avoid the float overflow (log_sum_exp trick)
            maxlogprob = torch.max(logprob, dim=0)[0]
            Eprob = torch.mean(torch.exp(logprob - maxlogprob), axis=0)
            nll_loss = torch.mean(-torch.log(Eprob) - maxlogprob)
            return nll_loss

        def supconloss(label_emb, feat_emb, embs):
            features = torch.cat((label_emb, feat_emb))
            labels = torch.cat((input_label, input_label)).float()
            n_label = labels.shape[1]
            emb_labels = torch.eye(n_label).to("cuda")
            mask = torch.matmul(labels, emb_labels)

            anchor_dot_contrast = torch.div(
                torch.matmul(features, embs),
                con_temprature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -mean_log_prob_pos
            loss = loss.mean()
            return loss
        
        def log_normal(x, m, v):
            log_prob = (-0.5 * (torch.log(v) + (x-m).pow(2) / v)).sum(-1)
            return log_prob

        def log_normal_mixture(z, m, v, mask=None):
            m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
            v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
            batch, mix, dim = m.size()
            z = z.view(batch, 1, dim).expand(batch, mix, dim)
            indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
            log_prob = log_mean_exp(indiv_log_prob, mask)
            return log_prob

        def log_mean_exp(x, mask):
            return log_sum_exp(x) - torch.log(mask.sum(1))

        def log_sum_exp(x):
            max_x = torch.max(x, 1)[0]
            new_x = x - max_x.unsqueeze(1).expand_as(x)
            return max_x + (new_x.exp().sum(1)).log()

        recon_graph = F.l1_loss(adj, data.A, reduction='mean')
        stats_mse = F.mse_loss(out_stats, inp_stats)
        bce_labels = compute_BCE_and_RL_loss(pred_label.unsqueeze(0))
        bce_graph_features = compute_BCE_and_RL_loss(pred_graph_label.unsqueeze(0))
        bce = alpha_bce * bce_labels + (1 - alpha_bce) * bce_graph_features
        kl_loss = (log_normal(fx_sample, mu, fx_var) - \
            log_normal_mixture(fx_sample, mu_label, fe_var, input_label)).mean()
        cpc_loss = supconloss(label_emb, feat_emb, embs)
        total_loss = bce_weight * bce + kl_weight * kl_loss + cpc_loss + recon_graph + mse_weight * stats_mse
        results = {
            'loss': total_loss,
            'kl': kl_loss,
            'cpc': cpc_loss,
            'recon': recon_graph,
            'bce': bce,
            'mse': stats_mse,
        }
        return results