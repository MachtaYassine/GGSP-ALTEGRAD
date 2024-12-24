import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv #Need to look at this one
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter
import sys
# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj

class Decoder_normalized(Decoder): #added by Yass mainly to accept (and output) normalized adjacency matrices
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super().__init__(latent_dim, hidden_dim, n_layers, n_nodes)

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
            
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 0)
        
        
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.tril(torch.transpose(adj, 1, 2),diagonal=-1)
        return adj
    
class DeepSets(nn.Module):
    def __init__(self, hidden_dim): # Keep the embedding dimension
        super(DeepSets, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, batch):    
        phi_x = self.tanh(self.fc1(x))
        sum_x = scatter(phi_x, batch, dim=0, reduce='sum')
        y = self.fc2(sum_x)
        
        return y.squeeze()  


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

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
    

class GIN_concat(GIN):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__(input_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.bn = nn.BatchNorm1d(hidden_dim+7)
        self.fc = nn.Linear(hidden_dim+7, latent_dim)
        

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
        out= torch.cat((out, data.stats), dim=1)
        out = self.bn(out)
        out = self.fc(out)
        return out



# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, deepsets: DeepSets = None):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        # self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = GIN_concat(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        # self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.decoder = Decoder(latent_dim+7, hidden_dim_dec, n_layers_dec, n_max_nodes)
        # self.decoder = Decoder_normalized(latent_dim+7, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.deepsets = deepsets

    def forward(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def edge_node_coherence_loss(self,adj_matrices,alpha1=1,alpha2=1,alpha3=1):
        """
        Computes the edge-node coherence loss for a given adjacency matrix.
        Every node with an edge must have a self-cycle.
        
        Parameters:
            adj_matrix (torch.Tensor): A square adjacency matrix of shape (N, N).
                                    Entries should be differentiable (e.g., predictions).
        
        Returns:
            torch.Tensor: The computed edge-node coherence loss.
        """
        row_sum = torch.sum(adj_matrices, dim=2)  # Shape: (B, N)
        
        
      
        # Extract diagonal entries (self-cycles) for each matrix in the batch
        diag = torch.diagonal(adj_matrices, dim1=1, dim2=2)  # Shape: (B, N)
        
        row_penalty = torch.relu(row_sum - row_sum.sum()*diag)  # Shape: (B, N)
        
        
        
        return row_penalty.sum()


    def loss_function(self, data, beta=0.05): 
        
        
        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but without any information about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        adj = self.decoder(x_g) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with no information whatsoever about the input prompt and stats
        
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld


        ###############################################
        #data is a torch_geometric.data.Data object that contains a batch of graphs
        # print(data.x.shape)# N_nodes_in_all_batch*(Spectral_embedding+1) these are the node features tensors, they are all aggregated in a single tensor!
        # print(data.edge_index.shape) # these are the edge indices, they are also aggregated in a single tensor, and they are indexed by the new node indices that can go up to 7000
        # print(data.A.shape) #BS*max_nodes*max_nodes this is not aggregated
        # print(data.stats.shape) # BS*7
        # print(len(data.filename)) #list of BS strings containing the filenames
        # print(data.ptr.shape) #BS+1 I don't understand this one ??
        # print(data.batch.shape) #N_nodes_in_all_batch this keeps track of "to which graph does the node belong to"
        # sys.exit()
        return loss, recon, kld

    

class VariationalAutoEncoder_concat(VariationalAutoEncoder):
    def __init__(
        self, 
        input_dim, 
        hidden_dim_enc, 
        hidden_dim_dec, 
        latent_dim, 
        n_layers_enc, 
        n_layers_dec, 
        n_max_nodes,
        deepsets: DeepSets = None,
        normalize: bool = False,
    ):
        super().__init__(input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, deepsets)
        self.encoder = GIN_concat(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        additional_dim = 7
        if normalize:
            self.decoder = Decoder_normalized(latent_dim+additional_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
        self.decoder = Decoder(latent_dim+additional_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)


    def forward(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        stats = data.stats  # Shape: (batch_size, num_stats)
        x_g = torch.cat((x_g, stats), dim=1) 
        adj = self.decoder(x_g)
        return adj


    def loss(
        self, 
        data, 
        beta=0.05, 
        contrastive_hyperparameters: list = None, 
        penalization_hyperparameters: float = None,
    ): #this loss variant concatenates the stats to the latent space before decoding
        
        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but without any information about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        # Concatenate stats and one-hot-encoded labels
        stats = data.stats  # Shape: (batch_size, num_stats)
        x_g = torch.cat((x_g, stats), dim=1) 
        adj = self.decoder(x_g) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with no information whatsoever about the input prompt and stats
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld
        results = {
            "recon": recon,
            "kld": kld,
        }
        if contrastive_hyperparameters is not None:
            contrastive_loss = self.get_weighted_contrastive_loss(mu, logvar, data, contrastive_hyperparameters[0], contrastive_hyperparameters[1])
            loss += contrastive_loss
            results["contrastive_loss"] = contrastive_loss
        if penalization_hyperparameters is not None:
            penalization = self.get_weighted_penalization(adj, data, penalization_hyperparameters)
            loss += penalization
            results["penalization"] = penalization
        results["loss"] = loss
        return results
            
    
    def loss_function_concat_stats_pn(self, data, beta=0.05,alpha=0.05): # This loss variants concatenates the stats to the latent space before decoding and tries to punish n_nodes and n_edges
        
        
        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but with info about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        x_g = torch.cat((x_g, data.stats), dim=1)
        adj = self.decoder(x_g) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with information about the input prompt and stats
        
        
        n_edges= adj.sum(dim=(1,2))/2
        num_nodes = torch.sum(torch.diagonal(adj, dim1=1, dim2=2),dim=1)
        coherence_loss = self.edge_node_coherence_loss(adj)
        
        MSE_n_nodes= torch.square(num_nodes - data.stats[:,0]).mean()
        MSE_n_edges= torch.square(n_edges - data.stats[:,1]).mean()

        feature_losses= (MSE_n_nodes+MSE_n_edges+coherence_loss)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld + alpha*feature_losses
        # print(loss)
        # print(feature_losses)
        return loss, recon, kld,feature_losses
    
    def get_weighted_contrastive_loss(self, mu, logvar, data, lambda_contrastive, lambda_entropy):
        """
        Computes the contrastive loss of the batch using cosine similarity. 
        The contrastive loss is multiplied by lambda_contrastive.
        Additionally, a regularization term is added to maximize the entropy between clusters.
        
        The dynamic margin is computed based on the mean pairwise cosine similarity between the samples.
        Cosine similarity is computed for both mu and logvar.
        """
        batch_size = mu.size(0)
        
        # Normalize the mu (mean vectors) and logvar (log variance vectors) for cosine similarity
        mu_norm = F.normalize(mu, p=2, dim=1)  # Normalize along the feature dimension for mu
        logvar_norm = F.normalize(logvar, p=2, dim=1)  # Normalize along the feature dimension for logvar

        # Compute cosine similarity using dot product (cosine similarity = mu[i] . mu[j])
        cosine_sim_matrix_mu = torch.matmul(mu_norm, mu_norm.t())  # Shape: (batch_size, batch_size)
        cosine_sim_matrix_logvar = torch.matmul(logvar_norm, logvar_norm.t())  # Shape: (batch_size, batch_size)

        # Compute the dynamic margin based on the average cosine similarity
        avg_cosine_sim_mu = cosine_sim_matrix_mu.sum() / (batch_size * (batch_size - 1))  # Average cosine similarity for mu
        avg_cosine_sim_logvar = cosine_sim_matrix_logvar.sum() / (batch_size * (batch_size - 1))  # Average cosine similarity for logvar
        
        # Combine the cosine similarity matrices for mu and logvar
        cosine_sim_matrix = (cosine_sim_matrix_mu + cosine_sim_matrix_logvar) / 2  # Averaging cosine similarities

        # Calculate dynamic margin based on the mean cosine similarity
        dynamic_margin = avg_cosine_sim_mu + avg_cosine_sim_logvar

        # Determine whether pairs belong to the same cluster
        labels = torch.tensor([data[i].label for i in range(len(data))]).to("cuda")
        same_cluster = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # Shape: (batch_size, batch_size)

        # Apply contrastive loss formula based on cosine similarity
        loss_same_cluster = same_cluster * (1 - cosine_sim_matrix).pow(2)  # Loss for same cluster: minimize cosine distance (maximize similarity)
        loss_diff_cluster = (1 - same_cluster) * F.relu(cosine_sim_matrix - dynamic_margin).pow(2)  # Loss for different clusters: enforce dynamic margin

        # Total contrastive loss
        contrastive_loss = (loss_same_cluster + loss_diff_cluster).sum() / (batch_size * (batch_size - 1))

        # Compute the entropy of the cluster assignments for each sample
        # Using the softmax over mu (assuming mu can be used to assign clusters)
        cluster_probs = F.softmax(mu, dim=1)  # Assuming softmax over the latent variables for clustering
        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8)) / batch_size  # Regularization term

        # Total loss includes the contrastive loss and the entropy regularization
        total_loss = lambda_contrastive * contrastive_loss + lambda_entropy * entropy_loss

        return total_loss
    
    
    def get_weighted_penalization(self, adj, data, lambda_penalization):
        """
        Compute a penalization representing the MSE with the expected number of 
        nodes and edges and multiply it by lambda_penalization.
        """
        n_edges= adj.sum(dim=(1,2))/2
        num_nodes = torch.sum(torch.diagonal(adj, dim1=1, dim2=2),dim=1)
        coherence_loss = self.edge_node_coherence_loss(adj)
        
        MSE_n_nodes = torch.square(num_nodes - data.stats[:,0]).mean()
        MSE_n_edges = torch.square(n_edges - data.stats[:,1]).mean()

        feature_losses = (MSE_n_nodes + MSE_n_edges + coherence_loss)

        return lambda_penalization * feature_losses
    



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

    def feat_forward(self, labels, data):
        """Encodes graphs and generate sample from the latent space"""
        mu, log_var = self.feat_encode(data)

        if not self.training:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        adj = self.decoder(torch.cat((z, labels), 1))

        _ , _ , feat_emb = self.label_forward(self.to_labels(adj))

        return adj, feat_emb, mu, log_var

    def forward(self, data):
        # Encoding
        cluster_labels = torch.tensor([data[i].label for i in range(len(data))], device="cuda")  # Shape: (batch_size,)
        labels = torch.cat((data.stats, cluster_labels.unsqueeze(1)), 1)
        adj, feat_emb, mu, log_var = self.feat_forward(labels, data)
        mu_label, log_var_label, label_emb = self.label_forward(labels)

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
            'input_label': labels,
        }
        return output
    
    def loss(
        self, 
        data,
        con_temprature = 1.0,
        alpha_mse = 0.5,
        mse_weight = 0.01,
        kl_weight = 1.0,
    ):
        output = self.forward(data)
        input_label = output['input_label']
        label_out, mu_label, log_var_label, label_emb = \
            output['label_out'], output['mu_label'], output['log_var_label'], output['label_emb']
        feat_out, mu, log_var, feat_emb, adj = \
            output['feat_out'], output['mu'], output['log_var'], output['feat_emb'], output['adj']
        embs = output['embs']

        fx_sample = self.reparameterize(mu, log_var)
        fx_var = torch.exp(log_var)
        fe_var = torch.exp(log_var_label)

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
            indiv_log_prob = log_normal(z, m, v)
            log_prob = log_mean_exp(indiv_log_prob, mask)
            return log_prob

        def log_mean_exp(x, mask):
            return log_sum_exp(x, mask) - torch.log(mask.sum(1))

        def log_sum_exp(x, mask):
            max_x = torch.max(x, 1)[0]
            new_x = x - max_x.unsqueeze(1).expand_as(x)
            return max_x + (new_x.exp().sum(1)).log()

        recon_graph = F.l1_loss(adj, data.A, reduction='mean')
        mse_labels = F.mse_loss(label_out, input_label)
        mse_graph_features = F.mse_loss(feat_out, input_label)
        mse = alpha_mse * mse_labels + (1 - alpha_mse) * mse_graph_features
        kl_loss = (log_normal(fx_sample, mu, fx_var) - \
            log_normal_mixture(fx_sample, mu_label, fe_var, input_label)).mean()
        cpc_loss = supconloss(label_emb, feat_emb, embs)
        total_loss = mse_weight * mse + kl_weight * kl_loss + cpc_loss + recon_graph
        results = {
            'loss': total_loss,
            'kl': kl_loss,
            'cpc': cpc_loss,
            'recon': recon_graph,
            'mse': mse,
        }
        return results
    
