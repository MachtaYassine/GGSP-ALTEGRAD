import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

def generate_edge_data(n_nodes, x):
    # List of edges for a fully connected graph (i != j)
    edge_index = []
    edge_attr = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Ensure i != j (no self-loops)
            edge_index.append([i, j])  # Add edge (i, j)
            # Concatenate the features of nodes i and j
            edge_attr.append(torch.cat([x[i], x[j]]))  # Concatenate their features

    # Convert to tensors
    edge_index = torch.tensor(edge_index).t().contiguous()  # Shape: (2, E)
    edge_attr = torch.stack(edge_attr)  # Shape: (E, 2*F)

    return edge_index, edge_attr


class GATDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, node_feat_dimension): 
        super(GATDecoder, self).__init__()
        
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        
        # mlp to get the initial edge index and edge attributes
        init_mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        init_mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))
        self.init_mlp = nn.ModuleList(init_mlp_layers)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=latent_dim, out_channels=hidden_dim, heads=1, concat=False)
            for _ in range(n_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim, n_nodes * (n_nodes - 1) // 2)  # Final linear layer to produce edge scores

    def forward(self, x, mask=None):
        
        # Step 2: Apply GAT layers
        for i in range(self.n_layers):
            ei=self.init_mlp[i](x)
            edge_index=ei.view(2,-1)
            x = self.gat_layers[i](x, edge_index=self.init_edge_index, edge_attr=self.init_edge_attr)

        # Step 3: Process the output through a final linear layer to get adjacency scores
        x = self.fc(x)
        
        # Step 4: Reshape to produce a dense adjacency matrix
        adj = torch.reshape(x, (x.size(0), -1, 2))  # Reshape to (batch_size, n_edges, 2)
        adj = F.gumbel_softmax(adj, tau=1, hard=True)[:,:,0]  # Sample adjacency matrix using Gumbel softmax

        # Step 5: Convert to adjacency matrix
        adj_matrix = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)  # Upper triangle indices for adjacency
        adj_matrix[:, idx[0], idx[1]] = adj
        adj_matrix = adj_matrix + adj_matrix.transpose(1, 2)  # Make it symmetric

        # Step 6: Apply mask (optional)
        if mask is not None:
            adj_matrix = adj_matrix * mask
        
        return adj_matrix
