from torch_geometric.utils import scatter
import torch.nn as nn


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