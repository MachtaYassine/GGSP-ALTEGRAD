import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain
from joblib import Parallel, delayed

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_adj, dense_to_sparse, degree

from extract_feats import extract_feats, extract_numbers

import sys

def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, normalize=False, labelize=False):

    data_lst = []
    if dataset == 'test':
        filename = f'./data/dataset_{dataset}_nodes_{n_max_nodes}_embed_dim{spectral_emb_dim}_with_labels_{labelize}.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            if labelize:
                data_lst, kmeans = assign_labels(data_lst)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                
                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id)) #prompt=desc for testing
            if labelize:
                data_lst, kmeans = assign_labels(data_lst)
            fr.close()                    
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = f'./data/dataset_{dataset}_nodes_{n_max_nodes}_embed_dim_{spectral_emb_dim}_norm_{normalize}_with_labels_{labelize}.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            if labelize:
                data_lst, kmeans = assign_labels(data_lst)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with sp.special.errstate(singular="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                #normalize adjacency matrix
                if normalize:
                    adj = adj + torch.eye(G.number_of_nodes())
                    # print(adj)
                    # sys.exit()
                    
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            if labelize:
                data_lst, kmeans = assign_labels(data_lst)
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')

    if labelize:
        return data_lst, kmeans
    return data_lst


        

def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def assign_labels(data, n_clusters=3):
    """
    Assigns cluster labels to each graph in the data list.
    
    Args:
    - data: List of Data objects containing graph information.
    - n_clusters: Number of clusters for K-means. Set to 3 after WCSS visualization.
    
    Returns:
    - data: Updated list with cluster labels assigned.
    """
    properties = torch.cat([graph.stats for graph in data], axis=0)

    properties_np = properties.numpy()

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(properties_np)

    # Assign labels back to each graph
    for graph, label in zip(data, labels):
        graph.label = torch.tensor([label])  # Add the label as a tensor

    return data, kmeans

def create_deepsets_train_dataset(hidden_dim, batch_size, device):
    n_train = 100000

    X_train, batch = [], []
    for i in range(n_train):
        sample = np.random.normal(0, 5, hidden_dim)
        X_train.append(sample)
        batch.append(i // batch_size)
    
    X_train = torch.tensor(X_train).to(device).to(torch.float)
    batch = torch.tensor(batch).to(device)
    y_train = scatter(X_train, batch, dim=0, reduce='sum').to(device)
    return X_train, y_train, batch

# Function to compute graph features
def compute_graph_features_from_adj(adj_matrix):
    graph = nx.from_numpy_array(adj_matrix)

    # Compute features
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    avg_degree = sum(dict(graph.degree()).values()) / n_nodes if n_nodes > 0 else 0
    n_triangles = sum(nx.triangles(graph).values()) // 3
    clustering_coeff = nx.average_clustering(graph)
    max_core = max(nx.core_number(graph).values())
    communities = nx.community.greedy_modularity_communities(graph)
    n_communities = len(communities)

    return [n_nodes, n_edges, avg_degree, n_triangles, clustering_coeff, max_core, n_communities]

def compute_graph_features_from_adj_torch(adj_matrix):
    """
    Computes graph features using PyTorch Geometric utilities.

    Args:
        adj_matrix: PyTorch tensor of shape (num_nodes, num_nodes), representing the adjacency matrix.

    Returns:
        A list of graph features.
    """
    # Basic graph properties
    num_nodes = adj_matrix.size(0)
    num_edges = torch.count_nonzero(adj_matrix) // 2  # Each edge is counted twice in an adjacency matrix
    degrees = adj_matrix.sum(dim=1)
    avg_degree = degrees.mean().item()
    n_triangles = torch.trace(torch.matrix_power(adj_matrix, 3)) // 6

    # Clustering coefficient
    denom = degrees * (degrees - 1)
    clustering_coeff = (torch.diagonal(torch.matrix_power(adj_matrix, 3)) / denom.clamp(min=1)).mean().item()

    # Community detection
    edge_index = dense_to_sparse(adj_matrix)[0]
    communities = greedy_modularity_communities(edge_index, num_nodes)
    num_communities = len(communities)

    # Maximum core number (approximation as max degree for simplicity)
    max_core = max_core_number(edge_index, num_nodes)

    return [
        num_nodes,
        num_edges.item(),
        avg_degree,
        n_triangles.item(),
        clustering_coeff,
        max_core,
        num_communities,
    ]


def greedy_modularity_communities(edge_index, num_nodes):
    """
    Implements modularity-based community detection using a greedy approach.

    Args:
        edge_index: Tensor of shape (2, num_edges) containing edge indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        communities: A list of lists, where each sublist represents a community's nodes.
    """
    # Initialize each node as its own community
    communities = [[i] for i in range(num_nodes)]

    # Convert edge_index to adjacency matrix
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    degrees = adj_matrix.sum(dim=1)

    m = adj_matrix.sum().item()  # Total edge weight

    def compute_modularity_gain(u, v, community):
        # Computes modularity gain for merging `u` and `v` into `community`
        deg_u = degrees[u]
        deg_v = degrees[v]
        delta_modularity = (
            adj_matrix[u, v] - (deg_u * deg_v) / (2 * m)
        )
        return delta_modularity

    improved = True
    while improved:
        improved = False
        for i, community in enumerate(communities):
            for j in range(len(communities)):
                if i == j:
                    continue
                # Try merging community i and j
                gain = sum(
                    compute_modularity_gain(u, v, communities[j])
                    for u in community
                    for v in communities[j]
                )
                if gain > 0:
                    # Merge communities
                    communities[j].extend(community)
                    communities.pop(i)
                    improved = True
                    break
            if improved:
                break

    return communities

def max_core_number(edge_index, num_nodes):
    """
    Compute the maximum core number (core decomposition) using PyTorch Geometric.
    
    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int): The number of nodes in the graph.
        
    Returns:
        torch.Tensor: The maximum core number of the graph.
    """
    # Compute the degree of each node
    deg = degree(edge_index[0], num_nodes=num_nodes).to(torch.int)
    
    # Core number initialization: each node's core number starts at its degree
    core_numbers = deg.clone()
    
    # Sorting nodes by degree (ascending order)
    sorted_nodes = torch.argsort(deg)
    
    # Core decomposition to compute the core number of each node
    for node in sorted_nodes:
        neighbors = edge_index[1][edge_index[0] == node]
        min_core = min([core_numbers[neighbor] for neighbor in neighbors] + [deg[node]])
        core_numbers[node] = min_core
    
    # The max core number is the maximum of the core numbers
    max_core = torch.max(core_numbers).item()
    
    return max_core


# def to_labels(adj, kmeans):
#     """
#     Computes graph features and clustering labels using torch_geometric utilities.
#     """
#     kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float) # Handles type errors
#     all_properties = []
#     for adj_matrix in adj:
#         # Compute graph features for each adjacency matrix
#         properties = compute_graph_features_from_adj_torch(adj_matrix)
#         all_properties.append(properties)
    
#     # Convert features to numpy float64
#     all_properties = np.array(all_properties, dtype=np.float64)
    
#     # Cluster label prediction
#     labels = kmeans.predict(all_properties)  # Predict labels for all graphs in the batch
    
#     # Add the labels to the properties
#     all_properties_with_labels = np.hstack((all_properties, labels.reshape(-1, 1)))
    
#     # Transform back to a tensor of size (batch_size, nfeatures + 1)
#     return torch.tensor(all_properties_with_labels, dtype=torch.float32).to(adj.device)

def to_labels(adj, kmeans):
    """
    Computes graph features and clustering labels using torch_geometric utilities.
    """
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)  # Handles type errors
    
    arr_adj = adj.detach().cpu().numpy()
    # Use joblib for parallel computation
    all_properties = Parallel(n_jobs=-1)(
        delayed(compute_graph_features_from_adj)(adj_matrix) for adj_matrix in arr_adj
    )
    
    # Convert features to numpy float64
    all_properties = np.array(all_properties, dtype=np.float64)
    
    # Cluster label prediction
    labels = kmeans.predict(all_properties)  # Predict labels for all graphs in the batch
    
    # Add the labels to the properties
    all_properties_with_labels = np.hstack((all_properties, labels.reshape(-1, 1)))
    
    # Transform back to a tensor of size (batch_size, nfeatures + 1)
    return torch.tensor(all_properties_with_labels, dtype=torch.float32).to(adj.device)



## testing script
if __name__ == "__main__":
    print(f"Visualizing the Test dataset")
    dataset = 'test'
    n_max_nodes = 50
    spectral_emb_dim = 10
    data_lst = preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim)
    print(len(data_lst))
    print(data_lst[0])
    print(data_lst[0].stats)# tensor of size 1x7
    # print(data_lst[0].prompt)
    print(data_lst[0].filename) #file name or index of the graph
    print("-------------------")
    print(f"Visualizing the Train dataset")
    dataset = 'train'
    n_max_nodes = 50
    spectral_emb_dim = 10
    data_lst = preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim,normalize=True)
    print(len(data_lst))
    print(data_lst[0])
    print('\n')
    print(data_lst[0].x) # tensor of shape num_nodes, spectral_emb_dim+1
    print('\n')
    print(data_lst[0].edge_index) # tensor of shape 2 x num_edges
    print('\n')
    print(data_lst[0].A) # tensor of shape 1 , max nodes, max nodes
    print('\n')
    print(data_lst[0].stats)# tensor of size 1x7
    print('\n')
    print(data_lst[0].filename) #file name or index of the graph
    