import pandas as pd
import networkx as nx
import ast
from typing import List
from utils import preprocess_dataset, Data
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse


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

# Load graphs into a DataFrame
def load_graphs_into_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Convert edge list to adjacency matrix and store in a DataFrame column
    def edge_list_to_adj_matrix(edge_list_str): 
        edge_list = ast.literal_eval(edge_list_str)
        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        return adj_matrix

    df['adj_matrix'] = df['edge_list'].apply(edge_list_to_adj_matrix)
    return df

# Vectorized feature computation
def compute_features_vectorized(df):
    # Convert adjacency matrices back to graphs and compute features
    df['features'] = df['adj_matrix'].apply(compute_graph_features_from_adj)
    feature_names = [
        "n_nodes", "n_edges", "avg_degree", 
        "n_triangles", "clustering_coeff", "max_k_core", "n_communities"
    ]

    # Split features into separate columns
    features_df = pd.DataFrame(df['features'].tolist(), columns=feature_names)
    return pd.concat([df.drop(columns=['features']), features_df], axis=1)

def compare_reconstructed_and_prompted_graphs(result_df: pd.DataFrame, data_lst: List[Data],csv_path: str):
    diff_nodes, diff_edges, diff_avg_degree, diff_triangles, diff_clustering_coeff, diff_max_core, diff_communities = [], [], [], [], [], [], []
    MSE_vectors=[]
    
    for i, data in enumerate(data_lst):
        filename=data.filename
        stats=data.stats
        #get the row of the dataframe where graph_id == filename
        row = result_df[result_df['graph_id'] == filename]
        if row.empty:
            print(f"Graph {filename} not found in the DataFrame")
            continue
        
        diff_nodes.append(stats[0,0] - row['n_nodes'].values[0])
        diff_edges.append(stats[0,1] - row['n_edges'].values[0])
        diff_avg_degree.append(stats[0,2] - row['avg_degree'].values[0])
        diff_triangles.append(stats[0,3] - row['n_triangles'].values[0])
        diff_clustering_coeff.append(stats[0,4] - row['clustering_coeff'].values[0])
        diff_max_core.append(stats[0,5] - row['max_k_core'].values[0])
        diff_communities.append(stats[0,6] - row['n_communities'].values[0])
        
        #compute MSe for the entire vector
        MSE_vector=np.square(stats - row[['n_nodes', 'n_edges', 'avg_degree', 'n_triangles', 'clustering_coeff', 'max_k_core', 'n_communities']].values[0])
        
    MSE_vectors.append(MSE_vector)
        
    print(f"Mean difference in number of nodes: {sum(diff_nodes) / len(diff_nodes)}")
    print(f"Mean difference in number of edges: {sum(diff_edges) / len(diff_edges)}")
    print(f"Mean difference in average degree: {sum(diff_avg_degree) / len(diff_avg_degree)}")
    print(f"Mean difference in number of triangles: {sum(diff_triangles) / len(diff_triangles)}")
    print(f"Mean difference in clustering coefficient: {sum(diff_clustering_coeff) / len(diff_clustering_coeff)}")
    print(f"Mean difference in max k-core: {sum(diff_max_core) / len(diff_max_core)}")
    print(f"Mean difference in number of communities: {sum(diff_communities) / len(diff_communities)}")
    
    #plot the differences in one major plot using subplots
    
    # Create subplots
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    # Data and titles
    data_and_titles = [
        (diff_nodes, 'Difference in number of nodes', axs[0, 0]),
        (diff_edges, 'Difference in number of edges', axs[0, 1]),
        (diff_avg_degree, 'Difference in average degree', axs[1, 0]),
        (diff_triangles, 'Difference in number of triangles', axs[1, 1]),
        (diff_clustering_coeff, 'Difference in clustering coefficient', axs[2, 0]),
        (diff_max_core, 'Difference in max k-core', axs[2, 1]),
        (diff_communities, 'Difference in number of communities', axs[3, 0]),
    ]

    # Plot each histogram with mean line and improved aesthetics
    for diff, title, ax in data_and_titles:
        ax.hist(diff, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        mean_value = np.mean(diff)
        ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('Difference', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Turn off the unused subplot
    axs[3, 1].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(csv_path.replace(".csv", "_differences_with_prompted_features_plot.png"))
    print(f"Plot saved as ",csv_path.replace(".csv", "_differences_with_prompted_features_plot.png"))
    
    
    #Average MSE for the entire dataset
    MSE_vectors=np.array(MSE_vectors)
    MSE_vectors=MSE_vectors.squeeze()
    MSE_vectors=MSE_vectors.mean(axis=0)
    print(f"Average MSE for the entire dataset: {MSE_vectors}") # Clearly this is wrong caus eour entry on kagggle sits at 0.89 whereas this is 1889.88....
    
    




def main():
    parser = argparse.ArgumentParser(description='Process a CSV file for graph features.')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    args = parser.parse_args()

    csv_path = args.csv if args.csv else "progression_archive/attempt_1/output.csv"  # Default example path
    if not os.path.isfile(csv_path.replace(".csv", "_with_features.csv")):
        graphs_df = load_graphs_into_dataframe(csv_path)
        result_df = compute_features_vectorized(graphs_df)
        result_df.to_csv(csv_path.replace(".csv", "_with_features.csv"), index=False)
    else:
        result_df = pd.read_csv(csv_path.replace(".csv", "_with_features.csv"))
        
    data_lst = preprocess_dataset("test", 50, 10)
    
    compare_reconstructed_and_prompted_graphs(result_df, data_lst, csv_path)

if __name__ == "__main__":
    main()


