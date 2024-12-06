import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import preprocess_dataset, Data
from typing import List

def get_features_distribution_per_dataset(data: List[Data]):
    list_of_nodes = []
    list_of_edges = []
    list_of_avg_degree = []
    list_of_triangles = []
    list_of_clustering_coeff = []
    list_of_max_core = []
    list_of_communities = []
    for graph in data:
        list_of_nodes.append(graph.stats[0,0].item())
        list_of_edges.append(graph.stats[0,1].item())
        list_of_avg_degree.append(graph.stats[0,2].item())
        list_of_triangles.append(graph.stats[0,3].item())
        list_of_clustering_coeff.append(graph.stats[0,4].item())
        list_of_max_core.append(graph.stats[0,5].item())
        list_of_communities.append(graph.stats[0,6].item())
        
    return list_of_nodes, list_of_edges, list_of_avg_degree, list_of_triangles, list_of_clustering_coeff, list_of_max_core, list_of_communities

def main():
    train_data = preprocess_dataset("train",50,10)
    test_data = preprocess_dataset("test",50,10)
    val_data = preprocess_dataset("valid",50,10)
    
    
    list_of_nodes_train, list_of_edges_train, list_of_avg_degree_train, list_of_triangles_train, list_of_clustering_coeff_train, list_of_max_core_train, list_of_communities_train = get_features_distribution_per_dataset(train_data)
    list_of_nodes_test, list_of_edges_test, list_of_avg_degree_test, list_of_triangles_test, list_of_clustering_coeff_test, list_of_max_core_test, list_of_communities_test = get_features_distribution_per_dataset(test_data)
    list_of_nodes_val, list_of_edges_val, list_of_avg_degree_val, list_of_triangles_val, list_of_clustering_coeff_val, list_of_max_core_val, list_of_communities_val = get_features_distribution_per_dataset(val_data)
    
    #plot overlayed histograms blue is train,green is val,red is test
    fig, axs = plt.subplots(2, 4, figsize=(15, 15))
    fig.suptitle(
        'Feature Distributions per Dataset\n'
        'We should not be having any "out of distribution problems"', 
        fontsize=16
    )

    # Data and titles for the subplots
    features = [
        (list_of_nodes_train, list_of_nodes_test, list_of_nodes_val, 'Number of Nodes'),
        (list_of_edges_train, list_of_edges_test, list_of_edges_val, 'Number of Edges'),
        (list_of_avg_degree_train, list_of_avg_degree_test, list_of_avg_degree_val, 'Average Degree'),
        (list_of_triangles_train, list_of_triangles_test, list_of_triangles_val, 'Number of Triangles'),
        (list_of_clustering_coeff_train, list_of_clustering_coeff_test, list_of_clustering_coeff_val, 'Clustering Coefficient'),
        (list_of_max_core_train, list_of_max_core_test, list_of_max_core_val, 'Max Core'),
        (list_of_communities_train, list_of_communities_test, list_of_communities_val, 'Number of Communities'),
    ]

    # Colors and labels
    colors = ['blue', 'red', 'green']
    labels = ['train', 'test', 'val']

    # Plot each feature
    for idx, (train_data, test_data, val_data, title) in enumerate(features):
        row, col = divmod(idx, 4)
        ax = axs[row, col]
        
        # Plot histograms for train, test, and val sets
        for data, color, label in zip([train_data, test_data, val_data], colors, labels):
            ax.hist(data, bins=50, alpha=0.5, color=color, label=label)
        
        # Set title and legend
        ax.set_title(title, fontsize=12)
        ax.legend()

    # Adjust layout to fit titles and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("feature_distributions_per_dataset.png")
    plt.show()
    
        
        


if __name__ == "__main__":
    main()