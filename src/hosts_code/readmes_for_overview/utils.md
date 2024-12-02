# Script Overview

This script contains a series of functions for processing graph datasets, including data preprocessing, feature extraction, and graph normalization. Here's a breakdown of the key parts:

### 1. **Preprocess Dataset (`preprocess_dataset`)**
   - This function handles loading and preprocessing a graph dataset. It supports two modes:
     - **Test mode**: Loads preprocessed data if available, or processes the data and saves it.
     - **Other datasets**: Loads and processes graphs from the dataset folder, computes graph features (such as eigenvalues, adjacency matrices, etc.), and saves the results.
   - For each graph, it:
     - Reads graph data (in formats like `GraphML` or `edgelist`) using `NetworkX`.
     - Extracts spectral features such as the Laplacian and its eigenvectors.
     - Creates a normalized adjacency matrix and stores features for each graph.

### 2. **Construct NetworkX Graph from Adjacency Matrix (`construct_nx_from_adj`)**
   - This function converts an adjacency matrix into a `NetworkX` graph and removes isolated nodes (nodes with zero degree).

### 3. **Handle NaN Values (`handle_nan`)**
   - This function replaces any `NaN` values with a predefined constant value (`-100`).

### 4. **Masked Normalization Functions (`masked_instance_norm2D`, `masked_layer_norm2D`)**
   - These functions implement instance and layer normalization in 2D using masks to ignore certain values. They are useful for normalizing graph data or batches where certain entries might be missing.

### 5. **Beta Schedule Functions (`cosine_beta_schedule`, `linear_beta_schedule`, `quadratic_beta_schedule`, `sigmoid_beta_schedule`)**
   - These functions generate different beta schedules for diffusion models. They control how the model's variance evolves over time, which is useful for training generative models, such as denoising diffusion models.

### 6. **Dependencies**
   - The script uses libraries like:
     - **PyTorch**: For tensor operations and neural network components.
     - **NetworkX**: For graph manipulation and analysis.
     - **Grakel**: For graph kernels.
     - **Scipy/NumPy**: For scientific computing and sparse matrix operations.
     - **Torch Geometric**: For working with graph-structured data in PyTorch.

### Key Points:
   - **Graph Loading & Preprocessing**: Handles graph files, computes features, and saves them.
   - **Feature Extraction**: Uses `extract_feats` and `extract_numbers` to extract statistics and numerical features from graph descriptions.
   - **Graph Normalization**: Provides normalization techniques for graphs with missing values.
   - **Beta Schedules**: Implements several beta schedules for controlling the noise schedule in diffusion models.


