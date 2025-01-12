
import argparse

def parse_train_arguments():
    """
    Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
    settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
    parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

    Returns:
        argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
    """

    # Argument parser
    parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

    # Argument parser for configuring the NeuralGraphGenerator model
    parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

    # autoencoder model tag
    parser.add_argument('--AE', type=str, required=True, help="Autoencoder model tag: 'base', 'concat', 'features' (not implemented yet) or 'GMVAE'")


    # name of this attempt
    parser.add_argument('--name', type=str, required=True, help="Name of this attempt this will save the results in progression archive and note down the MSE")

    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

    # Batch size for training
    parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

    # Wether to use DeepSets as feature aggregation
    parser.add_argument('--deepsets', action='store_true', default=False, help="Flag to enable/disable DeepSets training (default: disabled)")

    # Number of epochs for Deepsets training
    parser.add_argument('--epochs-deepsets', type=int, default=200, help="Number of training epochs for DeepSets (default: 30)")

    # Number of epochs for the autoencoder training
    parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

    # Hidden dimension size for the encoder network
    parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

    # Hidden dimension size for the decoder network
    parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

    # Dimensionality of the latent space
    parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")

    # Maximum number of nodes of graphs
    parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

    # Number of layers in the encoder network
    parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

    # Number of layers in the decoder network
    parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

    # Dimensionality of spectral embeddings for graph structure representation
    parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

    # Number of training epochs for the denoising model
    parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

    # Number of timesteps in the diffusion
    parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

    # Hidden dimension size for the denoising model
    parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

    # Number of layers in the denoising model
    parser.add_argument('--n-layers-denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

    # Flag to toggle training of the autoencoder (VGAE)
    parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

    # Flag to toggle training of the diffusion-based denoising model
    parser.add_argument('--train-denoiser', action='store_false', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

    # Dimensionality of conditioning vectors for conditional generation
    parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

    # Number of conditions used in conditional vector (number of properties)
    parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

    # Whether to train VAE_concat od GMVAE
    parser.add_argument('--feature-concat', action='store_true', default=False, help="Use GMVAE model by default, other is concat (default: disabled)")

    parser.add_argument('--normalize', action='store_true', default=False, help="Flag to enable/disable normalization of adjacency matrix (default: disabled)")

    # Labelize for contrastive learning
    parser.add_argument('--labelize', action='store_true', default=False, help="Flag to enable/disable labelization of graphs into clusters (default: disabled)")
    
    parser.add_argument('--additional', action='store_true', default=False, help="Flag to enabladding additional features to the nodes (default: disabled)")
    
    parser.add_argument('--constrain-denoiser', action='store_true', default=False, help="Flag to enable/disable constraining the denoiser")
    
    parser.add_argument('--no-attention', action='store_false', default=True, help="Flag to enable/disable attention mechanism in the encoder (default: enabled)")
    # Early stopping of VAE training
    parser.add_argument('--early-stopping', action='store_true', default=False, help="Flag to enable/disable early stopping of VAE training (default: disabled)")


    # Beta for KLD loss weight
    parser.add_argument('--beta', type=float, default=0.05, help="Weight for the KLD loss term in the total loss calculation (default: 0.05)")

    # Contrastive hyperparameters
    parser.add_argument('--contrastive-hyperparameters', type=float, nargs=2, default=None, 
                        help="Two hyperparameters for contrastive loss: contrastive and entropy weights []")

    # Penalization hyperparameter
    parser.add_argument('--penalization-hyperparameters', type=float, default=None, 
                        help="Hyperparameter weight for the adjacency penalization term []")

    # GMVAE loss hyperparameters
    parser.add_argument('--gmvae-loss-parameters', type=float, nargs=5, default=None, 
                        help="[con_temperature, alpha_bce, bce_weight, kl_weight, mse_weight] (default: None)")

    # Number of clusters
    parser.add_argument('--n-clusters', type=int, default=3, 
                        help="In how many clusters we separate the graphs features (default: 3)")

    args = parser.parse_args()
    return args