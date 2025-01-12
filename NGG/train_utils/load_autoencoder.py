
from NGG.utils.utils import to_labels

def load_autoencoder(args, VAE_class,VAE_tag,kmeans,device,deepsets):
    if args.contrastive_hyperparameters is not None or args.gmvae_loss_parameters is not None:
        if not args.labelize:
            raise ValueError("If using constrative hyperparamaters or penalization_hyperparameters, you need to labelize your data by specifying --labelize.")                   
    
    
    
    print(f"Loading {VAE_tag} model")
    if VAE_tag=='GMVAE':
        if not args.labelize:
            raise ValueError("If using VGAE, you need to labelize your data by specifying --labelize.")                 
        to_labels_func = lambda x: to_labels(x, kmeans)
        autoencoder = VAE_class(
            args.node_feature_dimension, 
            args.hidden_dim_encoder, 
            args.hidden_dim_decoder, 
            args.latent_dim, 
            args.n_layers_encoder, 
            args.n_layers_decoder, 
            args.n_max_nodes,
            to_labels_func,
            num_clusters=args.n_clusters
        ).to(device)        
        
                     
    else:
        autoencoder = VAE_class(
            args.node_feature_dimension, 
            args.hidden_dim_encoder, 
            args.hidden_dim_decoder, 
            args.latent_dim, 
            args.n_layers_encoder, 
            args.n_layers_decoder, 
            args.n_max_nodes,
            deepsets,
            args.normalize,
            args.no_attention,
        ).to(device)
    
    return autoencoder