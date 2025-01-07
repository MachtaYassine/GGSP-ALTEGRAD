import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from NGG.train_utils.train_autoencoder import train_autoencoder
from NGG.train_utils.denoiser_train import train_denoise
from NGG.train_utils.check_results import check_results
from NGG.autoencoders.autoencoder_base import VariationalAutoEncoder
from NGG.autoencoders.autoencoder_concat import VariationalAutoEncoder_concat
from NGG.autoencoders.autoencoder_GMVAE import GMVAE
from NGG.utils.utils import preprocess_dataset, linear_beta_schedule
from NGG.train_utils.load_or_not_deepset import load_or_not_deepset
from NGG.train_utils.load_autoencoder import load_autoencoder
from NGG.train_utils.parser import parse_train_arguments
from NGG.denoiser.denoise_model import DenoiseNN

def run_test():
    # Set test hyperparameters
    args = parse_train_arguments()
    args.epochs_autoencoder = 1  # Reduce epochs for quick testing
    args.epochs_denoise = 1  # Reduce epochs for quick testing
    args.batch_size = 16  # Smaller batch size for faster iteration
    args.lr = 1e-3  # Learning rate
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # VAE models to test
    vae_models = {
        # "concat": VariationalAutoEncoder_concat,
        "GMVAE": GMVAE,
    }
    
    
    
    for name, VAE_class in vae_models.items():
        args.AE = name
        print(f"Testing VAE model: {name}")
        if name == 'GMVAE':
            args.labelize = True    
            args.contrastive_hyperparameters=[1,0.1]
        print(args.labelize)
        
        # Load and preprocess datasets
        if args.labelize:
            trainset, kmeans = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
            validset, _ = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
            testset, _ = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
        else:
            trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
            validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
            testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
            kmeans = None
        
        
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        
        deepsets = load_or_not_deepset(args, device)
        print(f"DeepSets state: {'Enabled' if deepsets else 'Disabled'}")
        autoencoder = load_autoencoder(args, VAE_class,name, kmeans, device, deepsets)
        print(f"Autoencoder state: {'Enabled' if autoencoder else 'Disabled'}")
        optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        
        # Train the autoencoder
        autoencoder = train_autoencoder(args, autoencoder, train_loader, val_loader, device, optimizer, scheduler)
        
        # Define beta schedule and related calculations for diffusion process
        betas = linear_beta_schedule(timesteps=args.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # Initialize and train the denoising model
        denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
        optimizer = optim.Adam(denoise_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        
        denoise_model = train_denoise(args, denoise_model, autoencoder, optimizer, scheduler, train_loader, val_loader, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        denoise_model.eval()
        
        # Check results
        check_results(args, device, autoencoder, denoise_model, test_loader, testset, betas)
        
        print(f"Completed testing for VAE model: {name}\n")

if __name__ == "__main__":
    run_test()
