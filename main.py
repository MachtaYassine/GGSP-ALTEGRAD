import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder_concat, GMVAE
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

import sys

from torch.utils.data import Subset
np.random.seed(13)

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

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

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
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

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

# Beta for KLD loss weight
parser.add_argument('--beta', type=float, default=0.05, help="Weight for the KLD loss term in the total loss calculation (default: 0.05)")

# Contrastive hyperparameters
parser.add_argument('--contrastive-hyperparameters', type=float, nargs=2, default=None, 
                    help="Two hyperparameters for contrastive loss: contrastive and entropy weights (default: None)")

# Penalization hyperparameter
parser.add_argument('--penalization-hyperparameters', type=float, default=None, 
                    help="Hyperparameter weight for the adjacency penalization term (default: None)")

# Number of categories for the GMVAE model
parser.add_argument('--n-cat', type=int, default=3, help="Number of gaussians in the mixture model (default: 3)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim, args.normalize, args.labelize)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
if args.feature_concat:
    autoencoder = VariationalAutoEncoder_concat(
        args.spectral_emb_dim+1, 
        args.hidden_dim_encoder, 
        args.hidden_dim_decoder, 
        args.latent_dim, 
        args.n_layers_encoder, 
        args.n_layers_decoder, 
        args.n_max_nodes,
        args.labelize,
        args.normalize,
    ).to(device)
else:
    if not args.labelize:  # Check if --labelize argument is specified
        raise ValueError("If using GMVAE, you need to labelize your data by specifying --labelize.")
    autoencoder = GMVAE(
        args.spectral_emb_dim+1, 
        args.hidden_dim_encoder, 
        args.hidden_dim_decoder, 
        args.latent_dim, 
        args.n_layers_encoder, 
        args.n_layers_decoder, 
        args.n_max_nodes,
        args.n_cat,
    ).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


# Train VGAE model
if args.train_autoencoder:
    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_trackers = {}
        val_loss_trackers = {}
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Call loss function
            if args.feature_concat:
                loss_dict = autoencoder.loss(
                    data,
                    args.beta,
                    args.contrastive_hyperparameters,
                    args.penalization_hyperparameters,
                    )
            else:
                loss_dict = autoencoder.loss(
                    data,
                    args.beta,
                    args.contrastive_hyperparameters,
                    )
            
            # Aggregate loss values dynamically
            for key, value in loss_dict.items():
                if key not in train_loss_trackers:
                    train_loss_trackers[key] = 0.0
                train_loss_trackers[key] += value.item()

            # Backpropagation
            loss_dict["loss"].backward()
            optimizer.step()

        # Calculate averages for the epoch
        train_count = len(train_loader.dataset)
        for key in train_loss_trackers:
            train_loss_trackers[key] /= train_count

        # Validation
        autoencoder.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                # Call loss function
                if args.feature_concat:
                    loss_dict = autoencoder.loss(
                        data,
                        args.beta,
                        args.contrastive_hyperparameters,
                        args.penalization_hyperparameters,
                        )
                else:
                    loss_dict = autoencoder.loss(
                        data,
                        args.beta,
                        args.contrastive_hyperparameters,
                        )

                # Aggregate validation loss values
                for key, value in loss_dict.items():
                    if key not in val_loss_trackers:
                        val_loss_trackers[key] = 0.0
                    val_loss_trackers[key] += value.item()

        # Calculate averages for validation
        val_count = len(val_loader.dataset)
        for key in val_loss_trackers:
            val_loss_trackers[key] /= val_count

        # Print losses dynamically
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        loss_str = ", ".join(
            [f"{key.capitalize()} Loss: {value:.5f}" for key, value in train_loss_trackers.items()]
        )
        print(f"{dt_t} Epoch: {epoch:04d}, Train {loss_str}")
        loss_str = ", ".join(
            [f"{key.capitalize()} Loss: {value:.5f}" for key, value in val_loss_trackers.items()]
        )
        print(f"{dt_t} Epoch: {epoch:04d}, Val {loss_str}")

        scheduler.step()

        if best_val_loss >= val_loss_trackers["loss"]:
            best_val_loss = val_loss_trackers["loss"]
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
            
        if early_stop_counter >= 20:
            break
        
        if epoch > 20 and best_val_loss < val_loss_trackers["loss"]:
            early_stop_counter += 1
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()



# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

del train_loader, val_loader

# Save to a CSV file
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        
        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        # print(x_sample.shape)
        # sys.exit()

        if args.feature_concat:
            if args.labelize:
                labels = torch.tensor([data[i].label for i in range(len(data))], device=x_sample.device)  # Shape: (batch_size,)
                x_sample = torch.cat((x_sample, stat, labels.unsqueeze(1)), dim=1) 
            else:
                x_sample = torch.cat((x_sample, stat), dim=1) 
        else:
            # Assume labels from the test dataset
            labels = torch.tensor([data[i].label for i in range(len(data))], device=x_sample.device)  # Shape: (batch_size,)

            # Create one-hot encodings for the labels
            y_onehot = torch.zeros(labels.size(0), args.n_cat, device=labels.device)
            y_onehot.scatter_(1, labels.unsqueeze(1).long(), 1)

            # Concatenate z and y_onehot
            x_sample = torch.cat([x_sample, stat, y_onehot], dim=1)

        adj = autoencoder.decode_mu(x_sample)
        stat_d = torch.reshape(stat, (-1, args.n_condition))


        for i in range(stat.size(0)):
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])