from datetime import datetime
import numpy as np
import torch
from NGG.denoiser.denoise_model import p_losses

def train_denoise(args, denoise_model, autoencoder,optimizer,scheduler, train_loader, val_loader, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    print(f"Training denoising model, progress will be printed every 5 epochs")
    # Train denoising model
    if args.train_denoiser:
        best_val_loss = np.inf
        for epoch in range(1, args.epochs_denoise+1):
            denoise_model.train()
            autoencoder.eval()
            train_loss_all = 0
            train_count = 0
            train_loss_constraint = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss_dict = p_losses(denoise_model, x_g, t, data, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,args.constrain_denoiser,autoencoder, loss_type="l2")
                loss = loss_dict["loss_total"]
                loss.backward()
                train_loss_all += x_g.size(0) * loss.item()
                train_count += x_g.size(0)
                optimizer.step()
                
                if args.constrain_denoiser:
                    train_loss_constraint += loss_dict["loss_recon"].item() * x_g.size(0)

            denoise_model.eval()
            autoencoder.eval()
            val_loss_all = 0
            val_count = 0
            val_loss_constraint = 0
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, data, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,args.constrain_denoiser,autoencoder, loss_type="l2")
                loss = loss_dict["loss_total"]
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)
                if args.constrain_denoiser:
                    val_loss_constraint += loss_dict["loss_recon"].item() * x_g.size(0)

            if epoch % 5 == 0:
                dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                if args.constrain_denoiser:
                    print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Train Loss Constraint: {:.5f}, Val Loss Constraint: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count, train_loss_constraint/train_count, val_loss_constraint/val_count))
                else:
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
    return denoise_model