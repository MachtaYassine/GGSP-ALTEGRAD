from datetime import datetime
import numpy as np
import torch
from NGG.denoiser.denoise_model import p_losses

def train_denoise(args, denoise_model, autoencoder,optimizer,scheduler, train_loader, val_loader, device, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
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
                loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="l2")
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
                loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="l2")
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
    return denoise_model