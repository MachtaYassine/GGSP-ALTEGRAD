import os
import torch
import torch.nn.functional as F
from NGG.autoencoders.autoencoder_GMVAEv2 import GraphStatisticsModel


def load_or_not_stat_model(args, train_loader, device):
    # Train GraphStatisticsModel for features prediction from adjency matrix
    if args.stats_model:
        # File path for saving the model
        model_path = f'stat_model_{args.epochs_autoencoder}_{args.batch_size}.pth.tar'

        # Check if the model already exists
        if os.path.exists(model_path):
            print(f"Model found at {model_path}. Loading the saved model...")
            checkpoint = torch.load(model_path)
            n_features = args.n_condition + args.labelize
            stat_model = GraphStatisticsModel(args.n_max_nodes**2, 8192, n_features).to(device)
            stat_model.load_state_dict(checkpoint['state_dict'])
            optimizer_stat_model = torch.optim.Adam(stat_model.parameters(), lr=args.lr)
            optimizer_stat_model.load_state_dict(checkpoint['optimizer'])
            stat_model.eval()
            print("Loaded the saved DeepSets model.")
        else:
            print("No saved model found. Starting training from scratch...")

            # Create training data
            n_features = args.n_condition + args.labelize
            stat_model = GraphStatisticsModel(args.n_max_nodes**2, 8192, n_features).to(device)
            optimizer = torch.optim.Adam(stat_model.parameters(), lr=args.lr)

            # Training loop
            for epoch in range(args.epochs_autoencoder):
                stat_model.train()
                train_loss, count = 0.0, 0
                for data in train_loader:
                    data = data.to(device)
                    cluster_labels = torch.tensor([data[i].label for i in range(len(data))], device=device)
                    stats = torch.cat((data.stats, cluster_labels.unsqueeze(1)), 1)
                    optimizer.zero_grad()
                    output = stat_model(data.A)
                    loss = F.l1_loss(output, stats, reduction="mean")
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * output.size(0)
                    count += output.size(0)

                print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(train_loss / count))

            # Save the trained model to disk
            torch.save({
                'state_dict': stat_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)
            stat_model.eval()
            print("Finished training for Stats model")
            print()
    else:
        stat_model = None
        
    return stat_model