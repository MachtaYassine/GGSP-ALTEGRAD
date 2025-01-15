import torch 
import torch.nn as nn
import os

from NGG.autoencoders.components.deepsets import DeepSets
from NGG.utils.utils import create_deepsets_train_dataset

def load_or_not_deepset(args, device):
    # Train Deepset for aggregation of features
    if args.deepsets:
        # File path for saving the model
        model_path = f'model_deepsets_{args.epochs_deepsets}_{args.hidden_dim_encoder}_{args.batch_size}.pth.tar'

        # Check if the model already exists
        if os.path.exists(model_path):
            print(f"Model found at {model_path}. Loading the saved model...")
            checkpoint = torch.load(model_path)
            deepsets = DeepSets(args.hidden_dim_encoder).to(device)
            deepsets.load_state_dict(checkpoint['state_dict'])
            optimizer_deepset = torch.optim.Adam(deepsets.parameters(), lr=args.lr)
            optimizer_deepset.load_state_dict(checkpoint['optimizer'])
            deepsets.eval()
            print("Loaded the saved DeepSets model.")
        else:
            print("No saved model found. Starting training from scratch...")

            # Create training data
            X_train, y_train, batch = create_deepsets_train_dataset(args.hidden_dim_encoder, args.batch_size, device)
            deepsets = DeepSets(args.hidden_dim_encoder).to(device)
            optimizer_deepset = torch.optim.Adam(deepsets.parameters(), lr=args.lr)
            loss_function = nn.L1Loss()

            # Training loop
            for epoch in range(args.epochs_deepsets):
                deepsets.train()

                optimizer_deepset.zero_grad()
                output = deepsets(X_train, batch)
                loss = loss_function(output, y_train)
                loss.backward()
                optimizer_deepset.step()
                train_loss = loss.item() * output.size(0)
                count = output.size(0)

                print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(train_loss / count))

            # Save the trained model to disk
            torch.save({
                'state_dict': deepsets.state_dict(),
                'optimizer': optimizer_deepset.state_dict(),
            }, model_path)
            deepsets.eval()
            print("Finished training for DeepSets model")
            print()
    else:
        deepsets = None
        
    return deepsets