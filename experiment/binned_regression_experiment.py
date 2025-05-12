import torch
import torch.nn as nn
import torch.nn.functional as F
import src.machine_learning as ML
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
import hydra
from omegaconf import DictConfig, OmegaConf
import os


class GlyphClassifier(nn.Module):
    def __init__(self, NUM_bins, resolution):
        super(GlyphClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * resolution[0]//8 * resolution[1]//8, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_bins)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)     
        return x

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Getting the dataset
    dataset_file = config.dataset
    test_file = config.test
    train_dataset = ML.GlyphDataset(dataset_file, resize=config.image_resolution, split = "train",augmentation_rot=config.rotation,augmentation_tran=config.translation,stride=config.stride)
    validation_dataset = ML.GlyphDataset(dataset_file, resize=config.image_resolution,split = 'test',augmentation_rot=config.rotation,augmentation_tran=config.translation,stride=config.stride)
    test_dataset = ML.GlyphDataset(test_file, resize=config.image_resolution, split='test',augmentation_rot=config.rotation,augmentation_tran=config.translation,stride=config.stride)

    # Assign the loaders 

    train_loader = ML.create_loader(train_dataset, batch_size=config.batch_size, shuffle = True)
    test_loader = ML.create_loader(test_dataset, batch_size=config.batch_size, shuffle = False)
    validation_loader = ML.create_loader(validation_dataset, batch_size=config.batch_size, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bin_centers = torch.linspace(0, 100, config.num_bins + 1, device=device)[:-1] + 50 / config.num_bins

    model = GlyphClassifier(resolution=config.image_resolution, NUM_bins=config.num_bins).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR by half every 5 epochs

    experiment_name = f"exp-SimpleStar-{config.image_resolution[0]}x{config.image_resolution[1]}-{config.num_bins}bins-BinnedRegression-withvalidation"

    print(f"Experiment name: {experiment_name}")

    # Initialize W&B
    wandb.init(
        project="glyph-regression",
        name=experiment_name,
        config=config
    )
    wandb.watch(model, log="all", log_freq=10)
    

    train_losses = []
    epoch_train_losses = []
    val_losses = []
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, values in train_loader:
            images = images.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.sum(probabilities * bin_centers, dim=1)
            loss = criterion(predictions, values)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(loss.item())

            mae = F.l1_loss(predictions, values).item()
            wandb.log({
                "train_loss_step": loss.item(),
                "train_mae_step": mae,
                "train_mse_step": loss.item(),  # same as MSELoss
                "global_step": global_step
            })

            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss.item():.4f}")
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}")
        wandb.log({ "epoch_train_loss": avg_train_loss })

        # Validation at the end of the epoch 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, values in validation_loader:
                images = images.to(device)
                values = values.to(device)
                logits = model(images)
                probabilities = F.softmax(logits, dim=1)
                preds = torch.sum(probabilities * bin_centers, dim=1)
                val_loss += criterion(preds, values).item()
        
        scheduler.step()

        avg_val_loss = val_loss / len(validation_loader)
        val_losses.append(avg_val_loss)
        wandb.log({ "epoch_val_loss": avg_val_loss })
        print(f"Epoch {epoch+1}/{config.epochs} - Val Loss: {avg_val_loss:.4f}")

    # Plot training vs validation loss in W&B
    wandb.log({
        "losses": wandb.plot.line_series(
            xs=list(range(1, config.epochs + 1)),
            ys=[epoch_train_losses, val_losses],
            keys=["Train Loss", "Validation Loss"],
            title="Training vs Validation Loss",
            xname="Epoch"
        )
    })

    # Local plot (optional)
    ML.plot_training_loss(train_losses, val_losses)


    # --- Final Test Evaluation ---

    model.eval()
    predictions = []
    ground_truths = []
    sample_ids = []

    with torch.no_grad():
        for batch_idx, (images, values) in enumerate(test_loader):
            images = images.to(device)
            values = values.to(device)
            
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            batch_predictions = torch.sum(probabilities * bin_centers, dim=1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            ground_truths.extend(values.cpu().numpy())
            
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + len(images)
            ids = [test_dataset.samples[i]['file'] for i in range(start_idx, end_idx)]
            sample_ids.extend(ids)

    # --- Metrics ---
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    mse = np.mean((predictions - ground_truths)**2)
    mae = np.mean(np.abs(predictions - ground_truths))

    # --- Log to W&B ---
    wandb.log({
        "test_mse": mse,
        "test_mae": mae
    })

    # Dataframe creation
    df_val = pd.DataFrame({
        "sample_id": sample_ids,
        "ground_truth": ground_truths,
        "predicted_value": predictions
    })
    df_val['error'] = np.abs(df_val['ground_truth'] - df_val['predicted_value'])


    print(df_val)
    print(f"\n Final Test MSE: {mse:.4f}")
    print(f" Final Test MAE: {mae:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()