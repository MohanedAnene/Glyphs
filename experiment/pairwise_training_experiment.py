import sys
import os

# Import from a sibling folder 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import src.machine_learning as ML
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from torch.optim.lr_scheduler import StepLR
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from scipy.stats import spearmanr



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


def save_plot_with_subfolders(fig, base_name, root_dir="plots"):
    """
    Save plot to subfolders based on name structure.
    Handles names with '/' by creating subfolders.
    """
    # Remove .png if present
    base_name = base_name.replace('.png', '')
    
    # Split into path components
    if '/' in base_name:
        subfolder, filename = base_name.rsplit('/', 1)
        plot_dir = os.path.join(root_dir, subfolder)
    else:
        plot_dir = root_dir
        filename = base_name
    
    # Create directory structure if needed
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate full path and handle duplicates
    filepath = os.path.join(plot_dir, f"{filename}.png")
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(plot_dir, f"{filename}_{counter}.png")
        counter += 1
    
    # Save the figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[INFO] Plot saved to {filepath}")
    return filepath

def set_seed(seed):
    random.seed(seed)                      # Python built-in
    np.random.seed(seed)                   # Numpy
    torch.manual_seed(seed)                # CPU-based PyTorch ops
    torch.cuda.manual_seed(seed)           # CUDA GPU ops
    torch.cuda.manual_seed_all(seed)       # If multi-GPU
    torch.backends.cudnn.deterministic = True  # Force deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable performance auto-tuning (nondeterministic)

def save_scatter_plot(pred_distances, true_distances, losses, base_name, root_dir="plots"):
    """
    Create and save a scatter plot of true distances vs loss values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(true_distances, losses, alpha=0.5, s=10)
    plt.xlabel("True Distance Between Pairs")
    plt.ylabel("Computed Loss")
    plt.title("Loss vs True Pair Distance")
    plt.grid(True, alpha=0.3)
    
    # Modify base name for scatter plot
    if '/' in base_name:
        parts = base_name.rsplit('/', 1)
        scatter_name = f"{parts[0]}/scatter_{parts[1]}"
    else:
        scatter_name = f"scatter_{base_name}"
    
    save_plot_with_subfolders(plt, scatter_name, root_dir)
    plt.close()





@hydra.main(version_base=None, config_path="../cfgs", config_name="config")
def main(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)
    set_seed(config.seed)  # You will add this to the config next

    # Safe defaults for decay
    config.margin_decay = getattr(config, "margin_decay", 1.0)
    config.maxdistance_decay = getattr(config, "maxdistance_decay", 1.0)

    dataset_file = config.dataset
    test_file = config.test

    pairwise_train_dataset = ML.PairwiseGlyphDataset(dataset_file, resize=config.image_resolution, split="train",
        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)
    pairwise_validation_dataset = ML.PairwiseGlyphDataset(dataset_file, resize=config.image_resolution, split="test",
        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)
    test_dataset_eval = ML.GlyphDataset(zip_path=config.test, resize=config.image_resolution, split="test",
                                        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)
    


    pairwise_validation_dataset.make_pairs(N=config.num_pairs, max_distance=config.max_distance)
    validation_loader = ML.create_loader(pairwise_validation_dataset, batch_size=config.batch_size//2, shuffle=True)
    test_loader_eval = ML.create_loader(test_dataset_eval, batch_size=config.batch_size, shuffle=False)

    def pairwise_hinge_loss(pred1, pred2, true1, true2, margin=config.margin, lambda_range=0.01):
        direction = torch.sign(true1 - true2)
        ranking_loss = torch.clamp(margin - direction * (pred1 - pred2), min=0)
        range_penalty = (F.relu(-pred1) + F.relu(pred1 - 100) + F.relu(-pred2) + F.relu(pred2 - 100))
        return ranking_loss.mean() + lambda_range * range_penalty.mean()
    
    def collect_scatter_data(model, loader, device, bin_centers, margin):
        true_distances = []
        losses = []
        pred_distances = []
        
        model.eval()
        with torch.no_grad():
            for img1, val1, img2, val2 in loader:
                img1, val1 = img1.to(device), val1.to(device)
                img2, val2 = img2.to(device), val2.to(device)
                
                # Get predictions
                out1, out2 = model(img1), model(img2)
                prob1, prob2 = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
                pred1, pred2 = torch.sum(prob1 * bin_centers, dim=1), torch.sum(prob2 * bin_centers, dim=1)
                
                # Compute distances
                true_dist = torch.abs(val1 - val2)
                pred_dist = torch.abs(pred1 - pred2)
                
                # Compute loss
                loss = pairwise_hinge_loss(pred1, pred2, val1, val2, margin=margin)
                
                # Store values
                true_distances.extend(true_dist.cpu().numpy())
                pred_distances.extend(pred_dist.cpu().numpy())
                losses.extend([loss.item()] * len(true_dist))
        
        return pred_distances, true_distances, losses
    
    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bin_centers = torch.linspace(0, 100, config.num_bins + 1, device=device)[:-1] + 50 / config.num_bins
    model = GlyphClassifier(resolution=config.image_resolution, NUM_bins=config.num_bins).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=config.stepsize, gamma=config.learning_decay)
    experiment_name = config.name
    print(f"Experiment name: {experiment_name}")



    train_losses = []
    epoch_train_losses = []
    val_losses = []
    val_maes = []
    global_step = 0
    original_margin = config.margin
    original_max_distance = config.max_distance


    for epoch in range(config.epochs):
        if config.margin_decay < 1.0:
            config.margin *= config.margin_decay
        if config.maxdistance_decay < 1.0:
            config.max_distance *= config.maxdistance_decay

        pairwise_train_dataset.make_pairs(N=config.num_pairs, max_distance=config.max_distance)
        train_loader = ML.create_loader(pairwise_train_dataset, batch_size=config.batch_size//2, shuffle=True)

        model.train()
        running_loss = 0.0

        for img1, val1, img2, val2 in train_loader:
            img1, val1 = img1.to(device), val1.to(device)
            img2, val2 = img2.to(device), val2.to(device)
            out1, out2 = model(img1), model(img2)
            prob1, prob2 = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
            pred1, pred2 = torch.sum(prob1 * bin_centers, dim=1), torch.sum(prob2 * bin_centers, dim=1)
            loss = pairwise_hinge_loss(pred1, pred2, val1, val2, margin=config.margin)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item(); train_losses.append(loss.item())
            if global_step % 100 == 0:
                print(f"Step {global_step}: Pairwise Loss = {loss.item():.4f}")
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        model.eval(); val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for img1, val1, img2, val2 in validation_loader:
                img1, val1, img2, val2 = img1.to(device), val1.to(device), img2.to(device), val2.to(device)
                out1, out2 = model(img1), model(img2)
                prob1, prob2 = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
                pred1, pred2 = torch.sum(prob1 * bin_centers, dim=1), torch.sum(prob2 * bin_centers, dim=1)
                val_loss += pairwise_hinge_loss(pred1, pred2, val1, val2).item()
                val_mae += F.l1_loss(pred1, val1).item() + F.l1_loss(pred2, val2).item()

        val_losses.append(val_loss / len(validation_loader))
        val_maes.append(val_mae / (2 * len(validation_loader)))
        print(f"Epoch {epoch+1} - Val Loss: {val_losses[-1]:.4f} | Val MAE: {val_maes[-1]:.4f}")
        scheduler.step()

        if epoch == config.epochs - 1:  # Only on last epoch
            pred_dists, true_dists, loss_vals = collect_scatter_data(
                model, validation_loader, device, bin_centers, config.margin
            )
            save_scatter_plot(pred_dists, true_dists, loss_vals, config.name)
        

    # Create figure with adjusted layout
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.3)  # Make room for text at bottom

    # Fixed scale values
    LOSS_Y_MAX = 15.0  # Fixed maximum for loss axis
    MAE_Y_MAX = 12.0   # Fixed maximum for MAE axis

    # Plot training and validation loss
    ax1.plot(epoch_train_losses, label='Training Loss', color='blue', alpha=0.6, linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, LOSS_Y_MAX)  # Fixed scale
    ax1.set_xlim(0, len(epoch_train_losses)-1)  # Full epoch range
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot MAE on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(val_maes, label='Validation MAE', color='green', linestyle='--')
    ax2.set_ylabel("MAE")
    ax2.set_ylim(0, MAE_Y_MAX)  # Fixed scale
    ax2.legend(loc='upper right')

    # Get actual learning rates from scheduler
    def get_actual_lrs(scheduler, optimizer, epochs):
        lrs = []
        for _ in range(epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        return lrs

    # Reset scheduler to get actual LR values
    scheduler_copy = StepLR(optimizer, step_size=config.stepsize, gamma=config.learning_decay)
    actual_lrs = get_actual_lrs(scheduler_copy, optimizer, config.epochs)
    lr_5 = actual_lrs[4] if len(actual_lrs) > 4 else 0
    lr_10 = actual_lrs[9] if len(actual_lrs) > 9 else 0
    # Compute decayed values
    margin_5 = original_margin * (config.margin_decay ** 4)
    margin_10 = original_margin * (config.margin_decay ** 9)
    mxd_5 = original_max_distance * (config.maxdistance_decay ** 4)
    mxd_10 = original_max_distance * (config.maxdistance_decay ** 9)


    # Create info text with correct values
    info_text = (
    f"LR: {config.learning_rate:.1e} | LRD: {config.learning_decay} → "
    f"Epoch5: {lr_5:.10e}, Epoch10: {lr_10:.10e}\n"
    f"Margin: {original_margin:.2f} | Margin Decay: {config.margin_decay:.2f} → "
    f"Epoch5: {margin_5:.2f}, Epoch10: {margin_10:.2f}\n"
    f"Max Dist: {original_max_distance:.2f} | MaxDist Decay: {config.maxdistance_decay:.2f} → "
    f"Epoch5: {mxd_5:.2f}, Epoch10: {mxd_10:.2f}"
)


    # === FINAL TEST MAE + SPEARMAN ===
    model.eval()
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for imgs, values in test_loader_eval:
            imgs, values = imgs.to(device), values.to(device)
            out = model(imgs)
            probs = F.softmax(out, dim=1)
            preds = torch.sum(probs * bin_centers, dim=1)
            all_preds.append(preds.cpu())
            all_truths.append(values.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_truths = torch.cat(all_truths).numpy()

    final_mae = np.mean(np.abs(all_preds - all_truths))
    spearman_corr, _ = spearmanr(all_preds, all_truths)

    print(f"Final Test MAE: {final_mae:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    # Extend info text
    info_text += (
        f"\nFinal MAE: {final_mae:.4f} | Spearman: {spearman_corr:.4f}"
    )

    # Add updated text box (slightly to the right)
    fig.text(0.55, 0.05, info_text, ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))


    # Set title without .png
    raw_name = config.name.replace('.png', '')
    plt.title(f"Loss and MAE Over Epochs\n{raw_name}")

    # Save plot using our new function
    save_plot_with_subfolders(fig, config.name)
    plt.close()

    

if __name__ == "__main__":
    main()