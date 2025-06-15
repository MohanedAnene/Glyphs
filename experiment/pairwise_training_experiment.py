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
from torch.optim.lr_scheduler import StepLR
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


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

@hydra.main(version_base=None, config_path="../cfgs", config_name="config")
def main(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Safe defaults for decay
    config.margin_decay = getattr(config, "margin_decay", 1.0)
    config.maxdistance_decay = getattr(config, "maxdistance_decay", 1.0)

    dataset_file = config.dataset
    test_file = config.test

    pairwise_train_dataset = ML.PairwiseGlyphDataset(dataset_file, resize=config.image_resolution, split="train",
        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)
    pairwise_validation_dataset = ML.PairwiseGlyphDataset(dataset_file, resize=config.image_resolution, split="test",
        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)
    test_dataset_eval = ML.GlyphDataset(zip_path=config.test, resize=config.image_resolution, split='test',
        augmentation_rot=config.rotation, augmentation_tran_x=config.translation_x, augmentation_tran_y=config.translation_y)

    pairwise_validation_dataset.make_pairs(N=config.num_pairs, max_distance=config.max_distance)
    validation_loader = ML.create_loader(pairwise_validation_dataset, batch_size=config.batch_size//2, shuffle=True)
    test_loader_eval = ML.create_loader(test_dataset_eval, batch_size=config.batch_size, shuffle=False)

    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bin_centers = torch.linspace(0, 100, config.num_bins + 1, device=device)[:-1] + 50 / config.num_bins
    model = GlyphClassifier(resolution=config.image_resolution, NUM_bins=config.num_bins).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=config.stepsize, gamma=config.learning_decay)
    experiment_name = config.name
    print(f"Experiment name: {experiment_name}")

    def pairwise_hinge_loss(pred1, pred2, true1, true2, margin=config.margin, lambda_range=0.01):
        direction = torch.sign(true1 - true2)
        ranking_loss = torch.clamp(margin - direction * (pred1 - pred2), min=0)
        range_penalty = (F.relu(-pred1) + F.relu(pred1 - 100) + F.relu(-pred2) + F.relu(pred2 - 100))
        return ranking_loss.mean() + lambda_range * range_penalty.mean()

    train_losses = []
    epoch_train_losses = []
    val_losses = []
    val_maes = []
    global_step = 0

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

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epoch_train_losses, label='Training Loss', color='blue', alpha=0.6)
    ax1.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_ylim(0, 3)
    ax1.legend(loc='upper left'); ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(val_maes, label='Validation MAE', color='green', linestyle='--')
    ax2.set_ylabel("MAE"); ax2.set_ylim(0, 12); ax2.legend(loc='upper right')

    def decay_val(init, decay, epoch):
        return init * (decay ** (epoch - 1))

    mxd0, mxd_decay = config_dict["max_distance"], config_dict.get("maxdistance_decay", 1.0)
    m0, m_decay = config_dict["margin"], config_dict.get("margin_decay", 1.0)
    lr0, lr_decay = config_dict["learning_rate"], config_dict.get("learning_decay", 1.0)
    mxd_5, mxd_10 = decay_val(mxd0, mxd_decay, 5), decay_val(mxd0, mxd_decay, 10)
    m_5, m_10 = decay_val(m0, m_decay, 5), decay_val(m0, m_decay, 10)
    lr_5, lr_10 = decay_val(lr0, lr_decay, 5), decay_val(lr0, lr_decay, 10)

    info_text = (
        f"LR: {config.learning_rate:.1e} | LRD: {config.learning_decay} \u2192 Epoch5: {lr_5:.2f}, Epoch10: {lr_10:.2f}\n"
        f"Ma: {m0:.2f} | MaD: {m_decay} \u2192 Epoch5: {m_5:.2f}, Epoch10: {m_10:.2f}\n"
        f"MD: {mxd0:.2f} | MDD: {mxd_decay} \u2192 Epoch5: {mxd_5:.2f}, Epoch10: {mxd_10:.2f}"
    )
    fig.text(0.05, -0.1, info_text, ha='left', va='top', fontsize=9)

    raw_name = config.name.strip().rstrip(".png")
    plt.title(f"Loss and MAE Over Epochs\n{raw_name}")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    subfolder, base_name = ('', raw_name) if '/' not in raw_name else raw_name.rsplit('/', 1)
    plot_dir = os.path.join(os.getcwd(), "plots", subfolder)
    os.makedirs(plot_dir, exist_ok=True)
    final_name = base_name + ".png"
    counter = 1
    while os.path.exists(os.path.join(plot_dir, final_name)):
        final_name = f"{base_name}_{counter}.png"; counter += 1

    filepath = os.path.join(plot_dir, final_name)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"[INFO] Plot saved to {filepath}")

if __name__ == "__main__":
    main()
