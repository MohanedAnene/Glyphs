# src/train_model.py

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os
from pathlib import Path

from src.Model_BR import GlyphClassifier
import src.machine_learning as ML

def create_model(dataset: str, epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.0005,
                 image_resolution=(128, 128), num_bins: int = 5, rotation=0, translation=0):
    
    base_name = Path(dataset).stem  # "random-star" from "data/random-star.zip"
    validation_file = f"{Path(dataset).parent}/{base_name}-validation.zip"
    model_output_path = f"data/{base_name}.pt"

    print(f"📦 Training on: {dataset}")
    print(f"🧪 Validation on: {validation_file}")
    print(f"💾 Output model will be saved to: {model_output_path}\n")

    # === Dataset loading ===
    train_dataset = ML.GlyphDataset(dataset, resize=image_resolution, split="train",
                                    augmentation_rot=rotation, augmentation_tran=translation)
    validation_dataset = ML.GlyphDataset(dataset, resize=image_resolution, split="test",
                                         augmentation_rot=rotation, augmentation_tran=translation)

    train_loader = ML.create_loader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = ML.create_loader(validation_dataset, batch_size=batch_size, shuffle=False)

    # === Model setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlyphClassifier(resolution=image_resolution, NUM_bins=num_bins).to(device)

    bin_centers = torch.linspace(0, 100, num_bins + 1, device=device)[:-1] + 50 / num_bins
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # === Training ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, values in train_loader:
            images, values = images.to(device), values.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.sum(probabilities * bin_centers, dim=1)
            loss = criterion(predictions, values)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, values in validation_loader:
                images, values = images.to(device), values.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.sum(probs * bin_centers, dim=1)
                val_loss += criterion(preds, values).item()

        scheduler.step()
        avg_val_loss = val_loss / len(validation_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Val Loss: {avg_val_loss:.4f}")

    # === Save the model ===
    torch.save(model.state_dict(), model_output_path)
    print(f"\n✅ Model saved to: {model_output_path}")
