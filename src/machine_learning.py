import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import zipfile
import io
import json
import random
import matplotlib.pyplot as plt
import math


# --- Glyph Dataset ---
class GlyphDataset(Dataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None):
        self.resize = resize
        self.split = split

        if transform:
            self.transform = transform
        else:
            base_transforms = []
            if resize is not None:
                base_transforms.append(transforms.Resize(resize))
            base_transforms.append(transforms.ToTensor())
            self.transform = transforms.Compose(base_transforms)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('_dataset-info.json') as f:
                self.metadata = json.load(f)

            self.samples = []
            for split_key in self.metadata['samples']:
                self.samples.extend(self.metadata['samples'][split_key])

            self.image_data = {}
            for sample in self.samples:
                filename = sample['file']
                try:
                    with zip_ref.open(filename) as image_file:
                        self.image_data[filename] = image_file.read()
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")
                    self.image_data[filename] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['file']
        label = sample['value']
        binned_label = min(max(0, int(label / 10)), 9) ##this will be changed to something dynamic until we move to regression

        image_bytes = self.image_data.get(filename)
        if image_bytes is None:
            raise FileNotFoundError(f"Image data not found for file: {filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image), binned_label, label  # also return original value for display
        except Exception as e:
            raise RuntimeError(f"Failed to decode image {filename}") from e

    def show(self, num=10):
        if num <= 0:
            raise ValueError(f"Number of samples to display must be positive (got {num})")

        if len(self) == 0:
            raise ValueError("Dataset contains no samples")

        if num > len(self):
            raise ValueError(
                f"Requested {num} samples but dataset only contains {len(self)}. "
                "Please request fewer samples or add more data to the dataset."
            )

        indices = random.sample(range(len(self)), num)
        images = []
        labels = []
        original_values = []

        for i in indices:
            image, label, original = self[i]
            images.append(image)
            labels.append(label)
            original_values.append(original)

        nrow = min(5, num)
        ncol = math.ceil(num / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2.5, ncol * 2.5))
        axes = axes.flatten() if num > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.text(4, 12, f"Val: {original_values[idx]:.2f}\nClass: {labels[idx]}", fontsize=9, color='white', 
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# --- Glyph Classifier Model ---
class GlyphClassifier(nn.Module):
    def __init__(self):
        super(GlyphClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- Module-Level Globals ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlyphClassifier().to(device)
losses = []


# --- Helper Functions ---
def create_loader(dataset: GlyphDataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True if torch.cuda.is_available() else False)


def train(zip_path, batch_size=16, epochs=5):
    global model, losses
    train_dataset = GlyphDataset(zip_path, resize=(64, 64), split='train')
    train_loader = create_loader(train_dataset, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Loss: {losses[-1]:.4f}")


def disptrain():
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Training Loss')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test(zip_path):
    global model, test_predictions
    test_dataset = GlyphDataset(zip_path, resize=(64, 64), split='test')
    test_loader = create_loader(test_dataset, batch_size=10, shuffle=False)
    model.eval()
    correct, total = 0, 0
    test_predictions = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            test_predictions.append((images.cpu(), labels.cpu(), predicted.cpu()))

    print(f"Test Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")


def disptest():
    global test_predictions
    if not test_predictions:
        print("No test predictions available. Please run test() first.")
        return

    images, labels, predicted = test_predictions[0]
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"T:{labels[i].item()} / P:{predicted[i].item()}", fontsize=10)
    plt.suptitle("Test Set Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()
