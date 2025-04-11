import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import io
import zipfile
import json
import random
import matplotlib.pyplot as plt
import math


# --- Glyph Dataset ---
class GlyphDataset(Dataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None, num_classes=10):
        self.resize = resize
        self.split = split
        self.num_classes = num_classes

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

        binned_label = get_label_class(label, self.num_classes)

        image_bytes = self.image_data.get(filename)
        if image_bytes is None:
            raise FileNotFoundError(f"Image data not found for file: {filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image), binned_label, label
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

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1, ncol * 1))
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


# --- Helper Functions ---
def create_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True if torch.cuda.is_available() else False)

def visualize_loader(loader: DataLoader, max_images: int = 16, nrow: int = 4, silent: bool = False):
    try:
        images, labels, original_values = next(iter(loader))

        if len(images) > max_images:
            images = images[:max_images]
            labels = labels[:max_images]
            if not silent:
                print(f"Displaying first {max_images} images from batch of {len(images)}")

        ncol = math.ceil(len(images) / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1, ncol * 1))
        axes = axes.flatten() if max_images > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.text(4, 12, f"Val: {original_values[idx]:.2f}\nClass: {labels[idx]}", fontsize=9, color='white',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

            ax.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        if not silent:
            print(f"Error visualizing batch: {e}")

def get_label_class(value, num_classes=10):
    value = max(0.0, min(100.0, value))
    bin_width = 100.0 / num_classes
    class_idx = int(value / bin_width)
    return min(class_idx, num_classes - 1)

def plot_training_loss(losses, title="Training Loss Over Time", xlabel="Steps", ylabel="Loss", figsize=(8, 4)):
    """
    Plots the training loss recorded over training steps or epochs.

    Args:
        losses (list): A list of loss values.
        title (str, optional): The title for the plot. Defaults to "Training Loss Over Time".
        xlabel (str, optional): Label for the x-axis. Defaults to "Steps".
        ylabel (str, optional): Label for the y-axis. Defaults to "Loss".
        figsize (tuple, optional): Figure size (width, height). Defaults to (8, 4).
    """
    if not losses:
        print("Warning: Loss list is empty. Cannot plot.")
        return

    plt.figure(figsize=figsize)
    plt.plot(losses, label='Training Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()