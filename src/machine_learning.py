import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import io
import zipfile
import json
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
import itertools 


# --- Glyph Dataset ---
class GlyphDataset(Dataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None, stride=1,
                 augmentation_rot=0, augmentation_tran_x=0.0, augmentation_tran_y=0.0):
        self.resize = resize
        self.split = split
        self.stride = stride
        self.augmentation_rot = augmentation_rot
        self.augmentation_tran_x = augmentation_tran_x
        self.augmentation_tran_y = augmentation_tran_y

        if transform:
            self.transform = transform
        else:
            base_transforms = []
            if resize is not None:
                base_transforms.append(transforms.Resize(resize))
            else:
                base_transforms.append(transforms.Resize((224, 224)))
            base_transforms.append(transforms.ToTensor())
            self.transform = transforms.Compose(base_transforms)

        # Load metadata and samples first
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('_dataset-info.json') as f:
                self.metadata = json.load(f)

            if split in self.metadata['samples']:
                self.samples = self.metadata['samples'][split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset metadata.")

            if self.stride > 1:
                self.samples = self.samples[self.stride - 1::self.stride]

        # Now preload all images into memory (decoded PIL Images)
        self.preloaded_images = {}
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for sample in self.samples:
                filename = sample['file']
                try:
                    with zip_ref.open(filename) as image_file:
                        # Load, decode and resize image once during init
                        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
                        if self.resize is not None:
                            img = img.resize(self.resize)
                        self.preloaded_images[filename] = img
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")
                    self.preloaded_images[filename] = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['file']
        value = sample['value']

        image = self.preloaded_images.get(filename)
        if image is None:
            raise FileNotFoundError(f"Image data not found for file: {filename}")

        try:
            # Create a copy to avoid modifying the cached image
            image = image.copy()
            
            # Apply augmentations
            width, height = image.size
            if self.augmentation_tran_x != 0.0 or self.augmentation_tran_y != 0.0:
                shift_x = int(width * (self.augmentation_tran_x / 100))
                shift_y = int(height * (self.augmentation_tran_y / 100))
                
                translated_image = Image.new("RGB", (width, height), (255, 255, 255))
                paste_x = max(0, shift_x) if shift_x > 0 else 0
                paste_y = max(0, shift_y) if shift_y > 0 else 0
                crop_left = max(0, -shift_x)
                crop_upper = max(0, -shift_y)
                crop_right = min(width, width - shift_x)
                crop_lower = min(height, height - shift_y)
                
                if crop_left < crop_right and crop_upper < crop_lower:
                    image = image.crop((crop_left, crop_upper, crop_right, crop_lower))
                
                translated_image.paste(image, (paste_x, paste_y))
                image = translated_image

            if self.augmentation_rot > 0:
                angle = random.uniform(-self.augmentation_rot, self.augmentation_rot)
                image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

            # Apply final transforms (just ToTensor since we already resized)
            image = transforms.ToTensor()(image)
            return image, torch.tensor(value, dtype=torch.float32)

        except Exception as e:
            raise RuntimeError(f"Failed to process image {filename}: {str(e)}")


class PairwiseGlyphDataset(GlyphDataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None, stride=1,
                 augmentation_rot=0, augmentation_tran_x=0.0, augmentation_tran_y=0.0):
        super().__init__(zip_path, resize, split, transform, stride,
                        augmentation_rot, augmentation_tran_x, augmentation_tran_y)
        self.pairs = []

    def make_pairs(self, N=1000, max_distance=100.0, seed=None):
        """
        Creates a list of (idx1, idx2) pairs where abs(value1 - value2) <= max_distance.
        Stores it in self.pairs and optionally limits to N total pairs.
        """
        if seed is not None:
            random.seed(seed)
        self.pairs = []
        all_indices = list(range(len(self.samples)))
        values = [self.samples[i]['value'] for i in all_indices]

        # Try random combinations until we reach N (or a safe max)
        attempts = 0
        max_attempts = N * 10  # Prevent infinite loops

        while len(self.pairs) < N and attempts < max_attempts:
            idx1, idx2 = random.sample(all_indices, 2)
            val1, val2 = values[idx1], values[idx2]
            if abs(val1 - val2) <= max_distance:
                self.pairs.append((idx1, idx2))
            attempts += 1

        if len(self.pairs) < N:
            print(f"[INFO] Only {len(self.pairs)} valid pairs found under max_distance={max_distance}")

    def __len__(self):
        if not hasattr(self, 'pairs') or not self.pairs:
            raise ValueError("No pairs generated. Call `.make_pairs()` before using this dataset.")
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Returns (img1, val1, img2, val2) from precomputed pairs.
        """
        idx1, idx2 = self.pairs[index]
        img1, val1 = super().__getitem__(idx1)
        img2, val2 = super().__getitem__(idx2)
        return img1, val1, img2, val2

# --- Helper Functions ---
def create_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True if torch.cuda.is_available() else False, persistent_workers=True if num_workers > 0 else False)

def visualize_loader(loader: DataLoader, max_pairs: int = 8, silent: bool = False):
    try:
        batch = next(iter(loader))

        # Pairwise dataset: (img1s, val1s, img2s, val2s)
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            img1s, val1s, img2s, val2s = batch

            # Limit number of pairs
            total_pairs = min(max_pairs, len(img1s))
            img1s = img1s[:total_pairs]
            val1s = val1s[:total_pairs]
            img2s = img2s[:total_pairs]
            val2s = val2s[:total_pairs]

            # One row per pair, two images per row
            fig, axes = plt.subplots(total_pairs, 2, figsize=(4, total_pairs * 1.5))
            if total_pairs == 1:
                axes = [axes]  # Handle single pair as list

            for i in range(total_pairs):
                for j in range(2):
                    ax = axes[i][j]
                    img = (img1s[i] if j == 0 else img2s[i]).permute(1, 2, 0).numpy()
                    val = val1s[i] if j == 0 else val2s[i]
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"Value {j+1}: {val:.2f}", fontsize=9)

                delta = abs(val1s[i].item() - val2s[i].item())
                axes[i][0].text(4, 10, f"Δ = {delta:.2f}", fontsize=8, color='black',
                                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))

            plt.tight_layout()
            plt.show()

        else:
            # Normal dataset: (images, values)
            images, values = batch

            max_images = min(16, len(images))
            images = images[:max_images]
            values = values[:max_images]

            nrow = 4
            ncol = math.ceil(len(images) / nrow)

            fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.5, ncol * 1.5))
            axes = axes.flatten() if max_images > 1 else [axes]

            for idx, ax in enumerate(axes):
                if idx < len(images):
                    img = images[idx].permute(1, 2, 0).numpy()
                    val = values[idx]
                    ax.imshow(img)
                    ax.text(4, 12, f"Value: {val:.2f}", fontsize=9, color='white',
                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                ax.axis('off')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        if not silent:
            print(f"Error visualizing batch: {e}")






def plot_training_loss(train_losses, val_losses=None, title="Loss Over Time",
                       xlabel="Steps" , ylabel="Loss", figsize=(8, 4)):
    """
    Plots training loss (and optional validation loss) over time.

    Args:
        train_losses (list): A list of training loss values (can be per step or epoch).
        val_losses (list, optional): A list of validation loss values (must be per epoch).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Size of the plot.
    """
    if not train_losses:
        print("Warning: Training loss list is empty.")
        return

    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.6)

    if val_losses is not None:
        # Align validation points evenly spaced across the x-axis
        val_x = np.linspace(0, len(train_losses), len(val_losses))
        plt.plot(val_x, val_losses, label='Validation Loss', color='orange', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          figsize=(8, 6)):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 
    num_classes = len(classes)
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                # Set text color based on background intensity
                color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return ax


def show_incorrect_predictions(model, loader, bin_centers, max_display=10, device=None):

    import matplotlib.pyplot as plt
    import math
    import torch.nn.functional as F

    incorrect_samples = []
    model.eval()

    with torch.no_grad():
        for images, values in loader:
            images = images.to(device)
            values = torch.tensor(values, dtype=torch.float32, device=device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.sum(probabilities * bin_centers, dim=1)

            errors = torch.abs(predictions - values)

            for i in range(images.size(0)):
                incorrect_samples.append({
                    'image': images[i].cpu(),
                    'true_value': values[i].item(),
                    'pred_value': predictions[i].item(),
                    'abs_error': errors[i].item()
                })

                if len(incorrect_samples) >= max_display:
                    break
            if len(incorrect_samples) >= max_display:
                break

    if not incorrect_samples:
        print("No incorrect predictions found.")
        return

    # --- Visualization ---
    nrow = 5
    ncol = math.ceil(len(incorrect_samples) / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten() if max_display > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(incorrect_samples):
            sample = incorrect_samples[idx]
            img = sample['image'].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.text(
                4, 12,
                f"True: {sample['true_value']:.2f}\nPred: {sample['pred_value']:.2f}\nError: {sample['abs_error']:.2f}",
                fontsize=9, color='white',
                bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.3')
            )
        ax.axis('off')

    plt.tight_layout()
    plt.show()


