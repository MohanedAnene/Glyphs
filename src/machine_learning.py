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
    def __init__(self, zip_path, resize=None, split='train', transform=None, stride=1):
        self.resize = resize
        self.split = split
        self.stride = stride


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

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('_dataset-info.json') as f:
                self.metadata = json.load(f)

            if split in self.metadata['samples']:
                self.samples = self.metadata['samples'][split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset metadata. Available splits: {list(self.metadata['samples'].keys())}")
            
            if self.stride > 1:
                self.samples = self.samples[self.stride - 1 :: self.stride]



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
        value = sample['value']

        image_bytes = self.image_data.get(filename)
        if image_bytes is None:
            raise FileNotFoundError(f"Image data not found for file: {filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image), value
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
        values = []

        for i in indices:
            image, value = self[i]  # now you only return (image, value)
            images.append(image)
            values.append(value)

        nrow = min(5, num)
        ncol = math.ceil(num / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1, ncol * 1))
        axes = axes.flatten() if num > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)

                ax.text(
                    4, 12,
                    f"Value: {values[idx]:.2f}",  # Always just display the continuous value
                    fontsize=9, color='white',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
                )

            ax.axis('off')

        plt.tight_layout()
        plt.show()


# --- Helper Functions ---
def create_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True if torch.cuda.is_available() else False)

def visualize_loader(loader: DataLoader, max_images: int = 16, nrow: int = 4, silent: bool=False):
    try:
        images, values = next(iter(loader)) 

        if len(images) > max_images:
            images = images[:max_images]
            values = values[:max_images]
            if not silent:
                print(f"Displaying first {max_images} images from batch of {len(images)}")

        ncol = math.ceil(len(images) / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1, ncol * 1))
        axes = axes.flatten() if max_images > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.text(
                    4, 12, 
                    f"Value: {values[idx]:.2f}", 
                    fontsize=9, color='white',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
                )

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

def to_class_label(value, num_classes):
    value = max(0.0, min(100.0, value))
    bin_width = 100.0 / num_classes
    class_idx = int(value / bin_width)
    return min(class_idx, num_classes - 1)

def show_incorrect_predictions(model, loader, num_classes=10, max_display=10, device=None):
    incorrect_samples = []

    model.eval()
    with torch.no_grad():
        for images, values in loader:
            images = images.to(device)

            outputs = model(images)

            # Detect if it's regression or classification
            if outputs.shape[1] == 1:  # Regression case (1 output per image)
                preds = outputs.squeeze(1)  # shape [batch]
                preds = preds.cpu()
                values = torch.tensor(values)  # real values

                # Compute absolute error
                errors = torch.abs(preds - values)
                
                for i in range(images.size(0)):
                    incorrect_samples.append({
                        'image': images[i].cpu(),
                        'true_value': values[i].item(),
                        'pred_value': preds[i].item(),
                        'abs_error': errors[i].item()
                    })
                    
            else:  # Classification case (multi-class)
                _, preds = torch.max(outputs, 1)
                labels = torch.tensor(
                    [to_class_label(v, num_classes) for v in values], dtype=torch.long, device=device
                )

                for i in range(images.size(0)):
                    if preds[i] != labels[i]:
                        val = values[i].item()
                        pred_class = preds[i].item()

                        bin_width = 100.0 / num_classes
                        left = pred_class * bin_width
                        right = (pred_class + 1) * bin_width

                        diff_to_left = abs(val - left)
                        diff_to_right = abs(val - right)
                        closest_diff = min(diff_to_left, diff_to_right)

                        incorrect_samples.append({
                            'image': images[i].cpu(),
                            'true_value': val,
                            'pred_class': pred_class,
                            'closest_diff': closest_diff
                        })

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

            if 'pred_value' in sample:  # Regression display
                ax.text(
                    4, 12,
                    f"True: {sample['true_value']:.2f}\nPred: {sample['pred_value']:.2f}\nAbs Error: {sample['abs_error']:.2f}",
                    fontsize=9, color='white',
                    bbox=dict(facecolor='blue', alpha=0.7, boxstyle='round,pad=0.3')
                )
            else:  # Classification display
                ax.text(
                    4, 12,
                    f"Val: {sample['true_value']:.2f}\nPred class: {sample['pred_class']}\nClosest Diff: {sample['closest_diff']:.2f}",
                    fontsize=9, color='white',
                    bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.3')
                )

        ax.axis('off')

    plt.tight_layout()
    plt.show()

