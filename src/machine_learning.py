import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from PIL import Image
import zipfile
import io
import json
import random
import matplotlib.pyplot as plt
import math

class GlyphDataset(Dataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None):
        self.resize = resize
        self.split = split

        # Define transform pipeline
        if transform:
            self.transform = transform
        else:
            base_transforms = []
            if resize is not None:
                base_transforms.append(transforms.Resize(resize))
            base_transforms.append(transforms.ToTensor())
            self.transform = transforms.Compose(base_transforms)

        # Load metadata and extract samples
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('_dataset-info.json') as f:
                self.metadata = json.load(f)

            self.samples = self.metadata['samples'].get(split, [])

            # Preload and store raw image bytes (compressed)
            self.image_data = {}
            for sample in self.samples:
                filename = sample['file']
                try:
                    with zip_ref.open(filename) as image_file:
                        self.image_data[filename] = image_file.read()
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")
                    self.image_data[filename] = None  # Placeholder for failed loads

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['file']
        label = sample['value']

        image_bytes = self.image_data.get(filename)

        if image_bytes is None:
            # Create a blank image as fallback
            blank_image = Image.new('RGB', (64, 64) if self.resize is None else self.resize)
            return self.transform(blank_image), -1

        try:
            # Decode from bytes when needed
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image), label
        except Exception as e:
            print(f"[ERROR] Decoding failed for {filename}: {e}")
            blank_image = Image.new('RGB', (64, 64) if self.resize is None else self.resize)
            return self.transform(blank_image), -1


    def show(self, num=10):
        num = min(num, len(self))
        if num <= 0:
            print("No samples available to display.")
            return

        indices = random.sample(range(len(self)), num)
        images = []
        labels = []

        for i in indices:
            item = self[i]
            if item is not None:
                images.append(item[0])
                labels.append(item[1])

        if not images:
            print("No images to display.")
            return

        nrow = min(5, num)
        ncol = math.ceil(num / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2.5, ncol * 2.5))
        axes = axes.flatten() if num > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.text(4, 12, str(labels[idx]), fontsize=10, color='white', 
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            ax.axis('off')

        plt.tight_layout()
        plt.show()




def create_loader(dataset: GlyphDataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, silent: bool = False) -> DataLoader:
    """
    Creates a DataLoader from a GlyphDataset.
    
    Args:
        dataset: GlyphDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        silent: If True, suppress print output
        
    Returns:
        DataLoader configured with the specified parameters
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if not silent:
        print(f"Created DataLoader with:")
        print(f"- Dataset samples: {len(dataset)}")
        print(f"- Batch size: {batch_size}")
        print(f"- Number of batches: {len(loader)}")
        print(f"- Shuffling: {'Enabled' if shuffle else 'Disabled'}")

    return loader


def visualize_loader(loader: DataLoader, max_images: int = 16, nrow: int = 4, silent: bool = False):
    """
    Visualizes a batch from a DataLoader with optional silence mode.
    
    Args:
        loader: DataLoader to visualize
        max_images: Maximum number of images to display
        nrow: Number of images per row in grid
        silent: If True, suppresses all print statements
    """
    try:
        images, labels = next(iter(loader))

        if len(images) > max_images:
            images = images[:max_images]
            labels = labels[:max_images]
            if not silent:
                print(f"Displaying first {max_images} images from batch of {len(images)}")

        ncol = math.ceil(len(images) / nrow)

        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2.5, ncol * 2.5))
        axes = axes.flatten() if max_images > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.text(4, 12, str(labels[idx].item()), fontsize=10, color='white',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        if not silent:
            print(f"Error visualizing batch: {e}")