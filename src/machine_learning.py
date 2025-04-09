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
        """Get item implementation that fails fast on errors"""
        sample = self.samples[idx]
        filename = sample['file']
        label = sample['value']

        # Get image bytes - raise error if not found
        image_bytes = self.image_data.get(filename)
        if image_bytes is None:
            raise FileNotFoundError(f"Image data not found for file: {filename}")

        try:
            # Decode from bytes - let any exceptions propagate up
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return self.transform(image), label
        except Exception as e:
            # Raise a new exception with context
            raise RuntimeError(f"Failed to decode image {filename}") from e


    def show(self, num=10):
        """Display samples from the dataset in a grid.
        
        Args:
            num: Number of samples to display
            
        Raises:
            ValueError: If no samples are available or if requested number is invalid
            RuntimeError: If images fail to load
        """
        # Validate input
        if num <= 0:
            raise ValueError(f"Number of samples to display must be positive (got {num})")
        
        if len(self) == 0:
            raise ValueError("Dataset contains no samples")
        
        # Check if requested number exceeds available samples
        if num > len(self):
            raise ValueError(
                f"Requested {num} samples but dataset only contains {len(self)}. "
                "Please request fewer samples or add more data to the dataset."
            )
        
        # Get samples - let any loading errors propagate up
        indices = random.sample(range(len(self)), num)
        images = []
        labels = []
        
        for i in indices:
            image, label = self[i]  # Will raise exceptions if loading fails
            images.append(image)
            labels.append(label)
        
        # Create visualization
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




def create_loader(dataset: GlyphDataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
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