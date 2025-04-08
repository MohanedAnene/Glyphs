import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import zipfile
import io
import json
import random
import matplotlib.pyplot as plt
import math

class GlyphDataset(Dataset):
    def __init__(self, zip_path, resize=None, split='train', transform=None):
        self.zip_path = zip_path
        self.resize = resize
        self.split = split

        # Define default transform
        if transform:
            self.transform = transform
        else:
            base_transforms = []
            if resize is not None:
                base_transforms.append(transforms.Resize(resize))
            base_transforms.append(transforms.ToTensor())
            self.transform = transforms.Compose(base_transforms)

        # Load metadata and extract sample info
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('_dataset-info.json') as f:
                self.metadata = json.load(f)

        self.samples = self.metadata['samples'].get(split, [])
        self.zip_ref = zipfile.ZipFile(zip_path, 'r')  # keep open for __getitem__

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample['file']
        label = sample['value']

        try:
            with self.zip_ref.open(image_name) as image_file:
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
        except Exception as e:
            print(f"[ERROR] Failed to load {image_name}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (64, 64) if self.resize is None else Image.new('RGB', self.resize))
            return self.transform(blank_image), -1  # -1 as invalid label

    def show(self, num=10):
        # Ensure we don't request more samples than available
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

        # Calculate grid dimensions
        nrow = min(5, num)  # Max 5 columns
        ncol = math.ceil(num / nrow)
        
        grid = make_grid(images, nrow=nrow, padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Showing {num} samples from {self.split} set (Total: {len(self)})")
        plt.show()

        # Print the corresponding labels
        print("Sample labels:", labels)