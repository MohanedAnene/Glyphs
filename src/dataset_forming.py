import json
import zipfile
import mglyph as mg
import numpy as np
from datetime import datetime
import math
import random
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import torchvision


class GlyphExporter:
    def __init__(self, filepath: str, dataset_name="Unnamed Dataset"):
        """Initialize exporter with just zipping functionality"""
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.glyph_blobs = []  # Stores (split, export_bytesio) pairs
        self.metadata = {
            "name": dataset_name,
            "time-of-creation": datetime.now().isoformat(),
            "samples": {}
        }

    def add(self, export_result, split='train'):
        """Add a glyph export to the ZIP with its dataset split"""
        self.glyph_blobs.append((split, export_result))
        return self  # Enable method chaining

    def finalize(self):
        """Create final ZIP package with all glyphs"""
        with zipfile.ZipFile(self.filepath, 'w') as final_zip:
            for index, (split, blob) in enumerate(self.glyph_blobs):
                variant_name = f"sample_{index}"
                with zipfile.ZipFile(blob) as glyph_zip:
                    for file_name in glyph_zip.namelist():
                        if file_name.endswith('.png'):
                            new_name = f"{variant_name}-{file_name.split('-')[-1]}"
                            final_zip.writestr(new_name, glyph_zip.read(file_name))
                        elif file_name.endswith('.json'):
                            data = json.loads(glyph_zip.read(file_name).decode())

                            # Ensure the split exists in metadata
                            if split not in self.metadata['samples']:
                                self.metadata['samples'][split] = []

                            self.metadata['samples'][split].extend([
                                {
                                    "value": img[1],
                                    "file": f"{variant_name}-{img[0]}"
                                }
                                for img in data['images']
                            ])

            final_zip.writestr('_dataset-info.json', json.dumps(self.metadata, indent=2))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()




def display_dataset(zip_path, img_size=64, grid_size=(5, 5)):
    """
    Display glyphs in a 5x5 grid (always 25 images).
    
    Args:
        zip_path: Path to the zipped dataset
        img_size: Size to resize each image to (default 64)
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all PNG files
        png_files = [f for f in zip_ref.namelist() if f.endswith('.png')]
        
        # Check if there are at least 25 images
        if len(png_files) < 25:
            print(f"Not enough images for a grid (found {len(png_files)}). Need at least 25.")
            return
        
        # Select 25 random images
        selected_files = random.sample(png_files, 25)
        
        # Load and preprocess images
        images = []
        for file in selected_files:
            img_data = zip_ref.read(file)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img = img.resize((img_size, img_size))
            img_tensor = torchvision.transforms.ToTensor()(img)
            images.append(img_tensor)
        
        # Create grid
        grid = torchvision.utils.make_grid(
            torch.stack(images),
            nrow=5,  # Fixed to 5x5 grid
            padding=2,
            normalize=True,
            pad_value=0.9  # Light background
        )
        
        # Display with appropriate figure size
        fig_size = max(6, 5)  # Scale figure with grid size
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Displaying 5×5 grid ({len(png_files)} images available)")
        plt.tight_layout(pad=0)
        plt.show()
