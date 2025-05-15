import torch
import random
import os
import numpy as np
from src.machine_learning import GlyphDataset 

class GlyphAgent:
    def __init__(self, glyph_filename: str, model_filename: str, name: str = None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.glyph_filename = glyph_filename
        self.name = name if name else os.path.basename(glyph_filename)

        # Load model
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file '{model_filename}' does not exist.")
        self.model = torch.load(model_filename, map_location=self.device)
        self.model.eval()

        # Load glyphs
        if not os.path.exists(glyph_filename):
            raise FileNotFoundError(f"Glyph file '{glyph_filename}' not found.")

        self.dataset = GlyphDataset(glyph_filename, split="test", resize=(128, 128))
        self.glyph_map = {round(sample['value'], 2): sample for sample in self.dataset.metadata['samples']['test']}
        self.sample_lookup = {round(sample['value'], 2): idx for idx, sample in enumerate(self.dataset.samples)}

        print(f"[{self.name}] Agent initialized with {len(self.dataset)} glyphs.")

    def get_response(self, task: dict) -> dict:
        """
        Given task like {'x1': float, 'x2': float, 'distance': float},
        compare the values of glyphs nearest to x1 and x2 using the model.
        """
        x1 = task['x1']
        x2 = task['x2']

        # Find nearest glyph values in dataset
        nearest_x1 = min(self.glyph_map.keys(), key=lambda v: abs(v - x1))
        nearest_x2 = min(self.glyph_map.keys(), key=lambda v: abs(v - x2))

        # Get dataset indices for the matching  glyphs
        idx1 = self.sample_lookup[nearest_x1]
        idx2 = self.sample_lookup[nearest_x2]

        image1, _ = self.dataset[idx1]
        image2, _ = self.dataset[idx2]

        # Prepare for model input
        images = torch.stack([image1, image2]).to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            predictions = outputs.squeeze().cpu().numpy()

        value1 = predictions[0]
        value2 = predictions[1]

        # Decision logic
        if abs(value1 - value2) < 1e-3:
            choice = '=='
        elif value1 > value2:
            choice = '>'
        else:
            choice = '<'

        return {
            'choice': choice,
            'time': '-',  # Placeholder: insert timing if needed
            'glyph-name': self.name,
            'x1': x1,
            'x2': x2,
            'v1': float(value1),
            'v2': float(value2),
            'nearest_x1': nearest_x1,
            'nearest_x2': nearest_x2
        }
