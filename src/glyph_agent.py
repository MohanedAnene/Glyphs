import os
import torch
import zipfile
import json
from src.machine_learning import GlyphDataset
from src.Model_BR import GlyphClassifier


class GlyphAgent:
    def __init__(self, glyph_filename: str, model_filename: str, name: str = None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.glyph_filename = glyph_filename
        self.name = name if name else os.path.basename(glyph_filename)

        # === Load the model architecture ===
        self.model = GlyphClassifier(NUM_bins=5, resolution=(128, 128))
        state_dict = torch.load(model_filename, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # === Set up bin centers (must match training config) ===
        self.bin_centers = torch.linspace(0, 100, 5 + 1, device=self.device)[:-1] + 50 / 5

        # === Load glyph dataset ===
        if not os.path.exists(glyph_filename):
            raise FileNotFoundError(f"Glyph file '{glyph_filename}' not found.")

        self.dataset = GlyphDataset(glyph_filename, split='test', resize=(128, 128))
        self.samples = self.dataset.samples

        # === Read the JSON metadata from zip ===
        with zipfile.ZipFile(glyph_filename, 'r') as zf:
            with zf.open('_dataset-info.json') as f:
                metadata = json.load(f)

        # Combine both train and test samples into a flat list
        train_samples = metadata["samples"].get("train", [])
        test_samples = metadata["samples"].get("test", [])
        all_samples = train_samples + test_samples

        # Map rounded value to filename
        self.value_to_filename = {
            round(sample["value"], 2): sample["file"]
            for sample in all_samples
        }

        # Map filename to dataset index (from GlyphDataset)
        self.filename_to_index = {
            sample['file']: idx for idx, sample in enumerate(self.samples)
        }

        print(f"[{self.name}] Agent initialized with {len(self.samples)} glyphs.")

    def get_response(self, task: dict, verbose=False) -> dict:
        x1 = task['x1']
        x2 = task['x2']

        nearest_x1 = min(self.value_to_filename.keys(), key=lambda v: abs(v - x1))
        nearest_x2 = min(self.value_to_filename.keys(), key=lambda v: abs(v - x2))

        file1 = self.value_to_filename[nearest_x1]
        file2 = self.value_to_filename[nearest_x2]

        idx1 = self.filename_to_index[file1]
        idx2 = self.filename_to_index[file2]

        image1, _ = self.dataset[idx1]
        image2, _ = self.dataset[idx2]

        images = torch.stack([image1, image2]).to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.sum(probs * self.bin_centers, dim=1)
            predictions = preds.cpu().numpy().flatten()
            v1, v2 = float(predictions[0]), float(predictions[1])

        if abs(v1 - v2) < 1e-3:
            choice = '=='
        elif v1 > v2:
            choice = '>'
        else:
            choice = '<'

        if verbose:
            print(f"\n🧠 GlyphAgent: {self.name}")
            print(f"   x1 = {x1:.2f} (closest: {nearest_x1:.2f}) → predicted value: {v1:.3f}")
            print(f"   x2 = {x2:.2f} (closest: {nearest_x2:.2f}) → predicted value: {v2:.3f}")
            print(f"=> Model decides: x1 {choice} x2\n")

        return {
            'choice': choice,
            'glyph-name': self.name,
            'x1': x1,
            'x2': x2,
            'v1': v1,
            'v2': v2,
            'nearest_x1': nearest_x1,
            'nearest_x2': nearest_x2
        }

