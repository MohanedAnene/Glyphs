import os
import io
import torch
import zipfile
import json
from PIL import Image
from torchvision import transforms
from src.Model_BR import GlyphClassifier


class GlyphAgent:
    def __init__(self, glyph_data, model_filename: str, name: str = None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.name = name or "UnnamedGlyphSet"

        # === Load model ===
        self.model = GlyphClassifier(NUM_bins=5, resolution=(128, 128))
        state_dict = torch.load(model_filename, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # === Bin centers (binned regression with 5 bins) ===
        self.bin_centers = torch.linspace(0, 100, 6, device=self.device)[:-1] + 50 / 5

        # === Load ZIP: from path or BytesIO ===
        if isinstance(glyph_data, (str, os.PathLike)):
            with open(glyph_data, 'rb') as f:
                self.zip_buffer = io.BytesIO(f.read())
        elif isinstance(glyph_data, io.BytesIO):
            self.zip_buffer = glyph_data
        else:
            raise ValueError("glyph_data must be a file path or a BytesIO object")

        # === Parse metadata.json and load image bytes ===
        with zipfile.ZipFile(self.zip_buffer, 'r') as zf:
            if 'metadata.json' not in zf.namelist():
                raise RuntimeError("ZIP archive does not contain 'metadata.json'")
            with zf.open('metadata.json') as f:
                metadata = json.load(f)

            if "images" not in metadata:
                raise RuntimeError("'metadata.json' does not contain 'images' key")

            self.samples = [
                {"file": fname, "value": float(val)}
                for fname, val in metadata["images"]
            ]

            # Load image bytes
            self.image_data = {}
            for sample in self.samples:
                fname = sample['file']
                try:
                    with zf.open(fname) as img_file:
                        self.image_data[fname] = img_file.read()
                except Exception as e:
                    print(f"[WARNING] Could not read image {fname}: {e}")
                    self.image_data[fname] = None

        # === Preprocess all images ===
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.processed_samples = []
        for sample in self.samples:
            fname = sample["file"]
            value = sample["value"]
            raw_bytes = self.image_data.get(fname)
            if raw_bytes is None:
                continue
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            tensor = self.transform(image)
            self.processed_samples.append((tensor, value))

        # === Mapping from value ↔ file and file ↔ index ===
        self.value_to_filename = {
            round(s["value"], 2): s["file"] for s in self.samples
        }
        self.filename_to_index = {
            s["file"]: idx for idx, s in enumerate(self.samples)
        }

    def get_response(self, task: dict, verbose=False) -> dict:
        x1 = task['x1']
        x2 = task['x2']

        nearest_x1 = min(self.value_to_filename.keys(), key=lambda v: abs(v - x1))
        nearest_x2 = min(self.value_to_filename.keys(), key=lambda v: abs(v - x2))

        file1 = self.value_to_filename[nearest_x1]
        file2 = self.value_to_filename[nearest_x2]

        idx1 = self.filename_to_index[file1]
        idx2 = self.filename_to_index[file2]

        image1, _ = self.processed_samples[idx1]
        image2, _ = self.processed_samples[idx2]

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
