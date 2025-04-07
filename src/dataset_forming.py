import json
import zipfile
import mglyph as mg
import numpy as np
from datetime import datetime
import math
import random

class GlyphExporter:
    def __init__(self, filepath:str, dataset_name="Unnamed Dataset"):
        """Initialize exporter with just zipping functionality"""
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.glyph_blobs = []  # Stores (variant_name, export_bytesio) pairs
        self.metadata = {
            "name": dataset_name,
            "time-of-creation": datetime.now().isoformat(),
            "samples": []
        }

    def add(self, export_result):

        self.glyph_blobs.append((export_result))
        return self  # Enable method chaining

    def finalize(self):
        """Create final ZIP package with all glyphs"""
        with zipfile.ZipFile(self.filepath, 'w') as final_zip:
            for index, blob in enumerate(self.glyph_blobs):
                variant_name = f"sample_{index}"
                with zipfile.ZipFile(blob) as glyph_zip:
                    for file_name in glyph_zip.namelist():
                        if file_name.endswith('.png'):
                            new_name = f"{variant_name}-{file_name.split('-')[-1]}"
                            final_zip.writestr(new_name, glyph_zip.read(file_name))
                        elif file_name.endswith('.json'):
                            data = json.loads(glyph_zip.read(file_name).decode())
                            self.metadata['samples'].extend({
                                "value": img[1],
                                "file": f"{variant_name}-{img[0]}"
                            } for img in data['images'])


            final_zip.writestr('_dataset-info.json', json.dumps(self.metadata, indent=2))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()