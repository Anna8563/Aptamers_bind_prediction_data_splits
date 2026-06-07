import os
from src.models.screening import precompute_aptamer_embeddings

import pandas as pd
# Add project root to sys.path so "src" is importable
import sys
from pathlib import Path

root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))

os.environ["HF_HOME"] = str(root / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hf_home" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_home" / "transformers")
os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"


# --- Aptamer encoders ---
from src.encoders.aptamer_encoders import dnabert2_embed


# Paths
dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv"
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)

apt_cfgs = [
    {"name": "DNABERT2", "func": dnabert2_embed},
]

precompute_aptamer_embeddings(df, apt_cfgs, embeddings_dir=embeddings_dir, seq_col = "Aptamer_sequence")