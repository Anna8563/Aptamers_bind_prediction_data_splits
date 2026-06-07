import os
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


print("compute_embeddings_ankh")
from src.encoders.protein_encoders import ankh_large_encode_batch


# Paths
dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv"
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)


prot_cfgs = [
    {"name": "Ankh", "func": ankh_large_encode_batch}
]

from src.models.screening import precompute_protein_embeddings
precompute_protein_embeddings(df, prot_cfgs, embeddings_dir=embeddings_dir)