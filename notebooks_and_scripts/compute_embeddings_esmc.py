import pandas as pd
# Add project root to sys.path so "src" is importable
import sys
from pathlib import Path


root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))


from src.encoders.protein_encoders_esmc import esmc_encode_batch

# Paths
dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv"
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)


prot_cfgs = [
    {"name": "ESMC", "func": esmc_encode_batch}
]

from src.models.screening import precompute_protein_embeddings
precompute_protein_embeddings(df, prot_cfgs, embeddings_dir=embeddings_dir)