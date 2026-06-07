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
os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["USE_FLASH_ATTENTION"] = "0"



# --- Aptamer encoders ---
from src.encoders.aptamer_encoders import (
    onehot_with_type_bit,
    kmer_freq_with_type_bit,
    gena_embed
)

# Paths
dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv"
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"
gena_model_path = root / "src" / "models" / "gena_lm"
# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded:", df.shape)


apt_cfgs = [
    {"name": "OneHot", "func": onehot_with_type_bit},
    {"name": "Kmer3", "func": kmer_freq_with_type_bit, "kwargs": {"k": 3}},
    {"name": "Kmer4", "func": kmer_freq_with_type_bit, "kwargs": {"k": 4}},
    {"name": "GENA", "func": gena_embed, "kwargs": {"model_name": gena_model_path}}
]

from src.models.screening import precompute_aptamer_embeddings
precompute_aptamer_embeddings(df, apt_cfgs, embeddings_dir=embeddings_dir, seq_col = "Aptamer_sequence")