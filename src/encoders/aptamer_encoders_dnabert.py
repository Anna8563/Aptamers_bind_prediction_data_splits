import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def infer_types_from_sequences(seqs):
    """RNA if contains 'U', otherwise DNA."""
    types = []
    for s in seqs:
        if s is None:
            types.append("DNA")
        else:
            s = str(s).upper()
            types.append("RNA" if "U" in s else "DNA")
    return types


@torch.no_grad()
def dnabert2_embed(
    seqs,
    model_name="zhihan1996/DNABERT-2-117M",
    pooling="mean",
    max_len=512,
    batch_size=32,
    device=None
):
    """
    Generate embeddings for DNA/RNA sequences using DNABERT-2.

    Args:
        seqs (list[str]): List of sequences.
        model_name (str): Hugging Face model name.
        pooling (str): "mean" or "max" pooling.
        max_len (int): Maximum tokenized sequence length.
        batch_size (int): Number of sequences per batch.
        device (str): "cuda" or "cpu". If None, auto-detect.

    Returns:
        np.ndarray: Embeddings with an additional column encoding RNA/DNA type.
    """
    # infer types: RNA=1, DNA=0
    types = infer_types_from_sequences(seqs)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,

    )

    # model with safetensors
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=False
    ).to(device)
    model.eval()

    out = []
    for i in range(0, len(seqs), batch_size):
        batch = [str(s).upper().replace("U", "T") for s in seqs[i:i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            hidden_states = model(**inputs)[0]

        if pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif pooling == "max":
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError("pooling must be 'mean' or 'max'")

        out.append(pooled.cpu().numpy())

    E = np.vstack(out)

    # add RNA/DNA flag
    d = np.array([1.0 if t == "RNA" else 0.0 for t in types], dtype=np.float64).reshape(-1, 1)

    return np.concatenate([E, d], axis=1)