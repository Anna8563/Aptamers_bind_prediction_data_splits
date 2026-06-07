import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm


import sys
print(sys.executable)

def esmc_encode_batch(sequences, model_name="esmc_600m", device=None, batch_size=16):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Загружаем модель ESM C
    client = ESMC.from_pretrained(model_name).to(device)
    client.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding batches"):
            batch_seqs = sequences[i:i + batch_size]

            for seq in batch_seqs:
                prot = ESMProtein(sequence=seq)
                protein_tensor = client.encode(prot)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                emb = logits_output.embeddings.squeeze(0)  # (seq_len, embedding_dim)
                emb_mean = emb.mean(dim=0).cpu().numpy()
                embeddings.append(emb_mean)

    embeddings = np.vstack(embeddings)
    return embeddings