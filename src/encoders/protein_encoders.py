import re
import numpy as np
import torch
from tqdm import tqdm

import ankh
from tqdm import tqdm
import pandas as pd


def ankh_large_encode_batch(sequences, device=None, batch_size=4, return_numpy=True):
    # device
    print("Ankh function")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(torch.cuda.is_available())
    print(device)
    model, tokenizer = ankh.load_large_model()
    model = model.to(device)
    model.eval()
    print("Model downloaded successfully!")
    # preprocess: uppercase + replace invalid chars + space-separated tokens
    seqs = [" ".join(list(re.sub(r"[^ACDEFGHIKLMNPQRSTVWYXBZJUO]", "X", seq.upper()))) for seq in sequences]

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="Encoding Ankh_large batches"):
            batch = seqs[i : i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding="longest", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            attn = encoded["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            last_hidden = out.last_hidden_state  # shape: (batch_size, seq_len, embedding_dim)

            # агрегация: среднее по длине
            emb = last_hidden.mean(dim=1)  # shape (batch_size, embedding_dim)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def process_dataset(df, seq_col="seq", batch_func=ankh_large_encode_batch, **kwargs):
    embs = batch_func(df[seq_col].tolist(), **kwargs)
    df_emb = pd.DataFrame(embs, index=df.index)
    return pd.concat([df, df_emb], axis=1)