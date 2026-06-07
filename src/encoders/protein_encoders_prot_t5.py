import re
import numpy as np
import torch

import os
import sys

from pathlib import Path
root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))

os.environ["HF_HOME"] = str(root / "hf_home")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hf_home" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(root / "hf_home" / "transformers")

from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm



def prot_t5_encode_batch(sequences, model_name="Rostlab/prot_t5_xl_half_uniref50-enc", device=None, batch_size=16):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Загрузка токенизатора и модели
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)

    # Если устройство CPU, перевести модель в float32 (half precision не поддерживается)
    if device.type == "cpu":
        model = model.float()

    # Предобработка последовательностей: замена редких аминокислот и разделение пробелами
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

    embeddings = []

    # model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding batches"):
            batch_seqs = sequences[i:i + batch_size]

            # Токенизация с добавлением паддинга до длины самой длинной последовательности в батче
            encoded = tokenizer(batch_seqs, add_special_tokens=True, return_tensors="pt", padding="longest")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # shape (batch_size, seq_len, hidden_dim)

            # Усредняем по длине последовательности для каждого белка
            emb_batch = last_hidden.mean(dim=1)  # shape (batch_size, hidden_dim)
            print(emb_batch.shape)

            embeddings.append(emb_batch.cpu().numpy())

        embeddings = np.vstack(embeddings)

    return embeddings

