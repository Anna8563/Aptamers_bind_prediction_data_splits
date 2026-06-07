import gc
import pandas as pd
import numpy as np
import ast
import json
import warnings
from tqdm import tqdm
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import sys
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)



root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))


clf_results_path = root / "notebooks" / "notebooks" /" checkpoints" /" multimodel_results_screening_complete.csv"
clf_results = pd.read_csv(clf_results_path)

embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"
splits_dir = root / "dataset" / "splits"
checkpoint_dir = root / "notebooks" / "notebooks" / "checkpoints"

DATASET_PATH = root / "dataset" / "Aptamer_protein_dataset.csv"
base_df = pd.read_csv(DATASET_PATH)


def _compute_dataset_hash(df):
    """Compute hash of dataset for cache validation."""
    # Use shape, first few rows, and column names for hash
    data_str = f"{df.shape}_{df.iloc[:5].to_json()}_{list(df.columns)}"
    return hashlib.md5(data_str.encode()).hexdigest()


def _get_embedding_cache_dir(embeddings_dir="notebooks/data/embeddings"):
    """Get or create embeddings cache directory."""
    cache_dir = Path(embeddings_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def load_splits_with_threshold(split_mode, base_dir="..\dataset\splits"):
    """
    Load precomputed splits from JSON file.

    Parameters
    ----------
    split_mode : str
        One of {"stratified", "disjoint_aptamer", "disjoint_protein"}.
    base_dir : str
        Path to the directory containing split JSONs.

    Returns
    -------
    list of (train_idx, val_idx)
        Indices for each fold.
    """
    matches = list(base_dir.glob(f"*_{split_mode}.json"))    

    if not matches:
        raise FileNotFoundError(f"No file matching '*_{split_mode}.json' in {base_dir}")
    
    if len(matches) > 1:
        raise ValueError(f"Multiple files found for {split_mode}: {matches}")
    
    path = matches[0]

    with open(path, "r") as f:
        data = json.load(f)
    return [(np.array(d["train_idx"]), np.array(d["val_idx"])) for d in data]





def evaluate_from_best_params(
    params_df,
    base_df,
    model_cfg,
    splits_dir="dataset/splits",
    embeddings_dir="notebooks/data/embeddings",
    label_col="label",
    scale=True,
    random_state=42,
):

    df = params_df
    base_df = base_df.reset_index(drop=True)
    y = base_df[label_col].astype(int).to_numpy()
    cache_dir = _get_embedding_cache_dir(embeddings_dir)   
    records = []
    dataset_hash = _compute_dataset_hash(base_df)
    cache_dir = _get_embedding_cache_dir(embeddings_dir)
    

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating models"):
        split_mode = row["split"]
        a_name = row["aptamer_encoder"]
        p_name = row["protein_encoder"]

        print(f"→ Evaluating {split_mode} | {a_name} × {p_name}")

        # --- load embeddings ---
        Xa = np.load(cache_dir / f"apt_{a_name}_{dataset_hash}.npy").astype(np.float32)
        Xp = np.load(cache_dir / f"prot_{p_name}_{dataset_hash}.npy").astype(np.float32)
        X = np.concatenate([Xa, Xp], axis=1)

        # --- load splits ---
        splits = load_splits_with_threshold(split_mode, base_dir=splits_dir)

        # --- parse params ---
        if isinstance(row["best_params"], str):
            best_params = ast.literal_eval(row["best_params"])
        else:
            best_params = row["best_params"]

        roc_scores, mcc_scores = [], []

        for tr, va in splits:
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]

            if scale:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xva = scaler.transform(Xva)

            clf = model_cfg["model_class"](
                **best_params,
                random_state=random_state,
                verbose=-1
            )
            clf.fit(Xtr, ytr)

            if model_cfg.get("use_proba", True):
                s = clf.predict_proba(Xva)[:, 1]
            else:
                s = clf.decision_function(Xva)

            yhat = clf.predict(Xva)

            roc = roc_auc_score(yva, s) if len(np.unique(yva)) > 1 else np.nan
            mcc = matthews_corrcoef(yva, yhat)

            roc_scores.append(roc)
            mcc_scores.append(mcc)

        records.append({
            "split": split_mode,
            "aptamer_encoder": a_name,
            "protein_encoder": p_name,
            "ROC-AUC mean": np.nanmean(roc_scores),
            "ROC-AUC std": np.nanstd(roc_scores),
            "MCC mean": np.nanmean(mcc_scores),
            "MCC std": np.nanstd(mcc_scores),
            "model": model_cfg["name"],
        })

        del Xa, Xp, X, clf
        gc.collect()

    return pd.DataFrame(records)

lgbm_cfg = {
    "name": "LGBM",
    "model_class": LGBMClassifier,
    "use_proba": True
}


results_dir = root / "notebooks" / "results" / "threshold_results"

def split_sort_key(path):
    name = path.name  # e.g. split_065_090
    try:
        _, part1, part2 = name.split("_")
    except ValueError:
        return (1, 0)

    priority = 0 if part2.startswith("09") else 1

    return (priority, int(part2), int(part1))



for split_path in tqdm(sorted(splits_dir.glob("split_*"), key=split_sort_key), desc="Evaluate splits"):
    if not split_path.is_dir():
        continue

    split_name = split_path.name  # e.g. split_05_05
    tqdm.write(f"Split name: {split_name}")
    output_file = results_dir / f"{split_name}.csv"

    if split_name.startswith("split_050") or split_name.startswith("split_055") or split_name.startswith("split_060") or split_name.startswith("split_065"):
        tqdm.write(f"Skipping {split_name}")
        continue

    if output_file.exists():
        tqdm.write(f"Already exists, skipping: {output_file.name}")
        continue

    df_from_best = evaluate_from_best_params(
        params_df=clf_results,
        model_cfg=lgbm_cfg,
        base_df=base_df,
        label_col="Class label",
        embeddings_dir=embeddings_dir,
        splits_dir=split_path
    )

    # сохраняем результат
    df_from_best.to_csv(output_file)
    tqdm.write(f"Completed : {split_name}.csv")
