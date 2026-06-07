import os
import gc
import pandas as pd
import numpy as np
import ast
import json
import re
import warnings
from tqdm import tqdm
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from lightgbm import LGBMClassifier
import sys
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)


from src.models.screening import load_splits, screen_multimodel_optuna_prot

root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))


clf_results_path = root / "notebooks" / "notebooks" /" checkpoints" /" multimodel_results_screening_complete.csv"
clf_results = pd.read_csv(clf_results_path)

embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"
splits_dir = root / "dataset" / "splits"

checkpoint_dir = root / "notebooks" / "notebooks" / "checkpoints_proteins_2"


dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv" 
base_df = pd.read_csv(dataset_path)


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
        Xp = np.load(cache_dir / f"prot_{p_name}_{dataset_hash}.npy").astype(np.float32)
        X = Xp

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
            "protein_encoder": p_name,
            "ROC-AUC mean": np.nanmean(roc_scores),
            "ROC-AUC std": np.nanstd(roc_scores),
            "MCC mean": np.nanmean(mcc_scores),
            "MCC std": np.nanstd(mcc_scores),
            "model": model_cfg["name"],
        })

        del Xp, X, clf
        gc.collect()

    return pd.DataFrame(records)

lgbm_cfg = {
    "name": "LGBM",
    "model_class": LGBMClassifier,
    "use_proba": True
}


prot_cfgs = [
    {"name": "ESMC"},
    {"name": "Prot_T5"},
    {"name": "Ankh"},
]

import gc
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

# Очищаем память перед началом
gc.collect()

# Запускаем по одному split_mode за раз
all_results = []

from lightgbm import LGBMClassifier

lgbm_cfg = {
    "name": "lgbm",
    "model_class": LGBMClassifier,
    "param_space": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        },
    "fit_params": lambda trial: {"verbose": False}
}




print(checkpoint_dir)
print(embeddings_dir)
# Load dataset
df = pd.read_csv(dataset_path)

print("Dataset loaded:", df.shape)

# Check available splits
print("Available splits:", os.listdir(splits_dir))

all_results = []

model_cfgs = [lgbm_cfg]


results_dir = root / "notebooks" / "results" / "only_proteins"

def split_sort_key(path):
    name = path.name  # split_065_090
    try:
        _, part1, part2 = name.split("_")
    except ValueError:
        return (1, 0)

    priority = 0 if part2.startswith("09") else 1

    return (priority, int(part2), int(part1))



for split_path in tqdm(sorted(splits_dir.glob("split_*")), desc="Evaluate splits"):
    if not split_path.is_dir():
        continue

    split_name = split_path.name  # e.g. split_05_05
    tqdm.write(f"Split name: {split_name}")



    # split_050_060 -> ["050", "060"]
    parts = re.findall(r"\d+", split_name)

    if any(not p.endswith("0") for p in parts):
        tqdm.write(f"Skipping {split_name}")
        continue
    output_file = results_dir / f"{split_name}.csv"
    if split_name.startswith("split_050") or split_name.startswith("split_055") or split_name.startswith("split_060") or split_name.startswith("split_065"):
        tqdm.write(f"Skipping {split_name}")
        continue

    if output_file.exists():
        tqdm.write(f"Already exists, skipping: {output_file.name}")
        continue


    for model_cfg in model_cfgs:
        for split_mode in ["stratified", "disjoint_protein", "disjoint_aptamer"]:
        
            print(f"\n{'='*60}")
            print(f"MODEL: {model_cfg['name']} | SPLIT: {split_mode}")
            print(f"{'='*60}\n")
            
            try:
                results_df = screen_multimodel_optuna_prot(
                    df,
                    prot_cfgs,
                    model_cfg=model_cfg,
                    split_modes=(split_mode,),  # По одному за раз
                    n_trials=10,  # Количество trials
                    metric="roc_auc",
                    splits_dir=split_path,
                    embeddings_dir=embeddings_dir,
                    use_cached_embeddings=True,  # Используем кэш эмбеддингов
                    checkpoint_dir=checkpoint_dir,  # Директория для чекпоинтов
                    results_file=checkpoint_dir / f"{model_cfg['name']}_results_{split_mode}.csv",  # Онлайн сохранение
                    resume=True,  # Продолжить с чекпоинта если есть
                    use_pruning=True,  # Pruning для ускорения (~30-50%)
                    label_col = "Class label",
                    threshold = split_name
                )
                if results_df is not None and not results_df.empty:
                    all_results.append(results_df)
                
                # Сохраняем финальные результаты
                results_df.to_csv(f'{model_cfg["name"]}_results_{split_mode}_{split_name}.csv', index=False)
                print(f"✓ Saved final results for {split_mode}")
                
            except Exception as e:
                print(f"✗ Error processing {split_mode}: {e}")
                import traceback
                traceback.print_exc()
                # Пытаемся загрузить частичные результаты
                checkpoint_file = checkpoint_dir / f"{model_cfg['name']}_results_{split_mode}_{split_name}.csv"
                if Path(checkpoint_file).exists():
                    print(f"⚠ Загружаем частичные результаты из {checkpoint_file}")
                    try:
                        partial_df = pd.read_csv(checkpoint_file)
                        if results_df is not None and not results_df.empty:
                            all_results.append(partial_df)
                    except:
                        pass
            
            # Очищаем память после каждого split_mode
            gc.collect()
            print(f"Memory cleared after {split_mode}\n")

    # Объединяем все результаты
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_path = checkpoint_dir / 'multimodel_results_screening_complete_only_proteins.csv'
        results_df.to_csv(results_path, index=False)
        print(f"✓ All results saved to {results_path}")
    else:
        print("✗ No results collected")
