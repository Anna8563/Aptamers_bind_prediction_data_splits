# Add project root to sys.path so "src" is importable
import sys
from pathlib import Path

root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))

import numpy as np
if np.__version__.startswith('2.'):
    import subprocess
    import sys
    print(f"NumPy version {np.__version__} detected. Downgrading to < 2.0 for RDKit compatibility...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "--force-reinstall", "-q"])
    print("NumPy downgraded. Please restart the kernel and run again.")
    raise RuntimeError("NumPy downgraded. Please restart the kernel (Kernel -> Restart Kernel) and run cells again.")
else:
    print(f"NumPy version {np.__version__} is compatible with RDKit.")


# === Imports ===
import os
import pandas as pd
from src.models.screening import screen_multimodel_optuna_apt_prot


split = "split_08_08"
# Paths
dataset_path = root / "dataset" / "Aptamer_protein_dataset.csv"
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"
splits_dir = root / "dataset" / "splits" / split
checkpoint_dir = root / "notebooks" / "notebooks" / "checkpoints"


print(checkpoint_dir)
print(embeddings_dir)
# Load dataset
df = pd.read_csv(dataset_path)


print("Dataset loaded:", df.shape)

# Check available splits
print("Available splits:", os.listdir(splits_dir))


apt_cfgs = [
    {"name": "OneHot"},
    {"name": "Kmer3"},
    {"name": "Kmer4"},
    {"name": "GENA"},
    {"name": "DNABERT2"}
]

prot_cfgs = [
    {"name": "ESMC"},
    {"name": "Prot_T5"},
    {"name": "Ankh"},
    
]


import gc
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


from catboost import CatBoostClassifier

cat_cfg = {
    "name": "catboost",
    "model_class": CatBoostClassifier,
    "param_space": lambda trial: {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    },
    "fit_params": lambda trial: {"verbose": False}
}


from xgboost import XGBClassifier
xgb_cfg = {
    "name": "xgb",
    "model_class": XGBClassifier,
    "param_space": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    },
}


from sklearn.ensemble import RandomForestClassifier
rf_cfg = {
    "name": "rf",
    "model_class": RandomForestClassifier,
    "param_space": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    },
    "fit_params": lambda trial: {},
    "use_proba": True
}


from sklearn.linear_model import LogisticRegression
lr_cfg = {
    "name": "logreg",
    "model_class": LogisticRegression,
    "param_space": lambda trial: {
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "max_iter": 300,
        "solver": "lbfgs",
    },
    "use_proba": True
}

#model_cfgs = [xgb_cfg, cat_cfg, rf_cfg, lr_cfg]

model_cfgs = [lgbm_cfg]

for model_cfg in model_cfgs:
    for split_mode in ["stratified", "disjoint_protein", "disjoint_aptamer"]:
    
        print(f"\n{'='*60}")
        print(f"MODEL: {model_cfg['name']} | SPLIT: {split_mode}")
        print(f"{'='*60}\n")
        
        try:
            results_df = screen_multimodel_optuna_apt_prot(
                df,
                apt_cfgs,
                prot_cfgs,
                model_cfg=model_cfg,
                split_modes=(split_mode,),  # По одному за раз
                n_trials=10,  # Количество trials
                metric="roc_auc",
                splits_dir=splits_dir,
                embeddings_dir=embeddings_dir,
                use_cached_embeddings=True,  # Используем кэш эмбеддингов
                checkpoint_dir=checkpoint_dir,  # Директория для чекпоинтов
                results_file=checkpoint_dir / f"{model_cfg['name']}_results_{split_mode}.csv",
                resume=True,  # Продолжить с чекпоинта если есть
                use_pruning=True,  # Pruning для ускорения (~30-50%)
                label_col = "Class label"
            )
            if results_df is not None and not results_df.empty:
                all_results.append(results_df)
            
            # Сохраняем финальные результаты
            results_df.to_csv(f'{model_cfg["name"]}_split_{split}_results_{split_mode}.csv', index=False)
            print(f"✓ Saved final results for {split_mode}")
            
        except Exception as e:
            print(f"✗ Error processing {split_mode}: {e}")
            import traceback
            traceback.print_exc()
            # Пытаемся загрузить частичные результаты
            checkpoint_file = checkpoint_dir / f"{model_cfg['name']}_results_{split_mode}.csv"
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
    results_path = checkpoint_dir / 'multimodel_results_screening_complete.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ All results saved to {results_path}")
else:
    print("✗ No results collected")
