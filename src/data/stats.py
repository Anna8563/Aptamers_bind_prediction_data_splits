import pandas as pd
from difflib import SequenceMatcher
from src.models.screening import load_splits


def _seq_identity(a: str, b: str) -> float:
    """Sequence identity (0-1) using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


def analyze_generalization_splits_apt_prot(
    df: pd.DataFrame,
    split_modes=("stratified", "disjoint_aptamer", "disjoint_protein"),
    splits_dir="dataset/splits",
    apt_col="Aptamer_sequence",
    protein_col="Protein_sequence",
    apt_identity_threshold=0.8,
    prot_identity_threshold=0.8,
    max_train_seqs_for_similarity=20000,
):
    """
    Analyze each fold within each split_mode.
    Returns 15 rows (3 splits × 5 folds) instead of 3.
    """
    df = df.reset_index(drop=True)

    results = []

    for split_mode in split_modes:
        splits = load_splits(split_mode, base_dir=splits_dir)

        for fold_id, (train_idx, test_idx) in enumerate(splits):

            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            
            train_pairs = list(zip(train[apt_col], train[protein_col]))
            test_pairs = list(zip(test[apt_col], test[protein_col]))

            train_pair_dups = len(train_pairs) - len(set(train_pairs))
            test_pair_dups = len(test_pairs) - len(set(test_pairs))

            train_seq_dups = train[apt_col].duplicated().sum()
            test_seq_dups = test[apt_col].duplicated().sum()

            train_prot_dups = train[protein_col].duplicated().sum()
            test_prot_dups = test[protein_col].duplicated().sum()

            
            shared_pairs = set(train_pairs).intersection(set(test_pairs))
            shared_seqs = set(train[apt_col]).intersection(set(test[apt_col]))
            shared_proteins = set(train[protein_col]).intersection(set(test[protein_col]))

            # --- 3. Aptamer sequence-similarity leakage ---
            unique_train_seqs = list(train[apt_col].drop_duplicates())
            unique_test_seqs = list(test[apt_col].drop_duplicates())

            if len(unique_train_seqs) > max_train_seqs_for_similarity:
                unique_train_seqs = unique_train_seqs[:max_train_seqs_for_similarity]

            leakage_count = 0
            for s_test in unique_test_seqs:
                max_id = 0.0
                for s_train in unique_train_seqs:
                    sim = _seq_identity(s_test, s_train)
                    if sim > max_id:
                        max_id = sim
                    if max_id >= apt_identity_threshold:
                        break
                if max_id >= apt_identity_threshold:
                    leakage_count += 1

            frac_leaky_seqs = leakage_count / max(1, len(unique_test_seqs))

            # --- 4. Protein sequence-similarity leakage ---

            unique_prot_train_seqs = list(train[protein_col].drop_duplicates())
            unique_prot_test_seqs = list(test[protein_col].drop_duplicates())

            if len(unique_prot_train_seqs) > max_train_seqs_for_similarity:
                unique_prot_train_seqs = unique_prot_train_seqs[:max_train_seqs_for_similarity]

            prot_leakage_count = 0
            for s_test in unique_prot_test_seqs:
                max_id = 0.0
                for s_train in unique_prot_train_seqs:
                    sim = _seq_identity(s_test, s_train)
                    if sim > max_id:
                        max_id = sim
                    if max_id >= prot_identity_threshold:
                        break
                if max_id >= prot_identity_threshold:
                    prot_leakage_count += 1

            frac_leaky_prot_seqs = prot_leakage_count / max(1, len(unique_prot_test_seqs))

            results.append(
                {
                    "split": split_mode,
                    "fold": fold_id,
                    
                    "n_train_pairs": len(train),
                    "n_test_pairs": len(test),
                    "n_unique_train_pairs": len(set(train_pairs)),
                    "n_unique_test_pairs": len(set(test_pairs)),
                    
                    "train_pair_duplicates": train_pair_dups,
                    "test_pair_duplicates": test_pair_dups,
                    "train_seq_duplicates": int(train_seq_dups),
                    "test_seq_duplicates": int(test_seq_dups),
                    "train_protein_duplicates": int(train_prot_dups),
                    "test_protein_duplicates": int(test_prot_dups),
                    
                    "shared_pairs": len(shared_pairs),
                    "shared_seqs": len(shared_seqs),
                    "shared_proteins": len(shared_proteins),

                    f"test_seqs_identity>={apt_identity_threshold}": leakage_count,
                    f"frac_test_seqs_identity>={apt_identity_threshold}": frac_leaky_seqs,

                    f"test_prot_seqs_identity>={prot_identity_threshold}": prot_leakage_count,
                    f"frac_test_prot_seqs_identity>={prot_identity_threshold}": frac_leaky_prot_seqs,
                }
            )

    return pd.DataFrame(results)


def analyze_generalization_splits(
    df: pd.DataFrame,
    split_modes=("stratified", "disjoint_aptamer", "disjoint_protein"),
    splits_dir="dataset/splits",
    sequence_col="Aptamer_sequence",
    protein_col="Protein_sequence",
    identity_threshold=0.8,
    max_train_seqs_for_similarity=2000,
):
    """
    Analyze each fold within each split_mode.
    Returns 15 rows (3 splits × 5 folds) instead of 3.
    """
    df = df.reset_index(drop=True)

    results = []

    for split_mode in split_modes:
        splits = load_splits(split_mode, base_dir=splits_dir)

        for fold_id, (train_idx, test_idx) in enumerate(splits):

            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            
            train_pairs = list(zip(train[sequence_col], train[protein_col]))
            test_pairs = list(zip(test[sequence_col], test[protein_col]))

            train_pair_dups = len(train_pairs) - len(set(train_pairs))
            test_pair_dups = len(test_pairs) - len(set(test_pairs))

            train_seq_dups = train[sequence_col].duplicated().sum()
            test_seq_dups = test[sequence_col].duplicated().sum()

            train_prot_dups = train[protein_col].duplicated().sum()
            test_prot_dups = test[protein_col].duplicated().sum()

            
            shared_pairs = set(train_pairs).intersection(set(test_pairs))
            shared_seqs = set(train[sequence_col]).intersection(set(test[sequence_col]))
            shared_proteins = set(train[protein_col]).intersection(set(test[protein_col]))

            unique_train_seqs = list(train[sequence_col].drop_duplicates())
            unique_test_seqs = list(test[sequence_col].drop_duplicates())

            if len(unique_train_seqs) > max_train_seqs_for_similarity:
                unique_train_seqs = unique_train_seqs[:max_train_seqs_for_similarity]

            leakage_count = 0
            for s_test in unique_test_seqs:
                max_id = 0.0
                for s_train in unique_train_seqs:
                    sim = _seq_identity(s_test, s_train)
                    if sim > max_id:
                        max_id = sim
                    if max_id >= identity_threshold:
                        break
                if max_id >= identity_threshold:
                    leakage_count += 1

            frac_leaky_seqs = leakage_count / max(1, len(unique_test_seqs))



            unique_prot_train_seqs = list(train[protein_col].drop_duplicates())
            unique_prot_test_seqs = list(test[protein_col].drop_duplicates())

            if len(unique_prot_train_seqs) > max_train_seqs_for_similarity:
                unique_prot_train_seqs = unique_prot_train_seqs[:max_train_seqs_for_similarity]

            prot_leakage_count = 0
            for s_test in unique_prot_test_seqs:
                max_id = 0.0
                for s_train in unique_prot_train_seqs:
                    sim = _seq_identity(s_test, s_train)
                    if sim > max_id:
                        max_id = sim
                    if max_id >= identity_threshold:
                        break
                if max_id >= identity_threshold:
                    prot_leakage_count += 1

            frac_leaky_prot_seqs = prot_leakage_count / max(1, len(unique_prot_test_seqs))

            results.append(
                {
                    "split": split_mode,
                    "fold": fold_id,
                    
                    "n_train_pairs": len(train),
                    "n_test_pairs": len(test),
                    "n_unique_train_pairs": len(set(train_pairs)),
                    "n_unique_test_pairs": len(set(test_pairs)),
                    
                    "train_pair_duplicates": train_pair_dups,
                    "test_pair_duplicates": test_pair_dups,
                    "train_seq_duplicates": int(train_seq_dups),
                    "test_seq_duplicates": int(test_seq_dups),
                    "train_protein_duplicates": int(train_prot_dups),
                    "test_protein_duplicates": int(test_prot_dups),
                    
                    "shared_pairs": len(shared_pairs),
                    "shared_seqs": len(shared_seqs),
                    "shared_proteins": len(shared_proteins),
                    f"test_seqs_identity>={identity_threshold}": leakage_count,
                    f"frac_test_seqs_identity>={identity_threshold}": frac_leaky_seqs,

                    f"test_prot_seqs_identity>={identity_threshold}": prot_leakage_count,
                    f"frac_test_prot_seqs_identity>={identity_threshold}": frac_leaky_prot_seqs,
                }
            )

    return pd.DataFrame(results)
