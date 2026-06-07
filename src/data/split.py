import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from difflib import SequenceMatcher


# ===========================
# Internal utilities
# ===========================

def _seq_identity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _cluster_sequences(seqs, threshold=0.8):
    """
    Cluster sequences by identity >= threshold with transitivity.
    Uses connected components in similarity graph.
    """

    n = len(seqs)
    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _seq_identity(seqs[i], seqs[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = [-1] * n
    cluster_id = 0

    def dfs(v):
        stack = [v]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                groups[u] = cluster_id
                for nei in adj[u]:
                    if not visited[nei]:
                        stack.append(nei)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            cluster_id += 1

    return np.array(groups)


# ==============================================================
#   RANDOM
# ==============================================================

def random_group_splits(
    df,
    group_cols=("Aptamer_sequence", "Protein_sequence"),
    n_splits=5,
    random_state=42
):
    """
    Random Group KFold:
    ensures disjoint groups across folds, without stratification.

    :param df: DataFrame with data
    :param group_cols: columns used to define groups (combined)
    :param n_splits: number of folds
    :param random_state: random seed
    :return: list of (train_idx, val_idx) tuples
    """
    groups = df[list(group_cols)].astype(str).agg("||".join, axis=1)

    # Randomly shuffle unique groups
    unique_groups = np.unique(groups.values)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_groups)

    # Map shuffled groups back
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    shuffled_group_ids = groups.map(group_to_id)

    gkf = GroupKFold(n_splits=n_splits)

    return [
        (tr, va)
        for tr, va in gkf.split(
            X=np.zeros(len(df)),
            groups=shuffled_group_ids.values
        )
    ]


# ==============================================================
#   STRATIFIED GROUP
# ==============================================================

def stratified_group_splits(
    df,
    label_col="label",
    group_cols=("Aptamer_sequence", "Protein_sequence"),
    n_splits=5,
    random_state=42
):
    """
    Stratified group KFold:
    ensures disjoint groups across folds, while preserving label distribution.

    :param df: DataFrame with data
    :param label_col: column containing labels for stratification
    :param group_cols: columns used to define groups (combined)
    :param n_splits: number of folds
    :param random_state: random seed
    :return: list of (train_idx, val_idx) tuples
    """
    groups = df[list(group_cols)].astype(str).agg("||".join, axis=1)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    return [
        (tr, va)
        for tr, va in sgkf.split(
            X=np.zeros(len(df)),
            y=df[label_col].values,
            groups=groups.values
        )
    ]



# ==============================================================
# STRICT DISJOINT APTAMER SPLITS
# ==============================================================

def disjoint_aptamer_splits(
    df,
    n_splits=5,
    col="Aptamer_sequence",
    label_col="label",
    random_state=42,
    seq_similarity_threshold=0.8
):
    """Sequence groups are sequence-similarity clusters."""

    unique_seqs = df[col].astype(str).drop_duplicates().tolist()
    unique_clusters = _cluster_sequences(unique_seqs, threshold=seq_similarity_threshold)

    # map back to rows
    seq_to_cluster = {seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)}
    groups = df[col].map(seq_to_cluster).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return [(tr, va) for tr, va in sgkf.split(
        X=np.zeros(len(df)), y=df[label_col].values, groups=groups
    )]


# ==============================================================
# STRICT DISJOINT PROTEIN SPLITS
# ==============================================================


def disjoint_protein_splits(
    df,
    n_splits=5,
    col="Protein_sequence",
    label_col="label",
    random_state=42,
    seq_similarity_threshold=0.8
):
    """Sequence groups are sequence-similarity clusters."""

    unique_seqs = df[col].astype(str).drop_duplicates().tolist()
    unique_clusters = _cluster_sequences(unique_seqs, threshold=seq_similarity_threshold)

    # map back to rows
    seq_to_cluster = {seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)}
    groups = df[col].map(seq_to_cluster).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return [(tr, va) for tr, va in sgkf.split(
        X=np.zeros(len(df)), y=df[label_col].values, groups=groups
    )]