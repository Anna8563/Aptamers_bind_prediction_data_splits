import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Levenshtein import distance as levenshtein_distance

def compute_pairwise_levenshtein(sequences):
    dists = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            dists.append(levenshtein_distance(sequences[i], sequences[j]))
    return dists

# ============================================================
# Dataset summary
# ============================================================

def describe_dataset(df, name="Dataset", apt_col='Aptamer_sequence', prot_col='Protein_sequence', type_col='Aptamer_type'):
    aptamers = df[apt_col].dropna()
    proteins = df[prot_col].dropna()
    types = df[type_col].str.upper()

    stats = {
        'Dataset': name,
        'N rows': len(df),
        'N unique aptamers': aptamers.nunique(),
        'DNA:RNA ratio': f"{(types == 'DNA').sum()}:{(types == 'RNA').sum()}",
        'Mean aptamer length ± std': f"{aptamers.str.len().mean():.1f} ± {aptamers.str.len().std():.1f}",
        'N unique proteins': proteins.nunique(),
        'Mean protein length ± std': f"{proteins.str.len().mean():.1f} ± {proteins.str.len().std():.1f}"
    }

    return pd.DataFrame([stats])



# ============================================================
# Distribution plots
# ============================================================

def plot_dataset_distributions(df, figs_dir="figs", filename="sequence_distributions.png", base_color_a="#9da2fa", base_color_p="#9da2fa", apt_col='Aptamer_sequence', prot_col='Protein_sequence'):
    """
    Plots distributions:
      1) Aptamer length
      2) Aptamer pairwise Levenshtein distances
      3) Protein length
      4) Protein pairwise Levenshtein distances
    Saves figure in `figs_dir` at 1200 dpi.
    """

    os.makedirs(figs_dir, exist_ok=True)
    base_color_a = base_color_a  
    base_color_p = base_color_p
    base_color = "#344966" # Indigo

    sns.set_context("talk")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # --- 1. Aptamer length
    aptamers = df[apt_col].dropna()
    sns.histplot(aptamers.str.len(), bins=30, ax=axes[0],
                 color=base_color_a, kde=True)
    axes[0].set_title("Aptamer length", fontsize=24)
    axes[0].set_xlabel("Length", fontsize=20)
    axes[0].set_ylabel("Count", fontsize=20)

    # --- 2. Aptamer Levenshtein distances
    aptamer_list = aptamers.unique().tolist()
    if len(aptamer_list) > 1:
        apt_lev = compute_pairwise_levenshtein(aptamer_list)
        sns.histplot(apt_lev, bins=30, ax=axes[1],
                     color=base_color_a, kde=True)
        axes[1].set_title("Aptamer Levenshtein distances", fontsize=24)
        axes[1].set_xlabel("Distance", fontsize=20)
        axes[1].set_ylabel("Count", fontsize=20)
    else:
        axes[1].text(0.5, 0.5, "Not enough sequences",
                     ha="center", va="center", color=base_color_a, fontsize=16)
        axes[1].set_axis_off()

    # --- 3. Protein length
    proteins = df[prot_col].dropna()
    sns.histplot(proteins.str.len(), bins=30, ax=axes[2],
                 color=base_color_p, kde=True)
    axes[2].set_title("Protein length", fontsize=24)
    axes[2].set_xlabel("Length", fontsize=20)
    axes[2].set_ylabel("Count", fontsize=20)

    # --- 4. Protein Levenshtein distances
    protein_list = proteins.unique().tolist()
    if len(protein_list) > 1:
        prot_lev = compute_pairwise_levenshtein(protein_list)
        sns.histplot(prot_lev, bins=30, ax=axes[3],
                     color=base_color_p, kde=True)
        axes[3].set_title("Protein Levenshtein distances", fontsize=24)
        axes[3].set_xlabel("Distance", fontsize=20)
        axes[3].set_ylabel("Count", fontsize=20)
    else:
        axes[3].text(0.5, 0.5, "Not enough sequences",
                     ha="center", va="center", color=base_color_p, fontsize=16)
        axes[3].set_axis_off()

    # --- Styling
    for ax in axes:
        ax.spines['top'].set_color("#90A4AE")
        ax.spines['right'].set_color("#90A4AE")
        ax.tick_params(colors=base_color, labelsize=12)
        ax.yaxis.label.set_color(base_color)
        ax.xaxis.label.set_color(base_color)

    plt.tight_layout()
    out_path = os.path.join(figs_dir, filename)
    plt.savefig(out_path, dpi=1200)
    plt.show()
