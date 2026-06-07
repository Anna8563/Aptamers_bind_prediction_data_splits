# Aptamer-Protein Interaction Prediction
## Overview

Experimental identification of aptamer–protein interactions is costly and time-consuming, while existing computational methods often struggle to generalize to unseen protein targets. This project investigates data-driven approaches for interaction prediction by combining protein and aptamer representations. The primary goal is to systematically evaluate how different feature representations and data splitting strategies affect predictive performance and model generalization.

## Repository Structure
├── notebooks_and_scripts

│   ├── compute_aptamer_embeddings.py

│   ├── compute_aptamer_embeddings_dnabert.py

│   ├── compute_embeddings_ankh.py
│   ├── compute_embeddings_esmc.py
│   ├── compute_embeddings_prot_t5.py
│   ├── train_ml_models*.py
│   ├── train_dl_models*.py
│   ├── data_exploration.ipynb
│   └── results_plots.ipynb
├── src
│   ├── data           dataset statistics and splitting utilities
│   ├── encoders       aptamer and protein embedding pipelines
│   ├── models         interaction prediction models screening
│   └── viz            visualization utilities
└── README.md

.
├── notebooks_and_scripts
│   ├── compute_aptamer_embeddings_dnabert.py
│   ├── compute_aptamer_embeddings.py
│   ├── compute_embeddings_ankh.py
│   ├── compute_embeddings_esmc.py
│   ├── compute_embeddings_prot_t5.py
│   ├── data_exploration.ipynb
│   ├── results_plots.ipynb
│   ├── train_dl_models.py
│   ├── train_dl_models_thresholds.py
│   ├── train_ml_models_only_aptamers.py
│   ├── train_ml_models_only_proteins.py
│   ├── train_ml_models.py
│   └── train_ml_models_with_thresholds.py
├── README.md
├── src
│   ├── data
│   │   ├── split.py
│   │   └── stats.py
│   ├── encoders
│   │   ├── aptamer_encoders_dnabert.py
│   │   ├── aptamer_encoders.py
│   │   ├── protein_encoders_esmc.py
│   │   ├── protein_encoders_prot_t5.py
│   │   └── protein_encoders.py
│   ├── models
│   │   └── screening.py
│   └── viz
│       └── plots.py
## Encoders

Aptamers:

Sequence-based feature extraction
DNABERT-2 embeddings
GENA-LM

Proteins:

ProtT5
ESMC
ANKH
