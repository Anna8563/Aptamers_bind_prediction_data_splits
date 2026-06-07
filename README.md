# Aptamer-Protein Interaction Prediction
## Overview

Experimental identification of aptamer–protein interactions is costly and time-consuming, while existing computational methods often struggle to generalize to unseen protein targets. This project investigates data-driven approaches for interaction prediction by combining protein and aptamer representations. The primary goal is to systematically evaluate how different feature representations and data splitting strategies affect predictive performance and model generalization.

## Repository Structure

```text
.
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
│   ├── data
│   ├── encoders
│   ├── models
│   └── viz
└── README.md
```

| Directory               | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `src/data`              | Dataset statistics and data splitting utilities                 |
| `src/encoders`          | Aptamer and protein feature extraction and embedding generation |
| `src/models`            | Model training, screening, and evaluation workflows             |
| `src/viz`               | Visualization and plotting utilities                            |
| `notebooks_and_scripts` | Experiment scripts, notebooks, and result analysis              |

## Encoders

Aptamers:

Sequence-based feature extraction
DNABERT-2 embeddings
GENA-LM

Proteins:

ProtT5
ESMC
ANKH
