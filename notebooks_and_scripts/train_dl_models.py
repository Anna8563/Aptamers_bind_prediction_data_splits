
"""
SPLITS_DIR =  root / "dataset" / "splits" / "split_08_08"
"""

import sys
from pathlib import Path


root = Path(__file__).resolve()
while not (root / "src").exists() and root.parent != root:
    root = root.parent

sys.path.insert(0, str(root))


import os
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score
)


# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Model visualization and parameter counting
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("⚠ torchinfo not available - will use manual parameter counting")

# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# === Data Loading ===

from src.models.screening import load_splits_with_threshold
print("✓ Using existing load_splits")

# === Load Precomputed Embeddings ===

from src.models.screening import load_embeddings
print("✓ Using existing load_embeddings function")


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_parameter_count(count):
    """Format parameter count in human-readable format."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    else:
        return str(count)


def visualize_model_architecture(model, model_name, apt_dim, prot_dim, save_path=None):
    """
    Visualize model architecture as a diagram.
    """
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, f"{model_name} Architecture", 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Count parameters
    params = count_parameters(model)
    param_text = f"Total: {format_parameter_count(params['total'])} | "
    param_text += f"Trainable: {format_parameter_count(params['trainable'])}"
    ax.text(5, 11, param_text, ha='center', va='top', fontsize=10, style='italic')
    
    # Draw architecture based on model type
    if 'PretrainedEncoder' in model_name:
        # Aptamer branch
        apt_box = FancyBboxPatch((0.5, 8), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(apt_box)
        ax.text(1.75, 9, f"Aptamer\nInput\n({apt_dim})", ha='center', va='center', fontsize=9)
        
        # Protein branch
        prot_box = FancyBboxPatch((7.5, 8), 2, 1.5, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(prot_box)
        ax.text(8.75, 9, f"Protein\nInput\n({prot_dim})", ha='center', va='center', fontsize=9)
        
        # Projection layers
        apt_proj = FancyBboxPatch((0.5, 6), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(apt_proj)
        ax.text(1.75, 6.5, "Projection\nLayers", ha='center', va='center', fontsize=8)
        
        prot_proj = FancyBboxPatch((7.5, 6), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black', linewidth=1.5)
        ax.add_patch(prot_proj)
        ax.text(8.75, 6.5, "Projection\nLayers", ha='center', va='center', fontsize=8)
        
        # Fusion
        fusion_box = FancyBboxPatch((3.5, 4), 3, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(fusion_box)
        ax.text(5, 4.75, "Fusion Head\n(Concatenate + MLP)", ha='center', va='center', fontsize=9)
        
        # Output
        out_box = FancyBboxPatch((4.5, 2), 1, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(out_box)
        ax.text(5, 2.4, "Output\n(Logits)", ha='center', va='center', fontsize=9)
        
        # Arrows
        arrow1 = FancyArrowPatch((1.75, 8), (1.75, 7.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch((8.75, 8), (8.75, 7.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow2)
        arrow3 = FancyArrowPatch((1.75, 6), (3.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow3)
        arrow4 = FancyArrowPatch((8.75, 6), (6.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow4)
        arrow5 = FancyArrowPatch((5, 4), (5, 3.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow5)
        
    elif 'TwoTower' in model_name:
        # Two towers
        apt_tower = FancyBboxPatch((0.5, 7), 2.5, 3, boxstyle="round,pad=0.1",
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(apt_tower)
        ax.text(1.75, 8.5, "Aptamer Tower\n(MLP Layers)", ha='center', va='center', fontsize=9)
        
        prot_tower = FancyBboxPatch((7, 7), 2.5, 3, boxstyle="round,pad=0.1",
                                  facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(prot_tower)
        ax.text(8.25, 8.5, "Protein Tower\n(MLP Layers)", ha='center', va='center', fontsize=9)
        
        # Fusion
        fusion_box = FancyBboxPatch((3.5, 4), 3, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(fusion_box)
        ax.text(5, 4.75, "Fusion Layers\n(MLP)", ha='center', va='center', fontsize=9)
        
        # Output
        out_box = FancyBboxPatch((4.5, 2), 1, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(out_box)
        ax.text(5, 2.4, "Output", ha='center', va='center', fontsize=9)
        
        # Arrows
        arrow1 = FancyArrowPatch((1.75, 7), (1.75, 6.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch((8.25, 7), (8.25, 6.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow2)
        arrow3 = FancyArrowPatch((1.75, 6.5), (3.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow3)
        arrow4 = FancyArrowPatch((8.25, 6.5), (6.5, 5.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow4)
        arrow5 = FancyArrowPatch((5, 4), (5, 3.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow5)
        
    elif 'CrossAttention' in model_name:
        # Inputs
        apt_in = FancyBboxPatch((0.5, 8.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(apt_in)
        ax.text(1.75, 9, f"Aptamer\n({apt_dim})", ha='center', va='center', fontsize=9)
        
        prot_in = FancyBboxPatch((7.5, 8.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(prot_in)
        ax.text(8.75, 9, f"Protein\n({prot_dim})", ha='center', va='center', fontsize=9)
        
        # Projections
        apt_proj = FancyBboxPatch((0.5, 7), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(apt_proj)
        ax.text(1.75, 7.4, "Project", ha='center', va='center', fontsize=8)
        
        prot_proj = FancyBboxPatch((7.5, 7), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black', linewidth=1.5)
        ax.add_patch(prot_proj)
        ax.text(8.75, 7.4, "Project", ha='center', va='center', fontsize=8)
        
        # Cross-attention
        attn_box = FancyBboxPatch((3, 5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='lightyellow', edgecolor='black', linewidth=2)
        ax.add_patch(attn_box)
        ax.text(5, 5.75, "Cross-Attention\nLayers", ha='center', va='center', fontsize=9)
        
        # Classifier
        cls_box = FancyBboxPatch((3.5, 3), 3, 1, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(cls_box)
        ax.text(5, 3.5, "Classifier Head", ha='center', va='center', fontsize=9)
        
        # Output
        out_box = FancyBboxPatch((4.5, 1.5), 1, 0.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor='black', linewidth=2)
        ax.add_patch(out_box)
        ax.text(5, 1.9, "Output", ha='center', va='center', fontsize=9)
        
        # Arrows
        arrow1 = FancyArrowPatch((1.75, 8.5), (1.75, 8.2), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch((8.75, 8.5), (8.75, 8.2), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow2)
        arrow3 = FancyArrowPatch((1.75, 7), (3, 6.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow3)
        arrow4 = FancyArrowPatch((8.75, 7), (7, 6.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow4)
        arrow5 = FancyArrowPatch((5, 5), (5, 4.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow5)
        arrow6 = FancyArrowPatch((5, 3), (5, 2.5), arrowstyle='->', mutation_scale=20, linewidth=1.5, color='black')
        ax.add_patch(arrow6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Architecture saved to: {save_path}")
    plt.show()
    
    return params


print("✓ Visualization utilities ready")


# ### 1.3. Metrics Utilities (with Fallbacks)

def compute_classification_metrics(y_true, y_pred, y_scores=None):
    """
    Compute classification metrics: MCC, ROC-AUC, PR-AUC, F1.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Binary predictions
        y_scores: Continuous scores (probabilities) for ROC/PR-AUC
    
    Returns:
        dict with metrics
    """
    metrics = {}
    
    # MCC
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # F1
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC and PR-AUC (require continuous scores)
    if y_scores is not None and len(np.unique(y_true)) > 1:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_scores)
        metrics['PR-AUC'] = average_precision_score(y_true, y_scores)
    else:
        metrics['ROC-AUC'] = np.nan
        metrics['PR-AUC'] = np.nan
    
    return metrics



def aggregate_metrics(metrics_list):
    """
    Aggregate metrics across folds: mean ± std.
    
    Args:
        metrics_list: List of metric dicts (one per fold)
    
    Returns:
        dict with aggregated metrics
    """
    if not metrics_list:
        return {}
    
    # Collect all metric names
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [m.get(key, np.nan) for m in metrics_list]
        values = np.array(values)
        values = values[~np.isnan(values)]  # Remove NaN
        
        if len(values) > 0:
            aggregated[f"{key} mean"] = np.mean(values)
            aggregated[f"{key} std"] = np.std(values)
        else:
            aggregated[f"{key} mean"] = np.nan
            aggregated[f"{key} std"] = np.nan
    
    return aggregated


print("✓ Metrics utilities ready")


#     
# ## 2. Data Loading and Preparation
# 

#  
# Paths
DATASET_PATH = root / "dataset" / "Aptamer_protein_dataset.csv"
SPLITS_DIR =  root / "dataset" / "splits" / "split_08_08"
ARTIFACTS_DIR = root / "artifacts"
print("DATASET PATH", DATASET_PATH)
print("ARTIFACTS DIR", ARTIFACTS_DIR )
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "optuna"), exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, "metrics"), exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df['Class label'].value_counts()}")

# Prepare data
df = df.reset_index(drop=True)
y_class = df['Class label'].astype(int).values


# ## 3. PyTorch Dataset and DataLoader
class AptamerProteinDataset(Dataset):
    """PyTorch Dataset for aptamer-protein pairs."""
    
    def __init__(self, apt_features, prot_features, labels_class):
        """
        Args:
            apt_features: np.ndarray, aptamer features [N, D_apt]
            prot_features: np.ndarray, protein features [N, D_prot]
            labels_class: np.ndarray, binary classification labels [N]
        """
        self.apt_features = torch.FloatTensor(apt_features)
        self.prot_features = torch.FloatTensor(prot_features)
        self.labels_class = torch.LongTensor(labels_class)
    
    def __len__(self):
        return len(self.labels_class)
    
    def __getitem__(self, idx):
        item = {
            'apt': self.apt_features[idx],
            'prot': self.prot_features[idx],
            'label_class': self.labels_class[idx]
        }
        return item


print("✓ Dataset class defined")


#  
"""
Deep learning encoders with pretrained models and various top architectures.
"""

import math

class IdentityTop(nn.Module):
    """Identity top layer with optional linear projection."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass. x: (batch_size, seq_len, input_dim) -> (batch_size, output_dim)"""
        # Apply linear projection
        x = self.linear(x)
        
        # Mean pooling over sequence dimension
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class CNNTop(nn.Module):
    """CNN top layer with global max pooling."""
    
    def __init__(self, input_dim: int, output_dim: int, kernel_sizes=(3, 5, 7), dropout: float = 0.1):
        super().__init__()
        # Distribute output_dim across convolutions, handling remainder
        n_convs = len(kernel_sizes)
        base_channels = output_dim // n_convs
        remainder = output_dim % n_convs
        
        # Create convolutions with remainder distributed to first few layers
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Add 1 extra channel to first 'remainder' convolutions
            out_channels = base_channels + (1 if i < remainder else 0)
            self.convs.append(nn.Conv1d(input_dim, out_channels, k, padding=k//2))
        
        # Verify total output dimension matches output_dim
        actual_output_dim = sum(conv.out_channels for conv in self.convs)
        assert actual_output_dim == output_dim, f"Expected output_dim={output_dim}, got {actual_output_dim}"
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass. x: (batch_size, seq_len, input_dim) -> (batch_size, output_dim)"""
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            # Global max pooling
            conv_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(conv_out)
        
        # Concatenate outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass. x: (batch_size, seq_len, input_dim) -> (batch_size, input_dim)"""
        # Compute attention weights
        attention_weights = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask_value = -1e4 if attention_weights.dtype == torch.float16 else -1e9
            attention_weights = attention_weights.masked_fill(attention_mask == 0, mask_value)
        
        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Weighted sum
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return x


class LSTMTop(nn.Module):
    """LSTM top layer with attention pooling."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        # Use input_dim as LSTM input, hidden_dim as LSTM hidden size
        lstm_hidden = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.attention = AttentionPooling(lstm_output_dim)
        self.linear = nn.Linear(lstm_output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass. x: (batch_size, seq_len, input_dim) -> (batch_size, output_dim)"""
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention pooling
        x = self.attention(lstm_out, attention_mask)
        
        # Apply linear projection
        x = self.linear(x)
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class TransformerTop(nn.Module):
    """Transformer top layer with 1-2 encoder blocks."""
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = AttentionPooling(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass. x: (batch_size, seq_len, input_dim) -> (batch_size, output_dim)"""
        # Create padding mask for transformer
        if attention_mask is not None:
            padding_mask = attention_mask == 0
        else:
            padding_mask = None
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Apply attention pooling
        x = self.attention(x, attention_mask)
        
        # Apply linear projection
        x = self.linear(x)
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


print("✓ Top layer architectures (IdentityTop, CNNTop, LSTMTop, TransformerTop) defined")


# ## 4. Deep Learning Model Architectures
# 
# ### 4.1. Scenario 1: Pretrained Encoders + Unfreezing Layers
# 

class PretrainedEncoderModel(nn.Module):
    """
    Scenario 1: Pretrained encoders (HF transformers) with unfreezing capability.
    Uses frozen base encoders + trainable projection layers + fusion head.
    """
    
    def __init__(
        self,
        apt_dim,
        prot_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3,
        unfreeze_apt_layers=0,
        unfreeze_prot_layers=0,
        apt_encoder=None,
        prot_encoder=None
    ):
        super().__init__()
        self.apt_dim = apt_dim
        self.prot_dim = prot_dim
        self.apt_encoder = apt_encoder  
        self.prot_encoder = prot_encoder  
        
        # Deep projection layers for aptamer
        apt_layers = []
        prev_dim = apt_dim
        for dim in hidden_dims:
            apt_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.apt_proj = nn.Sequential(*apt_layers)
        
        # Deep projection layers for protein
        prot_layers = []
        prev_dim = prot_dim
        for dim in hidden_dims:
            prot_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.prot_proj = nn.Sequential(*prot_layers)
        
        # Deep fusion head
        fusion_dim = hidden_dims[-1] * 2
        fusion_dims = [fusion_dim] + hidden_dims[:-1] + [hidden_dims[-1] // 2]
        fusion_layers = []
        for i in range(len(fusion_dims) - 1):
            fusion_layers.extend([
                nn.Linear(fusion_dims[i], fusion_dims[i+1]),
                nn.BatchNorm1d(fusion_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        fusion_layers.append(nn.Linear(fusion_dims[-1], 1))
        self.fusion_head = nn.Sequential(*fusion_layers)
    
    def unfreeze_last_n_layers(self, encoder, n_layers):
        """Unfreeze last n layers of a transformer encoder."""
        if encoder is None:
            return
        # For HF models, iterate through encoder layers
        if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
            layers = encoder.encoder.layer
            total_layers = len(layers)
            for i in range(max(0, total_layers - n_layers), total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
    
    def forward(self, apt, prot):
        # Project features through deep networks
        apt_emb = self.apt_proj(apt)
        prot_emb = self.prot_proj(prot)
        
        # Concatenate
        fused = torch.cat([apt_emb, prot_emb], dim=1)
        
        # Classification head
        logits = self.fusion_head(fused)
        return logits.squeeze(-1)


print("✓ Scenario 1 model defined")


#     
# ### 4.2. Scenario 2: Feature-based Two-Tower MLP
# 

#  
class TwoTowerMLP(nn.Module):
    """
    Scenario 2: Feature-based two-tower architecture.
    Each tower processes its modality, then fusion layer combines them.
    """
    
    def __init__(
        self,
        apt_dim,
        prot_dim,
        tower_hidden_dims=[1024, 512, 256, 128],
        fusion_hidden_dims=[512, 256, 128, 64],
        dropout=0.3
    ):
        super().__init__()
        
        # Aptamer tower
        apt_layers = []
        prev_dim = apt_dim
        for dim in tower_hidden_dims:
            apt_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.apt_tower = nn.Sequential(*apt_layers)
        
        # Protein tower
        prot_layers = []
        prev_dim = prot_dim
        for dim in tower_hidden_dims:
            prot_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.prot_tower = nn.Sequential(*prot_layers)
        
        # Fusion layers
        fusion_dim = tower_hidden_dims[-1] * 2
        fusion_layers = []
        prev_dim = fusion_dim
        for dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, apt, prot):
        apt_emb = self.apt_tower(apt)
        prot_emb = self.prot_tower(prot)
        
        # Concatenate
        fused = torch.cat([apt_emb, prot_emb], dim=1)
        
        # Output
        logits = self.fusion(fused)
        return logits.squeeze(-1)


class TwoTowerWithTopHeads(nn.Module):
    """
    Two-tower architecture with sequence-based top heads (CNN, LSTM, Transformer).
    Adapts flat embeddings to sequences for top heads.
    Supports different top heads for aptamer and protein.
    """
    
    def __init__(
        self,
        apt_dim,
        prot_dim,
        apt_top_type="cnn",  # "identity", "cnn", "lstm", "transformer"
        prot_top_type="cnn",  # "identity", "cnn", "lstm", "transformer"
        seq_len=32,  # Length of pseudo-sequence
        hidden_dim=256,  # Hidden dimension for sequence processing
        output_dim=128,  # Output dimension from top head
        fusion_hidden_dims=[512, 256, 128, 64],
        dropout=0.3,
        apt_top_kwargs=None,  # Additional kwargs for aptamer top head
        prot_top_kwargs=None   # Additional kwargs for protein top head
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Project flat embeddings to sequences
        # We'll use a learnable projection to create sequences
        self.apt_seq_proj = nn.Sequential(
            nn.Linear(apt_dim, seq_len * hidden_dim),
            nn.LayerNorm(seq_len * hidden_dim)
        )
        self.prot_seq_proj = nn.Sequential(
            nn.Linear(prot_dim, seq_len * hidden_dim),
            nn.LayerNorm(seq_len * hidden_dim)
        )
        
        # Top heads
        top_classes = {
            "identity": IdentityTop,
            "cnn": CNNTop,
            "lstm": LSTMTop,
            "transformer": TransformerTop
        }
        
        if apt_top_type not in top_classes:
            raise ValueError(f"Unknown apt_top_type: {apt_top_type}. Must be one of {list(top_classes.keys())}")
        if prot_top_type not in top_classes:
            raise ValueError(f"Unknown prot_top_type: {prot_top_type}. Must be one of {list(top_classes.keys())}")
        
        # Filter kwargs to avoid conflicts with TwoTowerWithTopHeads params
        def filter_kwargs(kwargs_dict):
            if kwargs_dict is None:
                return {}
            return {k: v for k, v in kwargs_dict.items() 
                   if k not in ['seq_len', 'output_dim', 'fusion_hidden_dims', 'hidden_dim']}
        
        apt_kwargs = filter_kwargs(apt_top_kwargs)
        prot_kwargs = filter_kwargs(prot_top_kwargs)
        
        AptTopClass = top_classes[apt_top_type]
        ProtTopClass = top_classes[prot_top_type]
        
        self.apt_top = AptTopClass(hidden_dim, output_dim, dropout=dropout, **apt_kwargs)
        self.prot_top = ProtTopClass(hidden_dim, output_dim, dropout=dropout, **prot_kwargs)
        
        # Fusion layers
        fusion_dim = output_dim * 2
        fusion_layers = []
        prev_dim = fusion_dim
        for dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, apt, prot):
        # Project to sequences: (batch_size, feature_dim) -> (batch_size, seq_len, hidden_dim)
        apt_seq = self.apt_seq_proj(apt).view(-1, self.seq_len, self.hidden_dim)
        prot_seq = self.prot_seq_proj(prot).view(-1, self.seq_len, self.hidden_dim)
        
        # Apply top heads
        apt_emb = self.apt_top(apt_seq)
        prot_emb = self.prot_top(prot_seq)
        
        # Concatenate
        fused = torch.cat([apt_emb, prot_emb], dim=1)
        
        # Output
        logits = self.fusion(fused)
        return logits.squeeze(-1)


print("✓ Scenario 2 models defined (MLP and with Top Heads)")


#     
# ### 4.3. Scenario 3: Cross-Attention Fusion
# 

#  
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for fusing aptamer and protein representations.
    """
    
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, L_q, D] - e.g., aptamer features
            key_value: [B, L_kv, D] - e.g., protein features
        """
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        # Projections
        Q = self.q_proj(query).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, d]
        K = self.k_proj(key_value).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, d]
        V = self.v_proj(key_value).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, d]
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [B, H, L_q, L_kv]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, L_q, d]
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)  # [B, L_q, D]
        out = self.out_proj(out)
        
        # Residual + layer norm
        out = self.layer_norm(query + out)
        return out


class CrossAttentionModel(nn.Module):
    """
    Scenario 3: Cross-attention fusion model.
    Uses cross-attention to let aptamer and protein attend to each other.
    """
    
    def __init__(
        self,
        apt_dim,
        prot_dim,
        hidden_dim=512,
        n_attention_layers=4,
        n_heads=16,
        dropout=0.3,
        classifier_hidden_dims=[512, 256, 128]
    ):
        super().__init__()
        
        # Deep projection to common dimension
        apt_proj_layers = [
            nn.Linear(apt_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ]
        self.apt_proj = nn.Sequential(*apt_proj_layers)
        
        prot_proj_layers = [
            nn.Linear(prot_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ]
        self.prot_proj = nn.Sequential(*prot_proj_layers)
        
        # Multiple cross-attention layers
        self.attention_layers = nn.ModuleList([
            CrossAttentionFusion(hidden_dim, n_heads, dropout)
            for _ in range(n_attention_layers)
        ])
        
        # Deep classification head
        cls_layers = []
        prev_dim = hidden_dim * 2
        for dim in classifier_hidden_dims:
            cls_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        cls_layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*cls_layers)
    
    def forward(self, apt, prot):
        # Project to common dimension
        apt_emb = self.apt_proj(apt).unsqueeze(1)  # [B, 1, D]
        prot_emb = self.prot_proj(prot).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: aptamer attends to protein
        apt_attended = apt_emb
        for attn_layer in self.attention_layers:
            apt_attended = attn_layer(apt_attended, prot_emb)
        
        # Cross-attention: protein attends to aptamer
        prot_attended = prot_emb
        for attn_layer in self.attention_layers:
            prot_attended = attn_layer(prot_attended, apt_emb)
        
        # Concatenate and classify
        fused = torch.cat([apt_attended.squeeze(1), prot_attended.squeeze(1)], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)


print("✓ Scenario 3 model defined")



# ### 4.5. Model Visualization and Parameter Counting
# 

# Create sample models for visualization
# We'll use example dimensions based on OneHot + MorganFP
sample_apt_dim = 865  # OneHot: 216*4 + 1 = 865
sample_prot_dim = 2048  # MorganFP: 2048

# Create models with sample dimensions
models_to_visualize = [
    (PretrainedEncoderModel, "PretrainedEncoderModel", sample_apt_dim, sample_prot_dim, 
     {"hidden_dims": [512, 256, 128, 64], "dropout": 0.3}),
    (TwoTowerMLP, "TwoTowerMLP", sample_apt_dim, sample_prot_dim,
     {"tower_hidden_dims": [1024, 512, 256, 128], "fusion_hidden_dims": [512, 256, 128, 64], "dropout": 0.3}),
    (CrossAttentionModel, "CrossAttentionModel", sample_apt_dim, sample_prot_dim,
     {"hidden_dim": 512, "n_attention_layers": 4, "n_heads": 16, "dropout": 0.3})

]

print("Creating and visualizing model architectures...\n")
os.makedirs(os.path.join(ARTIFACTS_DIR, "figs"), exist_ok=True)

model_params_summary = []

for model_class, model_name, apt_dim, prot_dim, kwargs in models_to_visualize:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = model_class(apt_dim, prot_dim, **kwargs).to(device)
    
    # Count parameters
    params = count_parameters(model)
    model_params_summary.append({
        "Model": model_name,
        "Total Parameters": format_parameter_count(params['total']),
        "Trainable Parameters": format_parameter_count(params['trainable']),
        "Non-trainable Parameters": format_parameter_count(params['non_trainable']),
        "Total (raw)": params['total'],
        "Trainable (raw)": params['trainable']
    })
    
    print(f"Total parameters: {format_parameter_count(params['total'])}")
    print(f"Trainable parameters: {format_parameter_count(params['trainable'])}")
    
    # Visualize architecture
    save_path = os.path.join(ARTIFACTS_DIR, "figs", f"{model_name.lower()}_architecture.png")
    visualize_model_architecture(model, model_name, apt_dim, prot_dim, save_path=save_path)
    
    # Print model summary if torchinfo is available
    if TORCHINFO_AVAILABLE:
        print(f"\nDetailed model summary:")
        try:
            summary(model, input_size=[(1, apt_dim), (1, prot_dim)], device=device.type)
        except:
            print("  (Could not generate detailed summary)")

# Create summary table
params_df = pd.DataFrame(model_params_summary)
params_df = params_df.sort_values("Total (raw)", ascending=False)
print(f"\n{'='*60}")
print("Model Parameters Summary")
print(f"{'='*60}")
print(params_df[["Model", "Total Parameters", "Trainable Parameters"]].to_string(index=False))

# Save summary
params_summary_path = os.path.join(ARTIFACTS_DIR, "metrics", "model_parameters_summary.csv")
params_df.to_csv(params_summary_path, index=False)
print(f"\n✓ Parameters summary saved to: {params_summary_path}")


# ## 5. Training Utilities
# 


#  
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        apt = batch['apt'].to(device)
        prot = batch['prot'].to(device)
        label_class = batch['label_class'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(apt, prot)
        loss = criterion(logits, label_class.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_with_history(model, train_loader, val_loader, optimizer, criterion, device, 
                       n_epochs=50, patience=10):
    """
    Train model with full history tracking for visualization.
    
    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_mcc', 'val_roc_auc', etc.
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mcc': [],
        'val_roc_auc': [],
        'val_f1': []
    }
    
    best_val_score = -np.inf
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_preds_class = []
        all_scores = []
        all_labels_class = []
        
        with torch.no_grad():
            for batch in val_loader:
                apt = batch['apt'].to(device)
                prot = batch['prot'].to(device)
                label_class = batch['label_class'].to(device)

                logits = model(apt, prot)
                loss = criterion(logits, label_class.float())
                scores = torch.sigmoid(logits).cpu().numpy()
                
                val_loss += loss.item()
                n_val_batches += 1
                
                preds_class = (scores > 0.5).astype(int)
                all_preds_class.extend(preds_class)
                all_scores.extend(scores)
                all_labels_class.extend(label_class.cpu().numpy())
        
        val_loss = val_loss / max(n_val_batches, 1)
        val_metrics = compute_classification_metrics(
            np.array(all_labels_class),
            np.array(all_preds_class),
            np.array(all_scores)
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mcc'].append(val_metrics.get('MCC', np.nan))
        history['val_roc_auc'].append(val_metrics.get('ROC-AUC', np.nan))
        history['val_f1'].append(val_metrics.get('F1', np.nan))
        
        # Early stopping
        val_score = val_metrics.get('MCC', -np.inf)
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return history


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    all_preds_class = []
    all_scores = []
    all_labels_class = []
    
    with torch.no_grad():
        for batch in dataloader:
            apt = batch['apt'].to(device)
            prot = batch['prot'].to(device)
            label_class = batch['label_class'].cpu().numpy()
            

            logits = model(apt, prot)
            scores = torch.sigmoid(logits).cpu().numpy()
            preds_class = (scores > 0.5).astype(int)
            
            all_preds_class.extend(preds_class)
            all_scores.extend(scores)
            all_labels_class.extend(label_class)
    
    results = {
        'y_true': np.array(all_labels_class),
        'y_pred': np.array(all_preds_class),
        'y_scores': np.array(all_scores)
    }
    
    return results


print("✓ Training utilities defined")


# ## 6. Baseline Model Configurations
# 
# Load precomputed embeddings from notebooks/notebooks/data/embeddings
print("Loading precomputed embeddings from notebooks/notebooks/data/embeddings...")

# Define all encoder configurations (for loading)
apt_cfgs = [
    {"name": "OneHot"},
    {"name": "Kmer3"},
    {"name": "Kmer4"},
    {"name": "GENA"},
    {"name": "DNABERT2"},
]

prot_cfgs = [
    {"name": "ESMC"},
    {"name": "Prot_T5"},
    {"name": "Ankh"},

]

# Try to load precomputed embeddings
apt_features_map = {}
prot_features_map = {}
embeddings_dir = root / "notebooks" / "notebooks" / "data" / "embeddings"
print("embeddings_dir", embeddings_dir)
loaded_embeddings = load_embeddings(df, apt_cfgs, prot_cfgs, embeddings_dir=embeddings_dir)

if loaded_embeddings is not None:
    print(loaded_embeddings)
    apt_features_map, prot_features_map = loaded_embeddings
    print("\n✓ Successfully loaded precomputed embeddings!")
    print(f"  Aptamer encoders: {list(apt_features_map.keys())}")
    print(f"  Protein encoders: {list(prot_features_map.keys())}")
    for name, feat in apt_features_map.items():
        print(f"    {name}: shape {feat.shape}")
    for name, feat in prot_features_map.items():
        print(f"    {name}: shape {feat.shape}")

print(f"\n✓ Final: {len(apt_features_map)} aptamer encoders, {len(prot_features_map)} protein encoders")


# ## 7. Baseline Model Evaluation
# 

def run_baseline_evaluation(
    model_class,
    apt_features,
    prot_features,
    y_class,
    splits,
    model_kwargs=None,
    n_epochs=50,
    batch_size=64,
    lr=1e-3
):
    """
    Run baseline evaluation for a model configuration.
    
    Returns:
        List of metric dicts (one per fold)
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    apt_dim = apt_features.shape[1]
    prot_dim = prot_features.shape[1]
    
    all_fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{len(splits)}...", end=" ")
        
        # Split data
        apt_train, apt_val = apt_features[train_idx], apt_features[val_idx]
        prot_train, prot_val = prot_features[train_idx], prot_features[val_idx]
        y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
        
        # Normalize features
        scaler_apt = StandardScaler()
        scaler_prot = StandardScaler()
        apt_train = scaler_apt.fit_transform(apt_train)
        apt_val = scaler_apt.transform(apt_val)
        prot_train = scaler_prot.fit_transform(prot_train)
        prot_val = scaler_prot.transform(prot_val)
        
        # Create datasets
        train_dataset = AptamerProteinDataset(
            apt_train, prot_train, y_class_train
        )
        val_dataset = AptamerProteinDataset(
            apt_val, prot_val, y_class_val
        )
        
        # Use drop_last=True to avoid BatchNorm error with batch_size=1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = model_class(apt_dim, prot_dim, **model_kwargs).to(device)
        
        # Optimizer and loss
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_score = -np.inf
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device)
            
            # Validation
            val_results = evaluate(model, val_loader, device)
            val_metrics = compute_classification_metrics(
                val_results['y_true'],
                val_results['y_pred'],
                val_results['y_scores']
            )
            
            # Use MCC as validation score
            val_score = val_metrics['MCC']
            
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Final evaluation
        val_results = evaluate(model, val_loader, device)
        fold_metrics = compute_classification_metrics(
            val_results['y_true'],
            val_results['y_pred'],
            val_results['y_scores']
        )
        
        all_fold_metrics.append(fold_metrics)
        print(f"MCC: {fold_metrics['MCC']:.3f}, ROC-AUC: {fold_metrics['ROC-AUC']:.3f}")
    
    return all_fold_metrics


print("✓ Baseline evaluation function defined")


# Add TwoTower models with all combinations of top heads for aptamer and protein
top_head_types = ["identity", "cnn", "lstm", "transformer"]

# Base kwargs for all TwoTower models
base_two_tower_kwargs = {
    "seq_len": 32,
    "hidden_dim": 256,
    "output_dim": 128,
    "fusion_hidden_dims": [512, 256, 128, 64],
    "dropout": 0.3
}

# Top head specific kwargs
top_head_specific_kwargs = {
    "identity": {},
    "cnn": {"apt_top_kwargs": {"kernel_sizes": (3, 5, 7)}, 
            "prot_top_kwargs": {"kernel_sizes": (3, 5, 7)}},
    "lstm": {"apt_top_kwargs": {"num_layers": 2, "bidirectional": True},
             "prot_top_kwargs": {"num_layers": 2, "bidirectional": True}},
    "transformer": {"apt_top_kwargs": {"num_layers": 2, "num_heads": 8},
                    "prot_top_kwargs": {"num_layers": 2, "num_heads": 8}}
}

# Generate all combinations of aptamer and protein top heads
top_head_configs = []
for apt_top in top_head_types:
    for prot_top in top_head_types:
        config_name = f"TwoTower_{apt_top.capitalize()}_{prot_top.capitalize()}"
        
        # Base kwargs
        kwargs = base_two_tower_kwargs.copy()
        kwargs["apt_top_type"] = apt_top
        kwargs["prot_top_type"] = prot_top
        
        # Add specific kwargs for each top head type
        if apt_top in top_head_specific_kwargs and "apt_top_kwargs" in top_head_specific_kwargs[apt_top]:
            kwargs["apt_top_kwargs"] = top_head_specific_kwargs[apt_top]["apt_top_kwargs"]
        if prot_top in top_head_specific_kwargs and "prot_top_kwargs" in top_head_specific_kwargs[prot_top]:
            kwargs["prot_top_kwargs"] = top_head_specific_kwargs[prot_top]["prot_top_kwargs"]
        
        top_head_configs.append({
            "name": config_name,
            "class": TwoTowerWithTopHeads,
            "kwargs": kwargs,
        })

# Combine all model configurations
print(f"✓ Added {len(top_head_configs)} TwoTower models with all top head combinations")
print(f"  Combinations: {len(top_head_types)} aptamer × {len(top_head_types)} protein = {len(top_head_configs)}")


# ============================================================
# UTILITY FUNCTIONS FOR INCREMENTAL SAVING AND RESUME
# ============================================================

def load_existing_results(results_path, histories_path):
    """
    Load existing screening results and histories if they exist.
    
    Returns:
        (screening_results, all_histories, completed_configs)
    """
    screening_results = []
    all_histories = {}
    completed_configs = set()
    
    # Load results CSV
    if os.path.exists(results_path):
        try:
            existing_df = pd.read_csv(results_path)
            screening_results = existing_df.to_dict('records')
            completed_configs = {row['config'] for row in screening_results if 'config' in row}
            print(f"✓ Loaded {len(screening_results)} existing results from {results_path}")
        except Exception as e:
            print(f"⚠ Could not load existing results: {e}")
    
    # Load histories JSON
    if os.path.exists(histories_path):
        try:
            with open(histories_path, 'r') as f:
                all_histories = json.load(f)
                # Convert lists back to proper format
                for key, value in all_histories.items():
                    if isinstance(value, list):
                        all_histories[key] = value
            print(f"✓ Loaded {len(all_histories)} existing histories from {histories_path}")
        except Exception as e:
            print(f"⚠ Could not load existing histories: {e}")
    
    return screening_results, all_histories, completed_configs


def save_results_incremental(screening_results, all_histories, results_path, histories_path):
    """
    Save screening results and histories incrementally.
    
    Args:
        screening_results: List of result dictionaries
        all_histories: Dictionary of histories (will be updated with current split's histories)
        results_path: Path to save CSV results
        histories_path: Path to save JSON histories
    """
    try:
        # Save results CSV
        if len(screening_results) > 0:
            df = pd.DataFrame(screening_results)
            df.to_csv(results_path, index=False)
        
        # Save histories JSON (convert numpy arrays to lists for JSON serialization)
        histories_serializable = {}
        for key, value in all_histories.items():
            if isinstance(value, list):
                # Convert each fold history
                serialized_folds = []
                for fold_history in value:
                    serialized_fold = {}
                    for metric_name, metric_values in fold_history.items():
                        if isinstance(metric_values, list):
                            # Convert numpy arrays to lists
                            serialized_fold[metric_name] = [
                                float(v) if not np.isnan(v) else None 
                                for v in metric_values
                            ]
                        else:
                            serialized_fold[metric_name] = metric_values
                    serialized_folds.append(serialized_fold)
                histories_serializable[key] = serialized_folds
        
        with open(histories_path, 'w') as f:
            json.dump(histories_serializable, f, indent=2)
            
    except Exception as e:
        print(f"⚠ Error saving results: {e}")


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


print("✓ Utility functions for incremental saving and memory management defined")


# Updated run_baseline_evaluation with history tracking support
# This overrides the previous definition to add track_history parameter

def run_baseline_evaluation_with_history(
    model_class,
    apt_features,
    prot_features,
    y_class,
    splits,
    model_kwargs=None,
    n_epochs=50,
    batch_size=64,
    lr=1e-3,
    track_history=True,
    patience=15
):
    """
    Run baseline evaluation for a model configuration with training history tracking.
    
    Returns:
        all_fold_metrics: List of metric dicts (one per fold)
        all_fold_histories: List of training histories (one per fold) if track_history=True
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    apt_dim = apt_features.shape[1]
    prot_dim = prot_features.shape[1]
    
    all_fold_metrics = []
    all_fold_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold_idx + 1}/{len(splits)}...", end=" ")
        
        # Split data
        apt_train, apt_val = apt_features[train_idx], apt_features[val_idx]
        prot_train, prot_val = prot_features[train_idx], prot_features[val_idx]
        y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]

        
        # Normalize features
        scaler_apt = StandardScaler()
        scaler_prot = StandardScaler()
        apt_train = scaler_apt.fit_transform(apt_train)
        apt_val = scaler_apt.transform(apt_val)
        prot_train = scaler_prot.fit_transform(prot_train)
        prot_val = scaler_prot.transform(prot_val)
        
        # Create datasets
        train_dataset = AptamerProteinDataset(
            apt_train, prot_train, y_class_train
        )
        val_dataset = AptamerProteinDataset(
            apt_val, prot_val, y_class_val
        )
        
        # Use drop_last=True to avoid BatchNorm error with batch_size=1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = model_class(apt_dim, prot_dim, **model_kwargs).to(device)
        
        # Optimizer and loss
        optimizer = Adam(model.parameters(), lr=lr)

        criterion = nn.BCEWithLogitsLoss()
        
        # Training with history tracking
        if track_history:
            history = train_with_history(
                model, train_loader, val_loader, optimizer, criterion, device,
                n_epochs=n_epochs, patience=patience
            )
            all_fold_histories.append(history)
        else:
            # Training loop without history
            best_val_score = -np.inf
            patience_counter = 0
            
            for epoch in range(n_epochs):
                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                
                # Validation
                val_results = evaluate(model, val_loader, device)
                val_metrics = compute_classification_metrics(
                    val_results['y_true'],
                    val_results['y_pred'],
                    val_results['y_scores']
                )
                
                # Use MCC as validation score
                val_score = val_metrics['MCC']
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        # Final evaluation
        val_results = evaluate(model, val_loader, device)
        fold_metrics = compute_classification_metrics(
            val_results['y_true'],
            val_results['y_pred'],
            val_results['y_scores']
        )
        
        all_fold_metrics.append(fold_metrics)
        print(f"MCC: {fold_metrics['MCC']:.3f}")
    
    if track_history:
        return all_fold_metrics, all_fold_histories
    else:
        return all_fold_metrics

# Override the original function
run_baseline_evaluation = run_baseline_evaluation_with_history

print("✓ Updated run_baseline_evaluation with history tracking support")



# ### 7.1. Run Baseline Models

# ============================================================
# COMPREHENSIVE SCREENING WITH INCREMENTAL SAVING AND MULTIPLE SPLITS
# ============================================================

# Define split modes to evaluate
split_modes = ["stratified", "disjoint_aptamer", "disjoint_protein"]

# Select top encoders to reduce combinations (20-30 per split)

apt_encoders = ["Kmer3", "Kmer4", "OneHot", "GENA", "DNABERT2"]
  # Top 4
prot_encoders = ["Ankh", "ESMC", "Prot_T5"]

# Filter feature maps
apt_features_map = {k: v for k, v in apt_features_map.items() if k in apt_encoders}
prot_features_map = {k: v for k, v in prot_features_map.items() if k in prot_encoders}

print("apt_features_map", apt_features_map)
print("prot_features_map", prot_features_map)



# Select top models (7 best performing)
top_model_configs = [
    {
        "name": "TwoTowerMLP",
        "class": TwoTowerMLP,
        "kwargs": {
            "tower_hidden_dims": [1024, 512, 256, 128],
            "fusion_hidden_dims": [512, 256, 128, 64],
            "dropout": 0.3
        }
    },
    {
        "name": "CrossAttention",
        "class": CrossAttentionModel,
        "kwargs": {
            "hidden_dim": 512,
            "n_attention_layers": 4,
            "n_heads": 16,
            "dropout": 0.3,
            "classifier_hidden_dims": [512, 256, 128]
        }
    }

]

# Add best TwoTower top head combinations
top_two_tower_configs = [
    {
        "name": "TwoTower_CNN_CNN",
        "class": TwoTowerWithTopHeads,
        "kwargs": {
            "apt_top_type": "cnn",
            "prot_top_type": "cnn",
            "seq_len": 32,
            "hidden_dim": 256,
            "output_dim": 128,
            "fusion_hidden_dims": [512, 256, 128, 64],
            "dropout": 0.3,
            "apt_top_kwargs": {"kernel_sizes": (3, 5, 7)},
            "prot_top_kwargs": {"kernel_sizes": (3, 5, 7)}
        }
    },
    {
        "name": "TwoTower_LSTM_LSTM",
        "class": TwoTowerWithTopHeads,
        "kwargs": {
            "apt_top_type": "lstm",
            "prot_top_type": "lstm",
            "seq_len": 32,
            "hidden_dim": 256,
            "output_dim": 128,
            "fusion_hidden_dims": [512, 256, 128, 64],
            "dropout": 0.3,
            "apt_top_kwargs": {"num_layers": 2, "bidirectional": True},
            "prot_top_kwargs": {"num_layers": 2, "bidirectional": True}
        }
    },
    {
        "name": "TwoTower_Transformer_Transformer",
        "class": TwoTowerWithTopHeads,
        "kwargs": {
            "apt_top_type": "transformer",
            "prot_top_type": "transformer",
            "seq_len": 32,
            "hidden_dim": 256,
            "output_dim": 128,
            "fusion_hidden_dims": [512, 256, 128, 64],
            "dropout": 0.3,
            "apt_top_kwargs": {"num_layers": 2, "num_heads": 8},
            "prot_top_kwargs": {"num_layers": 2, "num_heads": 8}
        }
    }
]

model_configs = top_model_configs + top_two_tower_configs

print(f"Selected {len(model_configs)} models")
print(f"Selected {len(apt_features_map)} aptamer encoders: {list(apt_features_map.keys())}")
print(f"Selected {len(prot_features_map)} protein encoders: {list(prot_features_map.keys())}")
print(f"Total combinations per split: {len(model_configs) * len(apt_features_map) * len(prot_features_map)}")
print(f"Total splits: {len(split_modes)}")

# Training parameters
n_epochs = 50
patience = 15
batch_size = 64
lr = 1e-3

# Store all results across all splits
all_screening_results = []
all_histories = {}


# Iterate through all split modes
for split_mode in split_modes:
    print(f"\n{'='*80}")
    print(f"SPLIT MODE: {split_mode.upper()}")
    print(f"{'='*80}")
    
    # Load splits for this mode
    try:
        splits = load_splits_with_threshold(split_mode, base_dir=SPLITS_DIR)
        print(f"Loaded {len(splits)} folds for {split_mode} split")
    except Exception as e:
        print(f"⚠ Could not load splits for {split_mode}: {e}")
        continue
    
    # Define paths for incremental saving (per split mode)
    screening_results_path = os.path.join(ARTIFACTS_DIR, "metrics", f"comprehensive_screening_{split_mode}.csv")
    screening_histories_path = os.path.join(ARTIFACTS_DIR, "metrics", f"comprehensive_screening_{split_mode}_histories.json")
    
    # Load existing results if available (for resume capability)
    print(f"\nLoading existing results for {split_mode}...")
    screening_results, split_histories, completed_configs = load_existing_results(
        screening_results_path, screening_histories_path
    )
    
    # Add split_mode to completed configs to make them unique
    completed_configs = {f"{split_mode}::{config}" for config in completed_configs}
    
    if len(completed_configs) > 0:
        print(f"  Found {len(completed_configs)} already completed configurations for {split_mode}")
    
    # Update all_histories with split-specific histories
    for key, value in split_histories.items():
        all_histories[f"{split_mode}::{key}"] = value
    
    total_combinations = len(model_configs) * len(apt_features_map) * len(prot_features_map)
    current_combination = 0
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SCREENING: {split_mode.upper()}")
    print(f"{'='*80}")
    print(f"Models: {len(model_configs)}")
    print(f"Aptamer encoders: {len(apt_features_map)}")
    print(f"Protein encoders: {len(prot_features_map)}")
    print(f"Total combinations: {total_combinations}")
    print(f"Already completed: {len(completed_configs)}")
    print(f"Remaining: {total_combinations - len(completed_configs)}")
    print(f"{'='*80}\n")
    
    # Iterate through all model architectures
    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        model_class = model_cfg["class"]
        model_kwargs = model_cfg["kwargs"]
        
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}\n")
        
        # Iterate through all encoder combinations
        for apt_name, apt_feat in apt_features_map.items():
            for prot_name, prot_feat in prot_features_map.items():
                current_combination += 1
                config_name = f"{model_name} | {apt_name} + {prot_name}"
                unique_config_name = f"{split_mode}::{config_name}"
                
                # Skip if already completed
                if unique_config_name in completed_configs:
                    print(f"\n[{current_combination}/{total_combinations}] {config_name} - SKIPPED (already completed)")
                    continue
                
                print(f"\n[{current_combination}/{total_combinations}] {config_name}")
                print(f"{'='*80}")
                
                model = None  # Initialize for cleanup
                try:
                    # Verify feature shapes
                    print(f"  Aptamer features: {apt_feat.shape}, Protein features: {prot_feat.shape}")
                    
                    # Run evaluation with history tracking
                    metrics, histories = run_baseline_evaluation(
                        model_class,
                        apt_feat, prot_feat, y_class,
                        splits,
                        model_kwargs=model_kwargs,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        track_history=True,
                        patience=patience
                    )
                    
                    # Check if training actually ran (verify history length)
                    if histories and len(histories) > 0:
                        min_epochs = min(len(h['train_loss']) for h in histories)
                        max_epochs = max(len(h['train_loss']) for h in histories)
                        print(f"  Training epochs: {min_epochs}-{max_epochs} per fold")
                        
                        if min_epochs < 5:
                            print(f"  ⚠ WARNING: Training stopped very early (only {min_epochs} epochs)!")
                    
                    # Aggregate metrics
                    agg_metrics = aggregate_metrics(metrics)
                    
                    # Store results
                    result = {
                        "split_mode": split_mode,
                        "model": model_name,
                        "apt_encoder": apt_name,
                        "prot_encoder": prot_name,
                        "config": config_name,
                        **agg_metrics
                    }
                    screening_results.append(result)
                    all_screening_results.append(result)
                    
                    # Store histories for visualization
                    all_histories[unique_config_name] = histories
                    split_histories[config_name] = histories  # Also store in split-specific dict
                    
                    print(f"  ✓ MCC: {agg_metrics.get('MCC mean', np.nan):.3f} ± {agg_metrics.get('MCC std', np.nan):.3f}")
                    print(f"  ✓ ROC-AUC: {agg_metrics.get('ROC-AUC mean', np.nan):.3f} ± {agg_metrics.get('ROC-AUC std', np.nan):.3f}")
                    
                    # Save incrementally after each configuration
                    save_results_incremental(
                        screening_results, split_histories, 
                        screening_results_path, screening_histories_path
                    )
                    print(f"  💾 Saved results incrementally")
                    
                    # Clear memory to prevent overflow
                    try:
                        del model
                    except:
                        pass
                    clear_memory()
                    print(f"  🧹 Memory cleared")
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Save what we have so far even on error
                    try:
                        save_results_incremental(
                            screening_results, split_histories, 
                            screening_results_path, screening_histories_path
                        )
                        print(f"  💾 Saved partial results after error")
                    except:
                        pass
                    
                    # Clear memory after error
                    try:
                        del model
                    except:
                        pass
                    clear_memory()
                    continue
    
    # Final save for this split mode
    try:
        save_results_incremental(
            screening_results, split_histories, 
            screening_results_path, screening_histories_path
        )
        print(f"\n✓ Final results saved for {split_mode}")
    except Exception as e:
        print(f"⚠ Error saving final results for {split_mode}: {e}")

# Create combined results DataFrame
if len(all_screening_results) > 0:
    all_screening_df = pd.DataFrame(all_screening_results)
    all_screening_df = all_screening_df.sort_values("MCC mean", ascending=False)
    
    print("\n" + "="*80)
    print("COMBINED SCREENING RESULTS SUMMARY (ALL SPLITS)")
    print("="*80)
    print(all_screening_df[["split_mode", "model", "apt_encoder", "prot_encoder", "MCC mean", "MCC std", 
                            "ROC-AUC mean", "ROC-AUC std", "F1 mean"]].to_string(index=False))
    
    # Save combined results
    combined_path = os.path.join(ARTIFACTS_DIR, "metrics", "comprehensive_screening_all_splits.csv")
    all_screening_df.to_csv(combined_path, index=False)
    print(f"\n✓ Combined results saved to: {combined_path}")
else:
    print("\n⚠ No results to save!")
