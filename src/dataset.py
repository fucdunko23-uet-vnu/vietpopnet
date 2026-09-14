"""
VietPopNet — PyTorch Dataset & DataLoader
Phase 4: Full Training Pipeline
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class PopulationDataset(Dataset):
    """PyTorch Dataset cho VietPopNet microcensus data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature matrix (N, D) — numpy float32
            y: Target vector (N,)   — census_density (người/km²)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (N, 1) for regression output

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloaders(X_train, y_train, X_val, y_val,
                     batch_size: int = 32, num_workers: int = 0):
    """
    Tạo DataLoader chuẩn cho train và validation.

    Returns:
        train_loader, val_loader
    """
    train_ds = PopulationDataset(X_train, y_train)
    val_ds   = PopulationDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
