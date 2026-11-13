"""Step 5: the losses you compare. All take raw logits, shape (N,).

- bce:  plain binary cross-entropy (baseline)
- wbce: weighted BCE with per-region weights, as in the MURA paper
        (Rajpurkar et al. 2017): for region T,
            w_abnormal = |normal_T| / |all_T|,  w_normal = |abnormal_T| / |all_T|
- focal: binary focal loss (Lin et al. 2017), implemented from scratch:
            FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def mura_region_weights(train_df: pd.DataFrame) -> pd.Series:
    """Per-image weights from per-region class frequencies (computed on STUDIES)."""
    studies = train_df.drop_duplicates("study_id")
    frac_abn = studies.groupby("region").label.mean()
    w_abn = train_df.region.map(1 - frac_abn)
    w_norm = train_df.region.map(frac_abn)
    return w_abn.where(train_df.label == 1, w_norm).astype(float)


class WeightedBCE(nn.Module):
    """BCE where each sample carries its own weight (passed in by the training loop)."""

    def forward(self, logits, targets, weights):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (weights * loss).sum() / weights.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1) or None")
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets, weights=None):
        # log(p_t) computed stably from logits; p_t = prob of the true class
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


class PlainBCE(nn.Module):
    def forward(self, logits, targets, weights=None):
        return F.binary_cross_entropy_with_logits(logits, targets)


def build_loss(name: str, gamma: float = 2.0, alpha=None) -> nn.Module:
    if name == "bce":
        return PlainBCE()
    if name == "wbce":
        return WeightedBCE()
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"unknown loss {name}")
