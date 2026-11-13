"""Sanity checks for the hand-written losses.  Run: python -m pytest tests/ (from repo root)."""
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from losses import FocalLoss, WeightedBCE, mura_region_weights  # noqa: E402


def test_focal_gamma0_equals_bce():
    logits, y = torch.randn(64), torch.randint(0, 2, (64,)).float()
    assert torch.allclose(FocalLoss(gamma=0)(logits, y), F.binary_cross_entropy_with_logits(logits, y), atol=1e-6)


def test_focal_downweights_easy_examples():
    y = torch.ones(2)
    easy, hard = torch.tensor([5.0, 5.0]), torch.tensor([-1.0, -1.0])
    ratio_bce = (F.binary_cross_entropy_with_logits(easy, y) / F.binary_cross_entropy_with_logits(hard, y))
    ratio_fl = FocalLoss(gamma=2)(easy, y) / FocalLoss(gamma=2)(hard, y)
    assert ratio_fl < ratio_bce  # easy examples count relatively less under focal loss


def test_focal_matches_formula():
    logits, y = torch.tensor([0.3, -2.0, 1.5]), torch.tensor([1.0, 0.0, 0.0])
    p = torch.sigmoid(logits)
    p_t = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, torch.tensor(0.6), torch.tensor(0.4))
    expected = (-alpha_t * (1 - p_t) ** 2 * torch.log(p_t)).mean()
    assert torch.allclose(FocalLoss(gamma=2, alpha=0.6)(logits, y), expected, atol=1e-6)


def test_weighted_bce_unit_weights_equals_bce():
    logits, y = torch.randn(32), torch.randint(0, 2, (32,)).float()
    assert torch.allclose(WeightedBCE()(logits, y, torch.ones(32)),
                          F.binary_cross_entropy_with_logits(logits, y), atol=1e-6)


def test_mura_region_weights():
    # region A: 1 abnormal of 4 studies -> abnormal weight 0.75, normal weight 0.25
    df = pd.DataFrame({"region": ["A"] * 4, "study_id": list("abcd"), "label": [1, 0, 0, 0]})
    w = mura_region_weights(df)
    assert w.tolist() == [0.75, 0.25, 0.25, 0.25]
