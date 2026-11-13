"""Shared study-level evaluation: use this for ALL three algorithms.

Study probability = mean of its image probabilities (as in the MURA paper).
Metrics: AUROC, Cohen's kappa, F1, accuracy, ECE — overall and per region.

Usage on a saved predictions file:
    python src/evaluate.py --preds runs/baseline/test_image_preds.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


@torch.no_grad()
def predict_images(model, loader, device) -> pd.DataFrame:
    model.eval()
    probs, idxs = [], []
    for x, _, _, i in loader:
        logits = model(x.to(device, non_blocking=True)).squeeze(1)
        probs.append(torch.sigmoid(logits).float().cpu())
        idxs.append(i)
    df = loader.dataset.df.iloc[torch.cat(idxs).numpy()].copy()
    df["prob"] = torch.cat(probs).numpy()
    return df


def to_studies(image_preds: pd.DataFrame) -> pd.DataFrame:
    return (image_preds.groupby("study_id")
            .agg(region=("region", "first"), label=("label", "first"), prob=("prob", "mean"))
            .reset_index())


def ece(labels, probs, n_bins: int = 10) -> float:
    """Expected calibration error for the predicted probability of 'abnormal'."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "kappa": float(cohen_kappa_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "ece": ece(labels, probs),
    }
    out["auroc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    return out


def evaluate_studies(studies: pd.DataFrame, threshold: float = 0.5) -> dict:
    res = {"overall": metrics(studies.label, studies.prob, threshold), "per_region": {}}
    for region, g in studies.groupby("region"):
        res["per_region"][region] = metrics(g.label, g.prob, threshold)
    return res


def format_results(res: dict) -> str:
    rows = [{"region": "OVERALL", **res["overall"]}]
    rows += [{"region": r, **m} for r, m in res["per_region"].items()]
    return pd.DataFrame(rows).round(3).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with study_id, region, label, prob per image")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    res = evaluate_studies(to_studies(pd.read_csv(args.preds)), args.threshold)
    print(format_results(res))
    print(json.dumps(res["overall"], indent=2))


if __name__ == "__main__":
    main()
