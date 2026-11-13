"""Step 2: shared patient-level splits for the whole team.

- train / val: the official MURA training set, split BY PATIENT (~90/10), keeping
  the region x label mix similar in both (StratifiedGroupKFold).
- test: the official MURA validation set (the real MURA test set is hidden).

Everyone on the team trains and evaluates on these exact CSVs.

Usage:
    python src/make_splits.py --images data/images.csv --out splits/ --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42):
    train_all = df[df.official_split == "train"].reset_index(drop=True)
    test = df[df.official_split == "valid"].reset_index(drop=True)

    n_splits = round(1 / val_fraction)
    strata = train_all.region + "_" + train_all.label.astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(train_all, strata, groups=train_all.patient))
    train, val = train_all.iloc[train_idx], train_all.iloc[val_idx]

    # Hard guarantees: no patient appears in two splits.
    assert not set(train.patient) & set(val.patient), "patient leak train/val"
    leak = (set(train.patient) | set(val.patient)) & set(test.patient)
    if leak:
        print(f"WARNING: {len(leak)} patients appear in both official train and valid; "
              "dropping their images from train/val.")
        train = train[~train.patient.isin(leak)]
        val = val[~val.patient.isin(leak)]
    return train.assign(split="train"), val.assign(split="val"), test.assign(split="test")


def summary(train, val, test) -> pd.DataFrame:
    parts = []
    for d in (train, val, test):
        studies = d.drop_duplicates("study_id")
        parts.append(pd.DataFrame({
            "split": d.split.iloc[0],
            "images": [len(d)],
            "studies": [len(studies)],
            "patients": [d.patient.nunique()],
            "abnormal_study_frac": [round(studies.label.mean(), 3)],
        }))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.images)
    train, val, test = make_splits(df, args.val_fraction, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in (train, val, test):
        d.to_csv(out / f"{d.split.iloc[0]}.csv", index=False)
    print(summary(train, val, test).to_string(index=False))
    print(f"Wrote {out}/train.csv, val.csv, test.csv  (seed={args.seed})")


if __name__ == "__main__":
    main()
