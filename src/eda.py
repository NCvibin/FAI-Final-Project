"""Step 1b: where is the imbalance? Produces the tables/figure for the paper.

Usage:
    python src/eda.py --images data/images.csv --out results/eda
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataset import resolve_data_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images.csv")
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--data-root", default=None, help="MURA-v1.1 folder (or set MURA_ROOT)")
    ap.add_argument("--size-sample", type=int, default=500,
                    help="How many images to open to estimate image sizes")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.images)
    train = df[df.official_split == "train"]
    studies = train.drop_duplicates("study_id")

    # Studies per region x label: the imbalance table for the paper.
    table = (studies.groupby(["region", "label"]).size().unstack(fill_value=0)
             .rename(columns={0: "normal", 1: "abnormal"}))
    table["total"] = table.sum(axis=1)
    table["abnormal_frac"] = (table.abnormal / table.total).round(3)
    table["share_of_studies"] = (table.total / table.total.sum()).round(3)
    table = table.sort_values("total", ascending=False)
    table.to_csv(out / "studies_by_region_label.csv")
    print("Training studies by region and label:\n", table.to_string(), "\n")

    ips = train.groupby("study_id").size()
    print("Images per study:", ips.describe().round(2).to_dict(), "\n")
    ips.value_counts().sort_index().to_csv(out / "images_per_study.csv", header=["studies"])

    sample = train.sample(min(args.size_sample, len(train)), random_state=0)
    root = resolve_data_root(args.data_root)
    sizes = pd.DataFrame([Image.open(root / p).size for p in sample.path], columns=["w", "h"])
    print("Image size (sampled):", sizes.describe().loc[["min", "50%", "max"]].to_dict(), "\n")

    # Figure: normal vs abnormal studies per region.
    fig, ax = plt.subplots(figsize=(7, 3.5))
    table[["normal", "abnormal"]].plot.barh(ax=ax, color=["#8aa1b4", "#c4553f"], width=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("Training studies")
    ax.set_ylabel("")
    ax.set_title("MURA training studies by region and label")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "studies_by_region_label.png", dpi=200)
    print(f"Saved tables and figure to {out}/")


if __name__ == "__main__":
    main()
