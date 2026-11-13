"""Step 1: build one table with a row per MURA image.

MURA encodes everything in its folder names:
    MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png
so we walk the tree instead of trusting any CSV (Kaggle copies sometimes
drop or rename the CSVs).

Usage:
    python src/index_mura.py --root /path/to/download --out data/images.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

REGIONS = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
           "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]
STUDY_RE = re.compile(r"^study\d+_(positive|negative)$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def find_mura_root(root: Path) -> Path:
    """Return the folder that holds train/ and valid/ (works for nested Kaggle unzips)."""
    for cand in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if (cand / "train").is_dir() and (cand / "valid").is_dir():
            if any((cand / "train" / r).is_dir() for r in REGIONS):
                return cand
    raise FileNotFoundError(
        f"No folder with train/ and valid/ containing XR_* regions under {root}")


def index_images(mura_root: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "valid"]:
        for region in REGIONS:
            region_dir = mura_root / split / region
            if not region_dir.is_dir():
                continue
            for patient_dir in sorted(region_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                for study_dir in sorted(patient_dir.iterdir()):
                    m = STUDY_RE.match(study_dir.name)
                    if not (study_dir.is_dir() and m):
                        continue
                    label = 1 if m.group(1) == "positive" else 0
                    for img in sorted(study_dir.iterdir()):
                        # macOS zips can contain ._image1.png resource forks
                        if img.suffix.lower() not in IMAGE_EXTS or img.name.startswith("."):
                            continue
                        rows.append({
                            # relative to the MURA root, so split CSVs work on every teammate's machine
                            "path": img.relative_to(mura_root).as_posix(),
                            "official_split": split,
                            "region": region,
                            "patient": patient_dir.name,
                            "study_id": f"{split}/{region}/{patient_dir.name}/{study_dir.name}",
                            "label": label,
                        })
    if not rows:
        raise RuntimeError(f"Found no images under {mura_root}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder where MURA was downloaded/unzipped")
    ap.add_argument("--out", default="data/images.csv")
    args = ap.parse_args()

    mura_root = find_mura_root(Path(args.root).expanduser())
    df = index_images(mura_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"MURA root: {mura_root}")
    print(f"Pass this to the other scripts:  --data-root {mura_root}  (or export MURA_ROOT={mura_root})")
    print(f"{len(df):,} images, {df.study_id.nunique():,} studies, "
          f"{df.patient.nunique():,} patients -> {out}")
    print(df.groupby("official_split").agg(images=("path", "size"),
                                           studies=("study_id", "nunique")))


if __name__ == "__main__":
    main()
