"""Create a tiny fake MURA tree (same folder layout) to test the pipeline end to end.

Abnormal images get a bright blob so a model can actually learn something.
    python tests/make_fake_mura.py --out /tmp/fake_mura
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

REGIONS = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]


def make(out: Path, patients_per_region=(30, 8), seed=0):
    rng = random.Random(seed)
    pid = 0
    for split, n_pat in zip(["train", "valid"], patients_per_region):
        for region in REGIONS:
            for _ in range(n_pat):
                pid += 1
                for s in range(1, rng.randint(1, 2) + 1):
                    label = rng.random() < 0.4
                    d = out / "MURA-v1.1" / split / region / f"patient{pid:05d}" / \
                        f"study{s}_{'positive' if label else 'negative'}"
                    d.mkdir(parents=True, exist_ok=True)
                    for k in range(1, rng.randint(1, 3) + 1):
                        img = np.random.default_rng(pid * 10 + k).normal(80, 25, (96, 80))
                        if label:
                            img[30:60, 25:55] += 120
                        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "L").save(d / f"image{k}.png")
    # a macOS resource-fork file that the indexer must skip
    first_study = next((out / "MURA-v1.1" / "train").glob("XR_*/patient*/study*"))
    (first_study / "._image1.png").touch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    make(Path(ap.parse_args().out))
