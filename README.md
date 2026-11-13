# CS 5100 Project: Bone X-ray Abnormality Classification (MURA)

Study-level binary classification (normal vs. abnormal) across all seven upper-limb
regions of MURA v1.1, with results reported per region.

- **Shared by the team:** data index, patient-level splits, data loader, study-level evaluation.
- **Algorithm 1 (Hardik):** imbalance-aware transfer learning (BCE → weighted BCE → focal loss), Grad-CAM analysis.
- **Algorithm 2 / 3:** teammates add their own `src/` modules and reuse the splits and `evaluate.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/          # loss sanity checks
```

## 1. Get the data

**Kaggle:** follow the token steps at the top of `scripts/download_mura.sh`, then

```bash
kaggle datasets list -s "MURA"                 # pick a ~3 GB copy with train/ and valid/
./scripts/download_mura.sh owner/dataset-name  # unzips into data/raw/
```

Kaggle copies are unofficial re-uploads. For the paper, cite the official release
(Rajpurkar et al. 2017), and check the copy has ~40,500 images
(about 36,800 train / 3,200 valid) — step 2 prints the counts.

## 2. Index, explore, split (run once; share `splits/` with the team)

```bash
python src/index_mura.py --root data/raw --out data/images.csv   # prints your MURA-v1.1 path
export MURA_ROOT=/full/path/to/MURA-v1.1                           # put this in ~/.zshrc
python src/eda.py        --images data/images.csv --out results/eda
python src/make_splits.py --images data/images.csv --out splits --seed 42
```

- `index_mura.py` reads labels from folder names (`study1_positive` = abnormal) and
  finds the MURA folder even if Kaggle nests it.
- `eda.py` writes the region × label table and figure (the "where is the imbalance" figure).
- Image paths in the CSVs are **relative to the MURA-v1.1 folder**, so the same `splits/`
  work on every teammate's machine: each person downloads MURA and sets their own `MURA_ROOT`
  (or passes `--data-root`).
- `make_splits.py`: official train → train/val **by patient** (~90/10, stratified by
  region × label); official valid → test. Commit or share these CSVs so everyone uses identical splits.

## 3. Train

```bash
# debug first: one region, small model
python src/train.py --regions XR_WRIST --model resnet18 --epochs 3 --out runs/debug

# baseline, then the loss sequence
python src/train.py --loss bce                 --out runs/densenet_bce
python src/train.py --loss wbce                --out runs/densenet_wbce
python src/train.py --loss focal --gamma 2     --out runs/densenet_focal_g2
python src/train.py --loss focal --gamma 2 --alpha 0.6 --out runs/densenet_focal_g2_a06
```

Each run folder holds `config.json`, `history.csv` (per-epoch val metrics),
`best.pt`, `test_image_preds.csv` and `test_metrics.json`.
Device is picked automatically (CUDA → Apple MPS → CPU). Debug runs need the
default ImageNet weights: from-scratch models on a few hundred images give unstable eval-mode predictions.

## 4. Evaluate (same for all three algorithms)

```bash
python src/evaluate.py --preds runs/densenet_focal_g2/test_image_preds.csv
```

Study probability = mean of its image probabilities. Reports AUROC, Cohen's kappa
(MURA paper baseline: 0.705), F1, accuracy and ECE, overall and per region.
Teammates: produce a CSV with `study_id, region, label, prob` per image and run the same script.

## Losses (`src/losses.py`)

| `--loss` | Definition |
|---|---|
| `bce` | Plain binary cross-entropy |
| `wbce` | Per-region weights as in the MURA paper: abnormal weight = fraction normal in that region, normal weight = fraction abnormal |
| `focal` | FL = −α_t (1 − p_t)^γ log p_t, written from scratch (`--gamma`, optional `--alpha`) |

## Testing without the real data

```bash
python tests/make_fake_mura.py --out /tmp/fake_mura
python src/index_mura.py --root /tmp/fake_mura --out /tmp/fm/images.csv
python src/make_splits.py --images /tmp/fm/images.csv --out /tmp/fm/splits
MURA_ROOT=/tmp/fake_mura/MURA-v1.1 python src/train.py --splits /tmp/fm/splits --model resnet18 --no-pretrained \
    --img-size 64 --epochs 3 --workers 0 --out /tmp/fm/run
```

## Next (step 6)
`src/models.py` already exposes `gradcam_target_layer()`; Grad-CAM, the deletion test,
the weight-randomization sanity check and the marker/shortcut analysis go in `src/gradcam.py`.
