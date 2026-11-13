"""Step 4/5: train a binary normal/abnormal classifier with a chosen loss.

Examples:
    # debug: wrist only, small fast model
    python src/train.py --regions XR_WRIST --model resnet18 --loss bce --epochs 3 --out runs/debug

    # baseline (all regions)
    python src/train.py --model densenet169 --loss bce --out runs/densenet_bce

    # your algorithm
    python src/train.py --loss wbce --out runs/densenet_wbce
    python src/train.py --loss focal --gamma 2 --out runs/densenet_focal_g2
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MuraDataset, build_transforms
from evaluate import evaluate_studies, format_results, predict_images, to_studies
from losses import build_loss, mura_region_weights
from models import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--data-root", default=None, help="MURA-v1.1 folder (or set MURA_ROOT)")
    ap.add_argument("--regions", nargs="*", default=None, help="e.g. XR_WRIST XR_ELBOW (default: all)")
    ap.add_argument("--model", default="densenet169", choices=["densenet169", "resnet18", "resnet50"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--loss", default="bce", choices=["bce", "wbce", "focal"])
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss focusing parameter")
    ap.add_argument("--alpha", type=float, default=None, help="focal loss class weight for abnormal (0-1)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="early stopping on val AUROC")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="use only N train/val images (debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"device={device}  run={out}")

    sp = Path(args.splits)
    train_ds = MuraDataset(sp / "train.csv", build_transforms(args.img_size, True), args.regions, args.data_root)
    val_ds = MuraDataset(sp / "val.csv", build_transforms(args.img_size, False), args.regions, args.data_root)
    test_ds = MuraDataset(sp / "test.csv", build_transforms(args.img_size, False), args.regions, args.data_root)
    if args.limit:
        for ds in (train_ds, val_ds):
            ds.df = ds.df.sample(min(args.limit, len(ds.df)), random_state=args.seed).reset_index(drop=True)
            ds.weights = torch.ones(len(ds.df))
    if args.loss == "wbce":
        train_ds.set_weights(mura_region_weights(train_ds.df).values)
    print(f"images: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    pin = device.type == "cuda"
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                        num_workers=args.workers, pin_memory=pin)
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = build_loss(args.loss, args.gamma, args.alpha)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # As in the MURA paper: drop LR by 10x when the validation metric plateaus.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score, bad_epochs, history = (-1.0, -1.0), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y, w, _ in train_dl:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y, w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            n += len(y)

        val_res = evaluate_studies(to_studies(predict_images(model, val_dl, device)))["overall"]
        sched.step(val_res["auroc"])
        rec = {"epoch": epoch, "train_loss": running / n, "lr": opt.param_groups[0]["lr"],
               "secs": round(time.time() - t0), **{f"val_{k}": v for k, v in val_res.items()}}
        history.append(rec)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"epoch {epoch:2d}  loss {rec['train_loss']:.4f}  val AUROC {val_res['auroc']:.3f}  "
              f"kappa {val_res['kappa']:.3f}  ({rec['secs']}s)")

        # Select on val AUROC; break ties with kappa (AUROC alone can saturate).
        score = (round(val_res["auroc"], 4), val_res["kappa"])
        if score > best_score:
            best_score, bad_epochs = score, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    # Final: best checkpoint on the held-out test set (official MURA valid).
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    test_preds = predict_images(model, test_dl, device)
    test_preds.to_csv(out / "test_image_preds.csv", index=False)
    res = evaluate_studies(to_studies(test_preds))
    (out / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print("\nTEST (study level):\n" + format_results(res))


if __name__ == "__main__":
    main()
