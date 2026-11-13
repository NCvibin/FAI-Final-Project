#!/usr/bin/env bash
# Download a MURA copy from Kaggle and unzip it into data/raw/.
#
# One-time setup:
#   1. kaggle.com -> Settings -> API -> "Create New Token" (downloads kaggle.json)
#   2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   3. pip install kaggle
#   4. Find the dataset slug:  kaggle datasets list -s "MURA"
#      (pick one with ~3 GB and both train/ and valid/ folders; copy its "owner/name")
#
# Usage:
#   ./scripts/download_mura.sh owner/dataset-name [dest]
set -euo pipefail

SLUG="${1:?Usage: $0 owner/dataset-name [dest]   (find it with: kaggle datasets list -s MURA)}"
DEST="${2:-data/raw}"

command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle"; exit 1; }
[ -f "$HOME/.kaggle/kaggle.json" ] || [ -n "${KAGGLE_KEY:-}" ] || {
  echo "No Kaggle API token: see the setup steps at the top of this script"; exit 1; }

mkdir -p "$DEST"
echo "Downloading $SLUG -> $DEST (about 3 GB)..."
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip
echo "Done. Next: python src/index_mura.py --root $DEST"
