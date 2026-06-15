#!/usr/bin/env python3
"""Create train/val/test split text files from a GT labels directory.

Usage
-----
    python tools/make_splits.py --data-root data/GT
    python tools/make_splits.py --data-root data/GT --train 0.8 --val 0.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create train/val/test splits for GTJsonDataset."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing labels/ sub-directory.",
    )
    p.add_argument(
        "--train", type=float, default=0.85, help="Training fraction (default: 0.85)."
    )
    p.add_argument(
        "--val", type=float, default=0.10, help="Validation fraction (default: 0.10)."
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train + args.val >= 1.0:
        raise ValueError("train + val must be < 1.0")

    labels_dir = args.data_root / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    ids = sorted(p.stem for p in labels_dir.glob("*.json"))
    print(f"Found {len(ids)} samples in {labels_dir}")

    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

    out_dir = args.data_root / "splits"
    out_dir.mkdir(exist_ok=True)
    for name, split_ids in splits.items():
        out_file = out_dir / f"{name}.txt"
        out_file.write_text("\n".join(split_ids) + "\n", encoding="utf-8")
        print(f"  {name:5s}: {len(split_ids):>6d} samples → {out_file}")


if __name__ == "__main__":
    main()
