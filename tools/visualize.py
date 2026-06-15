#!/usr/bin/env python
"""Visualise POSE-24 keypoints: 2D overlay on image + 3D skeleton, GT vs Pred.

Examples
--------
GT only (fast, no model — sanity-check the labels)::

    PYTHONPATH=src python tools/visualize.py --num 4 --out vis/

GT vs prediction from a trained checkpoint::

    PYTHONPATH=src python tools/visualize.py \
        --checkpoint work_dirs/pose24/best_MPJPE_epoch_*.pth \
        --num 4 --out vis/

Each sample produces ``vis/sample_<id>.png`` with two panels: the image with
GT (green) + Pred (red) keypoints, and the 3D skeletons in the camera frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import pose24  # noqa: F401 — registers dataset / model / codec
from pose24.visualization import render_comparison

DEFAULT_CONFIG = "src/pose24/configs/rtmw3d_l_finetune_pose24.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Trained .pth. If omitted, only GT is rendered.",
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--num", type=int, default=4, help="Number of samples.")
    p.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit sample indices (overrides --num).",
    )
    p.add_argument("--out", default="vis", help="Output directory.")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_model(config: str, checkpoint: str, device: str):
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmpose.registry import MODELS

    init_default_scope("mmpose")
    cfg = Config.fromfile(config)
    cfg.model.backbone.init_cfg = None  # weights come from the checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device).eval()
    return model, device


def predict(model, dataset, idx: int, device: str):
    import torch
    from mmengine.dataset import pseudo_collate

    batch = pseudo_collate([dataset[idx]])
    with torch.no_grad():
        results = model.test_step(batch)
    pred = results[0].pred_instances
    pred_3d = np.asarray(pred.keypoints[0])  # (24, 3) camera m
    pred_2d = np.asarray(pred.transformed_keypoints[0])  # (24, 2) image px
    return pred_2d, pred_3d


def main() -> None:
    args = parse_args()
    from mmengine.config import Config
    from mmengine.registry import init_default_scope, DATASETS

    init_default_scope("mmpose")
    cfg = Config.fromfile(args.config)
    dataset = DATASETS.build(cfg[f"{args.split}_dataloader"].dataset)

    indices = (
        args.indices
        if args.indices is not None
        else list(range(min(args.num, len(dataset))))
    )

    model = device = None
    if args.checkpoint:
        model, device = build_model(args.config, args.checkpoint, args.device)
        print(f"loaded checkpoint: {args.checkpoint} (device={device})")
    else:
        print("no checkpoint → rendering GT only")

    out_dir = Path(args.out)
    for idx in indices:
        info = dataset.get_data_info(idx)
        gt_2d = np.asarray(info["keypoints"][0])  # (24, 2) full-image px
        gt_3d = np.asarray(info["keypoints_3d"][0])  # (24, 3) camera m
        vis = np.asarray(info.get("keypoints_visible", np.ones((1, 24)))[0]) > 0.5
        image = load_image_rgb(info["img_path"])

        pred_2d = pred_3d = None
        title = f"sample {idx}"
        if model is not None:
            pred_2d, pred_3d = predict(model, dataset, idx, device)
            mpjpe = np.linalg.norm(pred_3d - gt_3d, axis=-1).mean() * 1000
            title += f"  |  MPJPE = {mpjpe:.1f} mm"

        out_path = out_dir / f"sample_{idx}.png"
        render_comparison(
            image,
            gt_2d,
            gt_3d,
            pred_2d,
            pred_3d,
            visible=vis,
            title=title,
            out_path=str(out_path),
        )
        print(f"  wrote {out_path}")

    print(f"done → {out_dir}/")


if __name__ == "__main__":
    main()
