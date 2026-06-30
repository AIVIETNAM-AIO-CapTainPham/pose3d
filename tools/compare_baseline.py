"""Compare finetuned RTMPose3D-L (24-kp) vs the original pretrained RTMPose3D-L
(133-kp cocktail14) against the same GT (SAM-3D-Body), per-joint.

The original model has no anatomical-only joints (olecranon, cubital fossa,
acromion, neck) — those 7 of the 24 POSE keypoints are reported as NaN for it.
The remaining 17 map 1:1 to COCO-Wholebody indices 0-16 (same ordering).

Usage:
    PYTHONPATH=src uv run python tools/compare_baseline.py \
        --finetune-ckpt work_dirs/pose24_v4/best_MPJPE_epoch_222.pth \
        --finetune-config src/pose24/configs/rtmw3d_l_finetune_pose24_v4.py \
        --out work_dirs/pose24_v4/compare_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import pose24  # noqa: F401 - registers custom modules
from pose24.keypoints import NUM_KEYPOINTS, POSE24_KEYPOINT_NAMES

# POSE-24 indices 0-16 map 1:1 to COCO-Wholebody indices 0-16 (same body
# joint ordering convention). Indices 17-23 (olecranon/cubital fossa/
# acromion/neck) don't exist in COCO-Wholebody.
_PRETRAINED_AVAILABLE_INDICES = list(range(17))

_PRETRAINED_CKPT_URL = (
    "https://huggingface.co/rbarac/rtmpose3d/resolve/main/"
    "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)


def build_finetune_model(config_path: str, checkpoint: str, device: str):
    from mmengine.config import Config
    from mmpose.registry import MODELS

    cfg = Config.fromfile(config_path)
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(device).eval(), cfg


def build_pretrained_model(config_path: str, device: str):
    from mmengine.config import Config
    from mmpose.registry import MODELS
    from mmengine.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.model.backbone.init_cfg = None
    cfg.model.head.out_channels = 133
    model = MODELS.build(cfg.model)
    load_checkpoint(model, _PRETRAINED_CKPT_URL, map_location="cpu")
    return model.to(device).eval()


def run_inference(model, dataset, device: str, batch_size: int = 32):
    from mmengine.dataset import pseudo_collate

    all_pred = []  # (N, K, 3) camera-space coords
    all_gt = []  # (N, 24, 3)
    all_vis = []  # (N, 24) GT visibility/weight

    n = len(dataset)
    for start in range(0, n, batch_size):
        idxs = range(start, min(start + batch_size, n))
        batch = pseudo_collate([dataset[i] for i in idxs])
        with torch.no_grad():
            results = model.test_step(batch)
        for ds in results:
            gt_instances = ds.gt_instances
            pred_instances = ds.pred_instances
            all_pred.append(pred_instances.keypoints[0])  # (K, 3)
            all_gt.append(gt_instances.lifting_target[0])  # (24, 3)
            all_vis.append(gt_instances.lifting_target_visible[0])  # (24,)
        if (start // batch_size) % 10 == 0:
            print(f"  {start}/{n}")

    return (
        np.stack(all_pred),
        np.stack(all_gt),
        np.stack(all_vis),
    )


def root_center(coords: np.ndarray, root_index=(11, 12)) -> np.ndarray:
    root = coords[:, list(root_index)].mean(axis=1, keepdims=True)
    return coords - root


def per_joint_errors(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray) -> list[np.ndarray]:
    """Returns, per joint, the array of per-sample Euclidean errors (mm) where visible."""
    pred_c = root_center(pred)
    gt_c = root_center(gt)
    err = np.linalg.norm(pred_c - gt_c, axis=-1) * 1000  # (N, K) mm
    out = []
    for k in range(gt.shape[1]):
        mask = vis[:, k] > 0
        out.append(err[mask, k] if mask.any() else np.array([]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-ckpt", required=True)
    parser.add_argument(
        "--finetune-config",
        default="src/pose24/configs/rtmw3d_l_finetune_pose24_v4.py",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--out", default="work_dirs/compare_baseline")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples for a quick run")
    args = parser.parse_args()

    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS

    init_default_scope("mmpose")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Building finetuned (24-kp) model...")
    ft_model, cfg = build_finetune_model(args.finetune_config, args.finetune_ckpt, device)

    print("Building pretrained (133-kp) model...")
    pre_model = build_pretrained_model(args.finetune_config, device)

    ds_cfg = dict(cfg[f"{args.split}_dataloader"]["dataset"])
    if args.limit:
        ds_cfg["indices"] = args.limit
    dataset = DATASETS.build(ds_cfg)

    print(f"Running finetuned model on {len(dataset)} samples...")
    ft_pred, gt, vis = run_inference(ft_model, dataset, device)

    print(f"Running pretrained model on {len(dataset)} samples...")
    pre_pred_full, _, _ = run_inference(pre_model, dataset, device)
    # Pad pretrained's 17 available joints out to 24, NaN for the rest.
    pre_pred = np.full((pre_pred_full.shape[0], NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    pre_pred[:, _PRETRAINED_AVAILABLE_INDICES] = pre_pred_full[:, _PRETRAINED_AVAILABLE_INDICES]
    pre_vis = vis.copy()
    pre_vis[:, 17:] = 0  # mask out joints the pretrained model can't produce

    ft_errors = per_joint_errors(ft_pred, gt, vis)
    pre_errors = per_joint_errors(np.nan_to_num(pre_pred), gt, pre_vis)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-joint mean MPJPE bar chart ──────────────────────────────────────
    ft_means = [e.mean() if len(e) else np.nan for e in ft_errors]
    pre_means = [e.mean() if len(e) else np.nan for e in pre_errors]

    x = np.arange(NUM_KEYPOINTS)
    width = 0.35
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, pre_means, width, label="RTMPose3D-L gốc (pretrained, 133-kp)")
    ax.bar(x + width / 2, ft_means, width, label="Finetune hiện tại (24-kp)")
    ax.set_xticks(x)
    ax.set_xticklabels(POSE24_KEYPOINT_NAMES, rotation=60, ha="right")
    ax.set_ylabel("MPJPE per joint (mm)")
    ax.set_title(f"Per-joint MPJPE: gốc vs finetune (so với GT SAM-3D-Body, {args.split} split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_mpjpe_bar.png", dpi=150)
    print(f"Saved {out_dir / 'per_joint_mpjpe_bar.png'}")

    # ── Per-joint error distribution (boxplot), finetune only (full 24) ────
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.boxplot(ft_errors, tick_labels=POSE24_KEYPOINT_NAMES, showfliers=False)
    ax.set_xticklabels(POSE24_KEYPOINT_NAMES, rotation=60, ha="right")
    ax.set_ylabel("Error (mm)")
    ax.set_title(f"Per-joint error distribution — finetune model ({args.split} split)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_distribution_finetune.png", dpi=150)
    print(f"Saved {out_dir / 'per_joint_distribution_finetune.png'}")

    # ── Per-joint error distribution, pretrained (17 joints only) ──────────
    fig, ax = plt.subplots(figsize=(12, 6))
    names17 = [POSE24_KEYPOINT_NAMES[i] for i in _PRETRAINED_AVAILABLE_INDICES]
    errs17 = [pre_errors[i] for i in _PRETRAINED_AVAILABLE_INDICES]
    ax.boxplot(errs17, tick_labels=names17, showfliers=False)
    ax.set_xticklabels(names17, rotation=60, ha="right")
    ax.set_ylabel("Error (mm)")
    ax.set_title(f"Per-joint error distribution — pretrained gốc model ({args.split} split)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_distribution_pretrained.png", dpi=150)
    print(f"Saved {out_dir / 'per_joint_distribution_pretrained.png'}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\nPer-joint MPJPE (mm):")
    print(f"{'joint':<22}{'pretrained gốc':>16}{'finetune':>16}")
    for i, name in enumerate(POSE24_KEYPOINT_NAMES):
        pv = f"{pre_means[i]:.1f}" if not np.isnan(pre_means[i]) else "N/A"
        fv = f"{ft_means[i]:.1f}" if not np.isnan(ft_means[i]) else "N/A"
        print(f"{name:<22}{pv:>16}{fv:>16}")

    overall_ft = np.nanmean([e for e in ft_means if not np.isnan(e)])
    overall_pre17 = np.nanmean([pre_means[i] for i in _PRETRAINED_AVAILABLE_INDICES])
    print(f"\nOverall mean MPJPE (24 kp) finetune: {overall_ft:.1f}mm")
    print(f"Overall mean MPJPE (17 kp chung) pretrained gốc: {overall_pre17:.1f}mm")
    ft17 = np.nanmean([ft_means[i] for i in _PRETRAINED_AVAILABLE_INDICES])
    print(f"Overall mean MPJPE (17 kp chung) finetune: {ft17:.1f}mm")


if __name__ == "__main__":
    main()
