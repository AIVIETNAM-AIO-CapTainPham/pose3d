#!/usr/bin/env python
"""Evaluate a checkpoint on val/test split (no training).

Usage
-----
    PYTHONPATH=src uv run python tools/eval.py \
        work_dirs/pose24/best_MPJPE_epoch_100.pth
"""

from __future__ import annotations

import argparse

import pose24  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--config", default="src/pose24/configs/rtmw3d_l_finetune_pose24.py")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--work-dir", default="work_dirs/pose24_eval")
    return p.parse_args()


def main():
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS, MODELS

    args = parse_args()
    init_default_scope("mmpose")
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    ds_cfg = cfg[f"{args.split}_dataloader"]["dataset"]
    dataset = DATASETS.build(ds_cfg)

    from mmengine.dataset import pseudo_collate
    from mmpose.evaluation.metrics import SimpleMPJPE

    mpjpe_metric = SimpleMPJPE(mode="mpjpe")
    pmpjpe_metric = SimpleMPJPE(mode="p-mpjpe")
    mpjpe_metric.dataset_meta = dataset.metainfo
    pmpjpe_metric.dataset_meta = dataset.metainfo

    print(f"Evaluating {len(dataset)} samples on {device}…")
    batch_size = 32
    for start in range(0, len(dataset), batch_size):
        batch = pseudo_collate(
            [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        )
        with torch.no_grad():
            results = model.test_step(batch)
        for ds in results:
            mpjpe_metric.process({}, [ds.to_dict()])
            pmpjpe_metric.process({}, [ds.to_dict()])
        if (start // batch_size) % 10 == 0:
            print(f"  {start}/{len(dataset)}")

    print("\nResults:")
    print(mpjpe_metric.evaluate(len(dataset)))
    print(pmpjpe_metric.evaluate(len(dataset)))


if __name__ == "__main__":
    main()
