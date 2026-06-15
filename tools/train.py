#!/usr/bin/env python
"""Train / finetune entry point (no `mim` dependency).

Wraps mmengine's Runner and imports ``pose24`` so the custom dataset, codec,
estimator, head, losses and visualisation hook are registered.

Examples
--------
    PYTHONPATH=src python tools/train.py \
        src/pose24/configs/rtmw3d_l_finetune_pose24.py \
        --work-dir work_dirs/pose24

    # quick smoke run with config overrides:
    PYTHONPATH=src python tools/train.py <cfg> \
        --cfg-options train_cfg.max_epochs=1
"""

from __future__ import annotations

import argparse

import pose24  # noqa: F401 — registers all custom modules

# PyTorch ≥ 2.6 defaults torch.load(weights_only=True), which rejects the numpy
# objects mmengine stores in checkpoints (optimizer state, message_hub, meta),
# so `--resume` fails with an UnpicklingError. These are our own trusted
# checkpoints, so force weights_only=False for the whole train process.
import torch  # noqa: E402

_orig_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune RTMPose3D / POSE-24")
    p.add_argument("config")
    p.add_argument("--work-dir", default="work_dirs/pose24")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="Override config, e.g. train_cfg.max_epochs=1",
    )
    p.add_argument(
        "--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"]
    )
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args()


def _parse_cfg_options(pairs):
    from mmengine.config import DictAction

    if not pairs:
        return None
    action = DictAction(option_strings=[], dest="x")
    ns = argparse.Namespace()
    action(None, ns, pairs)
    return ns.x


def main() -> None:
    from mmengine.config import Config
    from mmengine.runner import Runner

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher

    overrides = _parse_cfg_options(args.cfg_options)
    if overrides:
        cfg.merge_from_dict(overrides)

    if args.amp and cfg.optim_wrapper.get("type", "OptimWrapper") == "OptimWrapper":
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    if args.resume:
        cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
