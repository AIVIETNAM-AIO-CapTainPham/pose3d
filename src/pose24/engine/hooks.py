"""Training hook that dumps GT-vs-Pred comparison images during validation.

After every ``interval`` validation epochs it renders a few val samples
(2D overlay + 3D skeleton, GT green / Pred red) into
``<work_dir>/vis/epoch_<N>/`` so a trainer can watch predictions converge onto
ground truth and spot systematic skew early.

Failures are swallowed (logged, never raised) so visualisation can never crash
a training run.
"""

from __future__ import annotations

import os.path as osp
from typing import Optional, Sequence

import numpy as np

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmpose.registry import HOOKS


@HOOKS.register_module()
class Pose3DVisualizationHook(Hook):
    """Dump GT-vs-Pred comparison images after validation epochs.

    Args:
        num_samples: How many val samples to render.
        interval: Render every ``interval`` validation epochs.
        indices: Explicit sample indices (overrides ``num_samples``).
        out_subdir: Sub-directory under ``work_dir`` for the images.
    """

    priority = "LOWEST"

    def __init__(
        self,
        num_samples: int = 4,
        interval: int = 1,
        indices: Optional[Sequence[int]] = None,
        out_subdir: str = "vis",
    ) -> None:
        self.num_samples = num_samples
        self.interval = interval
        self.indices = list(indices) if indices is not None else None
        self.out_subdir = out_subdir
        self._val_epoch = 0

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        self._val_epoch += 1
        if self._val_epoch % self.interval != 0:
            return
        try:
            self._render(runner)
        except Exception as exc:  # never crash training over a plot
            runner.logger.warning(f"[Pose3DVisualizationHook] skipped: {exc}")

    def _render(self, runner) -> None:
        import cv2
        import torch
        from mmengine.dataset import pseudo_collate
        from pose24.visualization import render_comparison

        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        dataset = runner.val_dataloader.dataset
        indices = (
            self.indices
            if self.indices is not None
            else list(range(min(self.num_samples, len(dataset))))
        )

        out_dir = osp.join(runner.work_dir, self.out_subdir, f"epoch_{runner.epoch}")
        was_training = model.training
        model.eval()
        for idx in indices:
            info = dataset.get_data_info(idx)
            gt_2d = np.asarray(info["keypoints"][0])
            gt_3d = np.asarray(info["keypoints_3d"][0])
            vis = np.asarray(info.get("keypoints_visible", np.ones((1, 24)))[0]) > 0.5
            image = cv2.cvtColor(cv2.imread(info["img_path"]), cv2.COLOR_BGR2RGB)

            batch = pseudo_collate([dataset[idx]])
            with torch.no_grad():
                res = model.test_step(batch)
            pred = res[0].pred_instances
            pred_2d = np.asarray(pred.transformed_keypoints[0])
            pred_3d = np.asarray(pred.keypoints[0])

            # Root-centre both skeletons so pose shapes overlap for comparison.
            root_idx = [11, 12]  # left_hip, right_hip
            gt_root = gt_3d[root_idx].mean(0)
            pred_root = pred_3d[root_idx].mean(0)
            gt_3d_rel = gt_3d - gt_root
            pred_3d_rel = pred_3d - pred_root
            mpjpe = (
                np.linalg.norm(pred_3d_rel[vis] - gt_3d_rel[vis], axis=-1).mean() * 1000
            )

            render_comparison(
                image,
                gt_2d,
                gt_3d_rel,
                pred_2d,
                pred_3d_rel,
                visible=vis,
                title=f"epoch {runner.epoch} | sample {idx} | " f"MPJPE={mpjpe:.1f} mm",
                out_path=osp.join(out_dir, f"sample_{idx}.png"),
            )
        if was_training:
            model.train()
        runner.logger.info(
            f"[Pose3DVisualizationHook] wrote {len(indices)} images → {out_dir}"
        )
