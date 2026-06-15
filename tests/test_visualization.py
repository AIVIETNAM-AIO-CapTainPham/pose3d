"""Unit tests for the pose visualisation helpers (no model needed)."""

from __future__ import annotations

import numpy as np
import pytest

import pose24  # noqa: F401
from pose24.keypoints import NUM_KEYPOINTS
from pose24.visualization import render_comparison


@pytest.fixture
def dummy():
    rng = np.random.default_rng(0)
    img = (rng.random((128, 96, 3)) * 255).astype(np.uint8)
    gt_2d = rng.random((NUM_KEYPOINTS, 2)) * [96, 128]
    gt_3d = rng.random((NUM_KEYPOINTS, 3))
    return img, gt_2d, gt_3d


class TestRenderComparison:
    def test_gt_only_writes_png(self, dummy, tmp_path) -> None:
        img, gt_2d, gt_3d = dummy
        out = tmp_path / "gt.png"
        fig = render_comparison(img, gt_2d, gt_3d, out_path=str(out))
        assert out.exists() and out.stat().st_size > 0
        assert len(fig.axes) == 2  # 2D + 3D panels

    def test_gt_vs_pred_writes_png(self, dummy, tmp_path) -> None:
        img, gt_2d, gt_3d = dummy
        rng = np.random.default_rng(1)
        pred_2d = gt_2d + rng.normal(0, 5, gt_2d.shape)
        pred_3d = gt_3d + rng.normal(0, 0.1, gt_3d.shape)
        out = tmp_path / "cmp.png"
        render_comparison(
            img, gt_2d, gt_3d, pred_2d, pred_3d, title="t", out_path=str(out)
        )
        assert out.exists() and out.stat().st_size > 0

    def test_rejects_wrong_shape(self, dummy) -> None:
        img, gt_2d, gt_3d = dummy
        with pytest.raises(ValueError):
            render_comparison(img, gt_2d[:10], gt_3d)


class TestVisualizationHook:
    def test_registered(self) -> None:
        from mmpose.registry import HOOKS

        assert "Pose3DVisualizationHook" in HOOKS._module_dict

    def test_builds_with_args(self) -> None:
        from mmpose.registry import HOOKS

        hook = HOOKS.build(
            dict(type="Pose3DVisualizationHook", num_samples=2, interval=5)
        )
        assert hook.num_samples == 2 and hook.interval == 5
