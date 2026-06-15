"""Regression tests for Flip3DKeypoints — keeps 2D/3D left-right in sync.

Guards against the bug where mmpose's RandomFlip mirrors the 2D keypoints (and
swaps left/right via flip_indices) but leaves keypoints_3d untouched, so the
per-joint depth label is assigned to the wrong left/right joint for ~50% of
flipped training samples.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import pose24  # noqa: F401


@pytest.fixture(scope="module")
def flipped_sample(gt_data_root: str, val_split: str):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS, TRANSFORMS

    init_default_scope("mmpose")
    cfg = Config.fromfile("src/pose24/configs/rtmw3d_l_finetune_pose24.py")
    ds_cfg = dict(cfg.val_dataloader.dataset)
    ds_cfg.update(data_root=gt_data_root, ann_file=val_split)
    ds = DATASETS.build(ds_cfg)

    load = TRANSFORMS.build(dict(type="LoadImage"))
    gbb = TRANSFORMS.build(dict(type="GetBBoxCenterScale"))
    flip = TRANSFORMS.build(dict(type="RandomFlip", direction="horizontal", prob=1.0))
    flip3d = TRANSFORMS.build(dict(type="Flip3DKeypoints"))

    r = copy.deepcopy(ds.get_data_info(0))
    r = gbb(load(r))
    before = dict(kp2d=r["keypoints"][0].copy(), kp3d=r["keypoints_3d"][0].copy())
    r = flip3d(flip(r))
    after = dict(kp2d=r["keypoints"][0].copy(), kp3d=r["keypoints_3d"][0].copy())
    return before, after


# left_hip=11, right_hip=12 in POSE-24
LH, RH = 11, 12


def test_left_right_consistent_2d_3d(flipped_sample):
    _, after = flipped_sample
    sign_2d = after["kp2d"][LH, 0] < after["kp2d"][RH, 0]
    sign_3d = after["kp3d"][LH, 0] < after["kp3d"][RH, 0]
    assert sign_2d == sign_3d, "2D and 3D left/right ordering disagree after flip"


def test_depth_follows_swap(flipped_sample):
    before, after = flipped_sample
    # New left_hip should carry OLD right_hip's depth (z) after the L/R swap.
    assert np.isclose(after["kp3d"][LH, 2], before["kp3d"][RH, 2])
    assert np.isclose(after["kp3d"][RH, 2], before["kp3d"][LH, 2])


def test_camera_x_mirrored(flipped_sample):
    before, after = flipped_sample
    # New left_hip x = -(old right_hip x): mirror about the camera optical axis.
    assert np.isclose(after["kp3d"][LH, 0], -before["kp3d"][RH, 0])


def test_no_flip_leaves_3d_untouched():
    from mmpose.registry import TRANSFORMS

    flip3d = TRANSFORMS.build(dict(type="Flip3DKeypoints"))
    kp3d = np.random.randn(1, 24, 3).astype(np.float32)
    out = flip3d(dict(flip=False, keypoints_3d=kp3d.copy()))
    assert np.array_equal(out["keypoints_3d"], kp3d)
