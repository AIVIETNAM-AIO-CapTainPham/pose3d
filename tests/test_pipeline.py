"""Integration test: full mmpose pipeline through GTJsonDataset."""

from __future__ import annotations

import pytest

import pose24  # noqa: F401
from pose24.datasets import GTJsonDataset


@pytest.fixture(scope="module")
def val_dataset_with_pipeline(val_split: str, gt_data_root: str) -> GTJsonDataset:
    from mmengine.config import Config
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    cfg = Config.fromfile("src/pose24/configs/rtmw3d_l_finetune_pose24.py")
    return GTJsonDataset(
        data_root=gt_data_root,
        ann_file=val_split,
        pipeline=cfg.val_pipeline,
        test_mode=True,
    )


class TestFullPipeline:
    def test_sample_has_inputs(self, val_dataset_with_pipeline: GTJsonDataset) -> None:
        import torch

        sample = val_dataset_with_pipeline[0]
        assert "inputs" in sample
        assert isinstance(sample["inputs"], torch.Tensor)
        assert sample["inputs"].shape == (3, 384, 288)

    def test_sample_has_data_samples(
        self, val_dataset_with_pipeline: GTJsonDataset
    ) -> None:
        sample = val_dataset_with_pipeline[0]
        assert "data_samples" in sample

    def test_z_labels_present(self, val_dataset_with_pipeline: GTJsonDataset) -> None:
        sample = val_dataset_with_pipeline[0]
        lbl = sample["data_samples"].gt_instance_labels
        assert hasattr(lbl, "keypoint_z_labels")
        assert hasattr(lbl, "with_z_label")
        assert lbl.with_z_label[0], "3D labels not generated"

    def test_z_label_shape(self, val_dataset_with_pipeline: GTJsonDataset) -> None:
        from pose24.keypoints import NUM_KEYPOINTS

        sample = val_dataset_with_pipeline[0]
        lbl = sample["data_samples"].gt_instance_labels
        # shape: (1, K, D*split_ratio) = (1, 24, 576)
        assert lbl.keypoint_z_labels.shape == (1, NUM_KEYPOINTS, 576)
