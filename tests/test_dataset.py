"""Unit tests for GTJsonDataset."""

from __future__ import annotations

import numpy as np
import pytest

import pose24  # noqa: F401 — registers GTJsonDataset
from pose24.datasets import GTJsonDataset
from pose24.keypoints import NUM_KEYPOINTS


class TestGTJsonDatasetSamples:
    """Tests that run against the real GT data (integration)."""

    @pytest.fixture(scope="class")
    def dataset(self, val_split: str, gt_data_root: str) -> GTJsonDataset:
        return GTJsonDataset(
            data_root=gt_data_root,
            ann_file=val_split,
            pipeline=[],
        )

    def test_length_positive(self, dataset: GTJsonDataset) -> None:
        assert len(dataset) > 0

    def test_sample_keys(self, dataset: GTJsonDataset) -> None:
        required = {
            "img_path",
            "bbox",
            "bbox_score",
            "keypoints",
            "keypoints_visible",
            "keypoints_3d",
        }
        sample = dataset[0]
        assert required.issubset(sample.keys())

    def test_bbox_shape(self, dataset: GTJsonDataset) -> None:
        bbox = dataset[0]["bbox"]
        assert bbox.shape == (1, 4)
        assert bbox.dtype == np.float32
        x1, y1, x2, y2 = bbox[0]
        assert x2 > x1 and y2 > y1

    def test_keypoints_shape(self, dataset: GTJsonDataset) -> None:
        kpts = dataset[0]["keypoints"]
        assert kpts.shape == (1, NUM_KEYPOINTS, 2)
        assert kpts.dtype == np.float32

    def test_keypoints_3d_shape(self, dataset: GTJsonDataset) -> None:
        kpts3d = dataset[0]["keypoints_3d"]
        assert kpts3d.shape == (1, NUM_KEYPOINTS, 3)
        assert kpts3d.dtype == np.float32

    def test_3d_nose_above_hip(self, dataset: GTJsonDataset) -> None:
        """In y_down camera frame nose.Y < hip.Y (nose is higher)."""
        for idx in range(min(20, len(dataset))):
            kpts3d = dataset[idx]["keypoints_3d"][0]  # (24, 3)
            nose_y = kpts3d[0, 1]
            l_hip_y = kpts3d[11, 1]
            r_hip_y = kpts3d[12, 1]
            assert (
                nose_y < l_hip_y
            ), f"sample {idx}: nose Y={nose_y:.3f} >= left_hip Y={l_hip_y:.3f}"
            assert (
                nose_y < r_hip_y
            ), f"sample {idx}: nose Y={nose_y:.3f} >= right_hip Y={r_hip_y:.3f}"

    def test_img_path_exists(self, dataset: GTJsonDataset) -> None:
        import os

        for idx in range(min(5, len(dataset))):
            path = dataset[idx]["img_path"]
            assert os.path.isfile(path), f"image not found: {path}"


class TestGTJsonDatasetKeypoints:
    """Tests for keypoint definitions (no data needed)."""

    def test_num_keypoints(self) -> None:
        from pose24.keypoints import NUM_KEYPOINTS, POSE24_KEYPOINT_NAMES

        assert NUM_KEYPOINTS == 24
        assert len(POSE24_KEYPOINT_NAMES) == 24

    def test_mhr70_indices_unique(self) -> None:
        from pose24.keypoints import MHR70_INDICES

        assert len(MHR70_INDICES) == 24
        assert len(set(MHR70_INDICES)) == 24, "MHR70_INDICES has duplicates"

    def test_flip_indices_valid(self) -> None:
        from pose24.keypoints import FLIP_INDICES, NUM_KEYPOINTS

        assert len(FLIP_INDICES) == NUM_KEYPOINTS
        # Applying flip twice should return to original
        double_flip = tuple(FLIP_INDICES[i] for i in FLIP_INDICES)
        assert double_flip == tuple(range(NUM_KEYPOINTS))

    def test_joint_parents_valid(self) -> None:
        from pose24.keypoints import JOINT_PARENTS, NUM_KEYPOINTS

        assert len(JOINT_PARENTS) == NUM_KEYPOINTS
        for i, p in enumerate(JOINT_PARENTS):
            assert 0 <= p < NUM_KEYPOINTS, f"JOINT_PARENTS[{i}]={p} out of range"
