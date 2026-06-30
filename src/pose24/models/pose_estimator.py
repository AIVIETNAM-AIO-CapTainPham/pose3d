"""TopdownPoseEstimator3D — converts SimCC output to camera-frame 3D keypoints."""

from __future__ import annotations

from itertools import zip_longest
from typing import Optional

import numpy as np

from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.typing import InstanceList, PixelDataList, SampleList


@MODELS.register_module()
class TopdownPoseEstimator3D(TopdownPoseEstimator):
    """Top-down estimator that lifts 2.5D SimCC predictions to camera space."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Fallback intrinsics calibrated from GT dataset (SAM-3D-Body).
        # Median focal length recovered from 50 GT files: ~2074 px.
        # The original RTMPose3D default (f=1145) was ~1.8x too short, causing
        # back-projected X,Y to be ~1.8x too large (MPJPE ~746 mm → scale error).
        self.camera_param = {
            "c": [512.54150496, 515.45148698],
            "f": [2074.0, 2074.0],
        }

    def add_pred_to_datasample(
        self,
        batch_pred_instances: InstanceList,
        batch_pred_fields: Optional[PixelDataList],
        batch_data_samples: SampleList,
    ) -> SampleList:
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get("output_keypoint_indices", None)
        mode = self.test_cfg.get("mode", "3d")
        assert mode in ["2d", "3d", "vis"]

        for pred_instances, pred_fields, data_sample in zip_longest(
            batch_pred_instances, batch_pred_fields, batch_data_samples
        ):
            gt_instances = data_sample.gt_instances

            input_center = data_sample.metainfo["input_center"]
            input_scale = data_sample.metainfo["input_scale"]
            input_size = data_sample.metainfo["input_size"]
            keypoints_3d = pred_instances.keypoints
            keypoints_simcc = pred_instances.keypoints_simcc

            # input space → image space
            keypoints_2d = keypoints_3d[..., :2].copy()
            keypoints_2d = (
                keypoints_2d / input_size * input_scale
                + input_center
                - 0.5 * input_scale
            )

            # image space → camera space
            if gt_instances.get("camera_params", None) is not None:
                camera_params = gt_instances.camera_params[0]
                f = np.array(camera_params["f"])
                c = np.array(camera_params["c"])
            else:
                # Only hit for samples truly missing focal_length_px (none in
                # GTJsonDataset's GT, since it now reads the per-sample value)
                # or for ad-hoc images (e.g. demo upload) with no GT at all.
                # f calibrated from GT median (~2074 px). Note ori_shape is
                # (H, W) per mmcv/mmpose convention — c must be (cx, cy) =
                # (W/2, H/2), so build it explicitly rather than halving
                # ori_shape directly (that would silently swap cx/cy on any
                # non-square image, which is ~96% of this dataset).
                f = np.array(self.camera_param["f"])
                h, w = data_sample.ori_shape
                c = np.array([w / 2.0, h / 2.0])
            kpts_pixel = np.concatenate(
                [keypoints_2d, (keypoints_3d[..., 2] + gt_instances.root_z)[..., None]],
                axis=-1,
            )
            kpts_cam = kpts_pixel.copy()
            kpts_cam[..., :2] = (kpts_pixel[..., :2] - c) / f * kpts_pixel[..., 2:]

            if mode == "3d":
                pred_instances.keypoints = kpts_cam
                pred_instances.transformed_keypoints = keypoints_2d
            elif mode == "vis":
                pred_instances.keypoints = keypoints_simcc
                pred_instances.transformed_keypoints = keypoints_2d
            else:
                pred_instances.keypoints = keypoints_2d
                pred_instances.transformed_keypoints = keypoints_2d

            if "keypoints_visible" not in pred_instances:
                pred_instances.keypoints_visible = pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith("keypoint"):
                        pred_instances.set_field(value[:, output_keypoint_indices], key)

            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices], key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
