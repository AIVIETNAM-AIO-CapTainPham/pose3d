"""Custom data transforms for POSE-24 3D finetuning."""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Flip3DKeypoints(BaseTransform):
    """Synchronise ``keypoints_3d`` with the 2-D :class:`RandomFlip`.

    **Why this exists.** mmpose's ``RandomFlip`` only flips the 2-D
    ``results['keypoints']`` — it mirrors the x-coordinate and swaps left↔right
    joints via ``flip_indices`` — but it leaves the auxiliary ``keypoints_3d``
    key completely untouched.

    For RTMPose3D the codec (:class:`SimCC3DLabel`) derives each joint's **Z
    (depth) label** from ``keypoints_3d[..., 2]``, indexed by the *same* joint
    order as the (already left/right-swapped) 2-D keypoints.  Without this
    transform, every horizontally-flipped sample (≈50 % of the training set)
    gets its depth supervision assigned to the *wrong* left/right joint, so the
    network learns a corrupted left/right depth correspondence.  Symptom: the
    2-D overlay looks correct, but the predicted 3-D skeleton has its left/right
    limbs swapped in depth.

    **Placement.** Put this transform *immediately after* ``RandomFlip`` in the
    pipeline.  It reads the ``flip`` / ``flip_direction`` flags and the
    ``flip_indices`` that ``RandomFlip`` writes into the results dict, then:

    1. reorders ``keypoints_3d`` along the joint axis by ``flip_indices``
       (the left↔right swap), and
    2. negates the camera **X** axis (``x_right``) so the pose is a true mirror.

    Y (down) and Z (forward) are unchanged by a horizontal mirror.

    Only ``direction='horizontal'`` is supported; other directions are skipped
    with the keypoints_3d left as-is (and would need their own handling).
    """

    def transform(self, results: dict) -> dict:
        if not results.get("flip", False):
            return results

        kpts3d = results.get("keypoints_3d", None)
        if kpts3d is None:
            return results

        if results.get("flip_direction", "horizontal") != "horizontal":
            # Non-horizontal flips are not used in our pipelines; leave 3D
            # untouched rather than apply a wrong transform silently.
            return results

        flip_indices = results.get("flip_indices", None)
        if flip_indices is None:
            return results

        kpts3d = np.asarray(kpts3d).copy()
        # 1. left↔right reorder along the joint axis (..., K, 3)
        kpts3d = kpts3d[..., flip_indices, :]
        # 2. mirror camera X (x_right); Y, Z unchanged by a horizontal flip
        kpts3d[..., 0] *= -1.0

        results["keypoints_3d"] = kpts3d
        return results
