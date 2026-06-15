"""SimCC3DLabel — 3D keypoint codec for RTMPose3D finetuning.

Ported from clone_code/pose3d/rtmpose3d (read-only reference).
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from mmpose.codecs.base import BaseKeypointCodec
from mmpose.codecs.utils.refinement import refine_simcc_dark
from mmpose.registry import KEYPOINT_CODECS

from .utils import get_simcc_maximum


@KEYPOINT_CODECS.register_module()
class SimCC3DLabel(BaseKeypointCodec):
    r"""Generate 3D keypoint SimCC labels.

    Encodes (x, y) 2D image coords and z depth into separate Gaussian label
    distributions along each axis.

    Args:
        input_size: (W, H, D) — image width/height and depth bins.
        sigma: Gaussian sigma for x/y/z.  Single float or 3-tuple.
        simcc_split_ratio: Label size multiplier relative to input_size.
        normalize: Normalise Gaussian labels to unit integral.
        use_dark: Apply DARK post-processing refinement during decode.
        root_index: Keypoint index(es) used as Z-axis root.
        z_range: Half-range of root-relative Z in metres.
    """

    auxiliary_encode_keys = {"keypoints_3d"}

    label_mapping_table = dict(
        keypoint_x_labels="keypoint_x_labels",
        keypoint_y_labels="keypoint_y_labels",
        keypoint_z_labels="keypoint_z_labels",
        keypoint_weights="keypoint_weights",
        weight_z="weight_z",
        with_z_label="with_z_label",
    )

    instance_mapping_table = dict(
        bbox="bboxes",
        bbox_score="bbox_scores",
        bbox_scale="bbox_scales",
        lifting_target="lifting_target",
        lifting_target_visible="lifting_target_visible",
        camera_param="camera_params",
        root_z="root_z",
    )

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        sigma: Union[float, int, Tuple[float, ...]] = 6.0,
        simcc_split_ratio: float = 2.0,
        normalize: bool = True,
        use_dark: bool = False,
        root_index: Union[int, Tuple[int, ...]] = 0,
        z_range: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize
        self.use_dark = use_dark
        self.sigma = np.array(
            [sigma, sigma, sigma] if isinstance(sigma, (float, int)) else sigma,
            dtype=np.float64,
        )
        self.root_index = (
            list(root_index) if isinstance(root_index, tuple) else [root_index]
        )
        self.root_z = [5.14388]
        self.z_range = z_range if z_range is not None else 2.1744869

    # ------------------------------------------------------------------

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_3d: Optional[np.ndarray] = None,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> dict:
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        lifting_target = [None]
        root_z = self.root_z
        with_z_label = False

        if keypoints_3d is not None:
            lifting_target = keypoints_3d.copy()
            root_z = keypoints_3d[..., self.root_index, 2].mean(1)
            keypoints_3d[..., 2] -= root_z
            keypoints_z = (keypoints_3d[..., 2] / self.z_range + 1) * (
                self.input_size[2] / 2
            )
            keypoints_3d = np.concatenate([keypoints, keypoints_z[..., None]], axis=-1)
            x, y, z, keypoint_weights = self._generate_gaussian(
                keypoints_3d, keypoints_visible
            )
            weight_z = keypoint_weights
            with_z_label = True
        else:
            if keypoints.shape != np.zeros([]).shape:
                keypoints_z = np.ones(
                    (keypoints.shape[0], keypoints.shape[1], 1), dtype=np.float32
                )
                keypoints = np.concatenate([keypoints, keypoints_z], axis=-1)
                x, y, z, keypoint_weights = self._generate_gaussian(
                    keypoints, keypoints_visible
                )
            else:
                x, y, z = np.zeros((3, 1), dtype=np.float32)
                keypoint_weights = np.ones((1,))
            weight_z = np.zeros_like(keypoint_weights)

        return dict(
            keypoint_x_labels=x,
            keypoint_y_labels=y,
            keypoint_z_labels=z,
            lifting_target=lifting_target,
            # Visibility for the lifting target, consumed by SimpleMPJPE.
            lifting_target_visible=keypoints_visible.copy(),
            root_z=root_z,
            keypoint_weights=keypoint_weights,
            weight_z=weight_z,
            with_z_label=[with_z_label],
        )

    def decode(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        keypoints, scores = get_simcc_maximum(x, y, z)

        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        if self.use_dark:
            x_blur = int((self.sigma[0] * 20 - 7) // 3)
            y_blur = int((self.sigma[1] * 20 - 7) // 3)
            z_blur = int((self.sigma[2] * 20 - 7) // 3)
            x_blur -= int((x_blur % 2) == 0)
            y_blur -= int((y_blur % 2) == 0)
            z_blur -= int((z_blur % 2) == 0)
            keypoints[:, :, 0] = refine_simcc_dark(keypoints[:, :, 0], x, x_blur)
            keypoints[:, :, 1] = refine_simcc_dark(keypoints[:, :, 1], y, y_blur)
            keypoints[:, :, 2] = refine_simcc_dark(keypoints[:, :, 2], z, z_blur)

        keypoints /= self.simcc_split_ratio
        keypoints_simcc = keypoints.copy()
        keypoints[..., 2:3] = (
            keypoints[..., 2:3] / (self.input_size[-1] / 2) - 1
        ) * self.z_range
        return keypoints, keypoints_simcc, scores

    # ------------------------------------------------------------------

    def _map_coordinates(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        keypoints_split = np.around(keypoints.copy() * self.simcc_split_ratio).astype(
            np.int64
        )
        return keypoints_split, keypoints_visible.copy()

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        N, K, _ = keypoints.shape
        w, h, d = self.input_size
        W = int(np.around(w * self.simcc_split_ratio))
        H = int(np.around(h * self.simcc_split_ratio))
        D = int(np.around(d * self.simcc_split_ratio))

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible
        )

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)
        target_z = np.zeros((N, K, D), dtype=np.float32)

        radius = self.sigma * 3
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)
        z = np.arange(0, D, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            if keypoints_visible[n, k] < 0.5:
                continue
            mu = keypoints_split[n, k]
            left, top, near = mu - radius
            right, bottom, far = mu + radius + 1
            if left >= W or top >= H or near >= D or right < 0 or bottom < 0 or far < 0:
                keypoint_weights[n, k] = 0
                continue
            mu_x, mu_y, mu_z = mu
            target_x[n, k] = np.exp(-((x - mu_x) ** 2) / (2 * self.sigma[0] ** 2))
            target_y[n, k] = np.exp(-((y - mu_y) ** 2) / (2 * self.sigma[1] ** 2))
            target_z[n, k] = np.exp(-((z - mu_z) ** 2) / (2 * self.sigma[2] ** 2))

        if self.normalize:
            norm = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm[0]
            target_y /= norm[1]
            target_z /= norm[2]

        return target_x, target_y, target_z, keypoint_weights
