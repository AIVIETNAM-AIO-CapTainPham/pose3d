"""Dataset for SAM-3D-Body GT JSON labels → 24 POSE keypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import mmpose.datasets  # noqa: F401 – registers all mmpose transforms
from mmengine.dataset import Compose
from mmpose.datasets.datasets.base import BaseCocoStyleDataset
from mmpose.registry import DATASETS, TRANSFORMS

from pose24.keypoints import MHR70_INDICES, NUM_KEYPOINTS

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Ankle/knee MPJPE runs 3-5x other joints (kinematic-chain error accumulation:
# the hip root is centered out for free, knee/ankle errors compound on top of
# it). GT has no real per-joint visibility, so keypoints_visible is otherwise
# flat 1.0 everywhere — this is the only lever available to tell the loss
# "these joints matter more". Values must stay >= 0.5: SimCC3DLabel's Gaussian
# generator skips emitting any target at all below that (hard cutoff, not a
# soft down-weight), while SimpleMPJPE's visibility mask is `astype(bool)` so
# any nonzero weight still counts the joint in eval MPJPE — going below 0.5
# would silently train the head toward an all-zero target while still being
# scored as if it were a normal sample.
_LEG_JOINT_INDICES = (13, 14, 15, 16)  # left/right knee, left/right ankle
_LEG_BOOST_WEIGHT = 1.8

# Absolute path to pose24.py metainfo config (resolved at import time)
_METAINFO_FILE = str(
    Path(__file__).resolve().parent.parent
    / "configs"
    / "_base_"
    / "datasets"
    / "pose24.py"
)


@DATASETS.register_module()
class GTJsonDataset(BaseCocoStyleDataset):
    """Per-sample GT JSON dataset produced by fix_pose3d.py.

    Expected directory layout::

        <data_root>/
        ├── images/   <sample_id>.jpg
        └── labels/   <sample_id>.json

    Each JSON must contain:

    * ``keypoints_3d_cam_m`` – (70, 3) MHR70 3-D camera-frame keypoints (m)
    * ``pred_keypoints_2d_px`` – (70, 2) projected 2-D pixel coords
    * ``bbox_xyxy_normalized`` – [x1, y1, x2, y2] in [0, 1]
    * ``prediction.image_size`` – ``{width, height}``

    Args:
        data_root: Dataset root directory.
        ann_file: Optional plain-text file listing sample IDs (one per line,
            no extension).  If empty/absent every JSON in ``labels/`` is used.
        pipeline: mmpose data-transform pipeline.
        test_mode: Set ``True`` for val/test splits.
    """

    METAINFO: dict = dict(from_file=_METAINFO_FILE)

    def __init__(
        self,
        data_root: str,
        ann_file: str = "",
        pipeline: list | None = None,
        test_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self._gt_data_root = Path(data_root)
        self._ann_file_path = ann_file
        # Pass pipeline=[] so mmengine's Compose doesn't try to resolve
        # mmpose transforms before they're in the mmengine::transform scope.
        # We rebuild the pipeline below using mmpose's TRANSFORMS registry.
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=[],
            test_mode=test_mode,
            **kwargs,
        )
        if pipeline:
            self.pipeline = Compose(
                [TRANSFORMS.build(t) if isinstance(t, dict) else t for t in pipeline]
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_keypoint_weights() -> np.ndarray:
        """Per-joint loss weight: boost knee/ankle."""
        weights = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        weights[list(_LEG_JOINT_INDICES)] = _LEG_BOOST_WEIGHT
        return weights

    def _find_image(self, sample_id: str, hint: str | None) -> str | None:
        img_dir = self._gt_data_root / "images"
        candidates: list[Path] = []
        if hint:
            candidates.append(img_dir / Path(hint).name)
        for ext in _IMAGE_EXTS:
            candidates.append(img_dir / f"{sample_id}{ext}")
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    # ------------------------------------------------------------------
    # BaseCocoStyleDataset interface
    # ------------------------------------------------------------------

    def load_data_list(self) -> list[dict]:
        labels_dir = self._gt_data_root / "labels"

        if self._ann_file_path and os.path.isfile(self._ann_file_path):
            ids = Path(self._ann_file_path).read_text(encoding="utf-8").splitlines()
            json_paths = [
                labels_dir / f"{sid.strip()}.json" for sid in ids if sid.strip()
            ]
        else:
            json_paths = sorted(labels_dir.glob("*.json"))

        data_list: list[dict] = []
        for idx, jp in enumerate(json_paths):
            info = self._parse_json(idx, jp)
            if info is not None:
                data_list.append(info)
        return data_list

    def _parse_json(self, idx: int, json_path: Path) -> dict | None:
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(raw, dict) or raw.get("status", "ok") != "ok":
            return None

        sample_id = str(raw.get("id") or json_path.stem)

        # 3-D keypoints: (70,3) MHR70 → (24,3) POSE
        kpts3d_raw = raw.get("keypoints_3d_cam_m")
        if not kpts3d_raw or len(kpts3d_raw) != 70:
            return None
        kpts3d = np.array(kpts3d_raw, dtype=np.float32)[list(MHR70_INDICES)]

        # 2-D pixel keypoints: (70,2) → (24,2)
        kpts2d_raw = raw.get("pred_keypoints_2d_px")
        if not kpts2d_raw or len(kpts2d_raw) != 70:
            return None
        kpts2d = np.array(kpts2d_raw, dtype=np.float32)[list(MHR70_INDICES)]

        # Image size
        pred = raw.get("prediction", {})
        img_sz = pred.get("image_size") or raw.get("image_size", {})
        width, height = int(img_sz.get("width", 0)), int(img_sz.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        # Bounding box: normalised → pixels, shape (1, 4)
        bn = raw.get("bbox_xyxy_normalized")
        if not bn or len(bn) != 4:
            return None
        x1, y1, x2, y2 = (
            float(bn[0]) * width,
            float(bn[1]) * height,
            float(bn[2]) * width,
            float(bn[3]) * height,
        )
        if x2 <= x1 or y2 <= y1:
            return None

        img_path = self._find_image(sample_id, raw.get("image_path"))
        if img_path is None:
            return None

        # Per-sample camera intrinsics (focal_length_px varies hugely across
        # this dataset's mixed sources — median ~2074px but anywhere from
        # ~775 to ~6900px). Without this, TopdownPoseEstimator3D falls back to
        # one fixed focal length for every sample, which back-projects 2D→3D
        # camera-space coords at the wrong scale whenever a sample's true f
        # differs from that fallback (the further f is from the fallback, the
        # larger the resulting MPJPE outlier — pure projection-scale error,
        # unrelated to how well the model actually localised the keypoints).
        focal_px = raw.get("focal_length_px")
        camera_param = (
            dict(f=[focal_px, focal_px], c=[width / 2.0, height / 2.0])
            if focal_px
            else None
        )

        keypoint_weights = self._compute_keypoint_weights()

        return dict(
            id=idx,
            img_id=idx,
            img_path=img_path,
            bbox=np.array([[x1, y1, x2, y2]], dtype=np.float32),  # (1,4)
            bbox_score=np.ones((1,), dtype=np.float32),
            keypoints=kpts2d[np.newaxis],  # (1,24,2)
            keypoints_visible=keypoint_weights[np.newaxis],  # (1,24)
            keypoints_3d=kpts3d[np.newaxis],  # (1,24,3)
            camera_param=[camera_param],  # per-instance, matches bbox's (1,...)
            num_keypoints=NUM_KEYPOINTS,
            category_id=1,
            iscrowd=False,
        )
