"""Pose visualisation helpers — 2D image overlay + 3D skeleton, GT vs Pred.

Pure rendering functions (no model / no mmpose deps) so they are easy to unit
test.  Coordinate conventions:

* 2D keypoints: full-image pixel coordinates, shape ``(K, 2)``.
* 3D keypoints: camera frame in metres, ``x_right_y_down_z_forward``,
  shape ``(K, 3)``.  For display we map (x, z, -y) so the person stands upright.

GT is drawn in green, Pred in red, so a trainer can see at a glance whether the
prediction is skewed relative to ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pose24.keypoints import NUM_KEYPOINTS, SKELETON_EDGES  # noqa: E402

GT_COLOR = "#2ecc40"  # green
PRED_COLOR = "#ff4136"  # red


def _check(kpts: np.ndarray, dim: int, name: str) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.shape != (NUM_KEYPOINTS, dim):
        raise ValueError(f"{name} must be ({NUM_KEYPOINTS}, {dim}), got {kpts.shape}")
    return kpts


def plot_2d_overlay(
    ax,
    image_rgb: np.ndarray,
    gt_xy: np.ndarray,
    pred_xy: Optional[np.ndarray] = None,
    visible: Optional[np.ndarray] = None,
) -> None:
    """Draw GT (and optional Pred) keypoints + skeleton over an image."""
    gt_xy = _check(gt_xy, 2, "gt_xy")
    if visible is None:
        visible = np.ones(NUM_KEYPOINTS, dtype=bool)
    visible = np.asarray(visible).astype(bool)

    ax.imshow(image_rgb)
    _draw_2d_skeleton(ax, gt_xy, visible, GT_COLOR, "GT")
    if pred_xy is not None:
        pred_xy = _check(pred_xy, 2, "pred_xy")
        _draw_2d_skeleton(ax, pred_xy, visible, PRED_COLOR, "Pred")
    ax.set_title("2D keypoints on image")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)


def _draw_2d_skeleton(ax, xy, visible, color, label) -> None:
    for i, j in SKELETON_EDGES:
        if visible[i] and visible[j]:
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )
    vis = visible
    ax.scatter(
        xy[vis, 0],
        xy[vis, 1],
        c=color,
        s=14,
        edgecolors="white",
        linewidths=0.4,
        label=label,
        zorder=3,
    )


def plot_3d_skeleton(
    ax,
    gt_3d: np.ndarray,
    pred_3d: Optional[np.ndarray] = None,
) -> None:
    """Draw GT (and optional Pred) skeleton in a 3D axis (upright view)."""
    gt_3d = _check(gt_3d, 3, "gt_3d")
    _draw_3d_skeleton(ax, gt_3d, GT_COLOR, "GT")
    pts = [gt_3d]
    if pred_3d is not None:
        pred_3d = _check(pred_3d, 3, "pred_3d")
        _draw_3d_skeleton(ax, pred_3d, PRED_COLOR, "Pred")
        pts.append(pred_3d)

    _set_equal_3d(ax, np.concatenate(pts, axis=0))
    ax.set_title("3D keypoints (camera frame)")
    ax.set_xlabel("x (right)")
    ax.set_ylabel("z (forward)")
    ax.set_zlabel("y (up)")
    ax.legend(loc="upper right", fontsize=8)


def _draw_3d_skeleton(ax, kpts, color, label) -> None:
    # Map camera (x, y_down, z_fwd) → display (x, z, -y) so up is up.
    X, Y, Z = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    dx, dy, dz = X, Z, -Y
    for i, j in SKELETON_EDGES:
        ax.plot(
            [dx[i], dx[j]],
            [dy[i], dy[j]],
            [dz[i], dz[j]],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    ax.scatter(dx, dy, dz, c=color, s=16, label=label, depthshade=False)


def _set_equal_3d(ax, pts) -> None:
    X, Y, Z = pts[:, 0], pts[:, 2], -pts[:, 1]
    max_range = np.array([np.ptp(X), np.ptp(Y), np.ptp(Z)]).max() / 2.0
    mid = [(v.max() + v.min()) / 2 for v in (X, Y, Z)]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def render_comparison(
    image_rgb: np.ndarray,
    gt_2d: np.ndarray,
    gt_3d: np.ndarray,
    pred_2d: Optional[np.ndarray] = None,
    pred_3d: Optional[np.ndarray] = None,
    visible: Optional[np.ndarray] = None,
    title: str = "",
    out_path: Optional[str] = None,
):
    """Render a 2-panel figure (2D overlay + 3D skeleton) and optionally save.

    Returns the matplotlib Figure.
    """
    fig = plt.figure(figsize=(12, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    plot_2d_overlay(ax2d, image_rgb, gt_2d, pred_2d, visible)
    plot_3d_skeleton(ax3d, gt_3d, pred_3d)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig
