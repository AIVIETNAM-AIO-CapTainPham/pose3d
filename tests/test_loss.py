"""Unit tests for Pose3DStructureLoss (pure tensor math, no model build)."""

from __future__ import annotations

import torch

import pose24  # noqa: F401 — registers Pose3DStructureLoss
from pose24.keypoints import NUM_KEYPOINTS
from pose24.models import Pose3DStructureLoss


def _coords(n: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(n, NUM_KEYPOINTS, 3)


class TestStructureLoss:
    def test_zero_when_identical(self) -> None:
        loss = Pose3DStructureLoss()
        x = _coords()
        w = torch.ones(x.shape[:2])
        out = loss(x, x.clone(), w)
        assert float(out) == 0.0

    def test_positive_when_different(self) -> None:
        loss = Pose3DStructureLoss()
        pred, target = _coords(), _coords() + 0.1
        w = torch.ones(pred.shape[:2])
        assert float(loss(pred, target, w)) > 0.0

    def test_invisible_joints_masked(self) -> None:
        """A joint with zero weight must not affect the relative/bone terms."""
        loss = Pose3DStructureLoss(root_weight=0.0)  # isolate rel+bone
        pred, target = _coords(), _coords()
        w = torch.ones(pred.shape[:2])
        # Corrupt a non-root joint heavily, but mask it out.
        pred_bad = pred.clone()
        pred_bad[:, 9] += 100.0  # left_wrist
        w_masked = w.clone()
        w_masked[:, 9] = 0.0
        base = loss(pred, target, w)
        masked = loss(pred_bad, target, w_masked)
        assert torch.allclose(base, masked, atol=1e-5)

    def test_gradient_flows(self) -> None:
        loss = Pose3DStructureLoss()
        pred = _coords().requires_grad_(True)
        target = _coords() + 0.1  # must differ from pred for non-zero loss
        w = torch.ones(pred.shape[:2])
        loss(pred, target, w).backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0

    def test_loss_name(self) -> None:
        assert Pose3DStructureLoss().loss_name == "loss_struct"

    def test_requires_coords_flag(self) -> None:
        assert Pose3DStructureLoss.requires_coords is True
