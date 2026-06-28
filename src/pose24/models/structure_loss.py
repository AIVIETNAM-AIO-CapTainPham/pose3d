"""Pose3DStructureLoss — differentiable structural pose loss.

Decomposes the pose into a *root* term (pelvis localization), a *relative*
term (root-relative per-joint geometry) and a *bone* term (edge-length
consistency), each a masked Smooth-L1 so invisible joints contribute no
gradient.

The RTMPose3D head emits SimCC distributions rather than regressing metric
coordinates directly, so this loss operates on **normalised [0, 1]
coordinates** obtained from the head via *soft-argmax* — fully differentiable,
so (unlike the argmax-based ``BoneLoss`` it replaces) gradient actually flows
back to ``cls_x/y/z``.

The loss expects ``pred`` / ``target`` of shape ``(N, K, 3)`` in [0, 1] and a
visibility weight of shape ``(N, K)``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmpose.registry import MODELS

from pose24.keypoints import ROOT_INDICES, SKELETON_EDGES


@MODELS.register_module()
class Pose3DStructureLoss(nn.Module):
    """Masked Smooth-L1 structural loss (root + relative + bone).

    Args:
        root_index: Keypoint indices whose mean defines the pelvis root.
        skeleton_edges: ``(start, end)`` joint pairs defining bones.
        root_weight: Weight of the absolute-root term.
        relative_weight: Weight of the root-relative per-joint term.
        bone_weight: Weight of the bone-length term.
        beta: Smooth-L1 transition point (in normalised units).
        loss_weight: Overall scaling applied to the summed loss.
    """

    # Tells RTMW3DHead to feed soft-argmax coordinates (not SimCC labels).
    requires_coords: bool = True

    def __init__(
        self,
        root_index: Sequence[int] = ROOT_INDICES,
        skeleton_edges: Sequence[Sequence[int]] = SKELETON_EDGES,
        root_weight: float = 1.0,
        relative_weight: float = 2.0,
        bone_weight: float = 0.5,
        beta: float = 0.05,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.root_index = list(root_index)
        edges = torch.tensor(skeleton_edges, dtype=torch.long)
        self.register_buffer("edge_start", edges[:, 0], persistent=False)
        self.register_buffer("edge_end", edges[:, 1], persistent=False)
        self.root_weight = root_weight
        self.relative_weight = relative_weight
        self.bone_weight = bone_weight
        self.beta = beta
        self.loss_weight = loss_weight
        self._loss_name = "loss_struct"

    @staticmethod
    def _masked_smooth_l1(
        pred: Tensor, target: Tensor, mask: Tensor, beta: float
    ) -> Tensor:
        loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        return loss.sum() / mask.expand_as(loss).sum().clamp_min(1.0)

    def forward(self, pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        # pred / target: (N, K, 3) in [0, 1];  weight: (N, K)
        pred_root = pred[:, self.root_index].mean(dim=1, keepdim=True)
        target_root = target[:, self.root_index].mean(dim=1, keepdim=True)
        pred_rel = pred - pred_root
        target_rel = target - target_root

        # Root term (pelvis is effectively always visible → unmasked).
        loss_root = F.smooth_l1_loss(pred_root, target_root, beta=self.beta)

        # Relative term: per-joint root-relative geometry, masked by visibility.
        loss_rel = self._masked_smooth_l1(pred_rel, target_rel, weight, self.beta)

        # Bone term: edge lengths in root-relative space, masked per edge.
        ps, pe = pred_rel[:, self.edge_start], pred_rel[:, self.edge_end]
        ts, te = target_rel[:, self.edge_start], target_rel[:, self.edge_end]
        pred_len = torch.linalg.vector_norm(ps - pe, dim=-1)
        target_len = torch.linalg.vector_norm(ts - te, dim=-1)
        edge_mask = weight[:, self.edge_start] * weight[:, self.edge_end]
        loss_bone = self._masked_smooth_l1(pred_len, target_len, edge_mask, self.beta)

        total = (
            self.root_weight * loss_root
            + self.relative_weight * loss_rel
            + self.bone_weight * loss_bone
        )
        return self.loss_weight * total

    @property
    def loss_name(self) -> str:
        return self._loss_name
