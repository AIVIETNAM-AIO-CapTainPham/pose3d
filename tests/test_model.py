"""Integration test: build the full RTMPose3D model and run loss + predict.

Backbone pretrained download is disabled (init_cfg=None) so this runs offline
on CPU.  Heavy but exercises estimator + head + both losses end-to-end.
"""

from __future__ import annotations

import pytest

import pose24  # noqa: F401 — registers estimator/head/loss/codec/dataset


@pytest.fixture(scope="module")
def cfg():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    init_default_scope("mmpose")
    c = Config.fromfile("src/pose24/configs/rtmw3d_l_finetune_pose24.py")
    c.model.backbone.init_cfg = None  # no network during tests
    return c


@pytest.fixture(scope="module")
def model(cfg):
    from mmpose.registry import MODELS

    return MODELS.build(cfg.model).eval()


def _batch(cfg, split_key: str, n: int = 2):
    from mmpose.registry import DATASETS
    from mmengine.dataset import pseudo_collate

    ds = DATASETS.build(cfg[split_key].dataset)
    return pseudo_collate([ds[i] for i in range(n)])


class TestModelRegistration:
    def test_custom_modules_registered(self) -> None:
        from mmpose.registry import MODELS, KEYPOINT_CODECS

        assert "TopdownPoseEstimator3D" in MODELS._module_dict
        assert "RTMW3DHead" in MODELS._module_dict
        assert "KLDiscretLossWithWeight" in MODELS._module_dict
        assert "SimCC3DLabel" in KEYPOINT_CODECS._module_dict


class TestModelBuild:
    def test_head_output_dims(self, model) -> None:
        # cls_x = 288*2, cls_y = 384*2, cls_z = 288*2
        assert model.head.cls_x.out_features == 576
        assert model.head.cls_y.out_features == 768
        assert model.head.cls_z.out_features == 576

    def test_two_losses(self, model) -> None:
        names = [type(loss).__name__ for loss in model.head.loss_module]
        assert names == ["KLDiscretLossWithWeight", "Pose3DStructureLoss"]


class TestForward:
    def test_loss_step(self, cfg, model) -> None:
        batch = _batch(cfg, "train_dataloader")
        data = model.data_preprocessor(batch, training=True)
        losses = model(data["inputs"], data["data_samples"], mode="loss")
        assert "loss_kld" in losses
        assert "loss_struct" in losses
        for v in losses.values():
            assert v.isfinite().all(), "non-finite loss"

    def test_struct_loss_gradient_reaches_depth_head(self, cfg, model) -> None:
        """The structural loss must backprop to cls_z (the depth head).

        This is the whole point of the soft-argmax rewrite: the argmax-based
        BoneLoss it replaces produced zero gradient.
        """
        batch = _batch(cfg, "train_dataloader")
        data = model.data_preprocessor(batch, training=True)
        losses = model(data["inputs"], data["data_samples"], mode="loss")
        model.zero_grad()
        losses["loss_struct"].backward()
        gz = model.head.cls_z.weight.grad
        assert (
            gz is not None and gz.abs().sum() > 0
        ), "loss_struct produced no gradient on the depth head"

    def test_predict_step(self, cfg, model) -> None:
        import torch
        from pose24.keypoints import NUM_KEYPOINTS

        batch = _batch(cfg, "val_dataloader")
        data = model.data_preprocessor(batch, training=False)
        with torch.no_grad():
            out = model(data["inputs"], data["data_samples"], mode="predict")
        kp = out[0].pred_instances.keypoints
        assert kp.shape[1:] == (NUM_KEYPOINTS, 3)
