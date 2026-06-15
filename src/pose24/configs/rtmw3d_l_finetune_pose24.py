"""RTMPose3D-L finetune config for 24-keypoint MHR70 body subset.

Usage
-----
    mim train mmpose src/pose24/configs/rtmw3d_l_finetune_pose24.py \
        --work-dir work_dirs/pose24
"""

_base_ = ["mmpose::_base_/default_runtime.py"]

custom_imports = dict(imports=["pose24"], allow_failed_imports=False)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Pose3dLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# ── Runtime ────────────────────────────────────────────────────────────────
max_epochs = 100
stage2_num_epochs = 10
base_lr = 5e-5
num_keypoints = 24
randomness = dict(seed=2024)
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# ── Optimizer ──────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# Decay sớm: warmup 500 iter, rồi cosine giảm từ epoch 20 → 100.
# T_max = end − begin = 100 − 20 = 80 để LR chạm đáy đúng lúc kết thúc.
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=20,
        end=max_epochs,
        T_max=80,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
auto_scale_lr = dict(base_batch_size=512)

# ── Codec ──────────────────────────────────────────────────────────────────
# z_range=1.5 calibrated from GT stats: root-relative Z in ≈ ±0.9 m
codec = dict(
    type="SimCC3DLabel",
    input_size=(288, 384, 288),
    sigma=(6.0, 6.93, 6.0),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12),
    z_range=1.5,
)

# ── Backbone pretrained weights ────────────────────────────────────────────
_backbone_ckpt = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth"
)

# ── Model ──────────────────────────────────────────────────────────────────
model = dict(
    type="TopdownPoseEstimator3D",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=_backbone_ckpt),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(1, 2),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    head=dict(
        type="RTMW3DHead",
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=codec["input_size"],
        in_featuremap_size=tuple(s // 32 for s in codec["input_size"]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=[
            # Primary driver: SimCC classification on x/y/z (differentiable).
            dict(
                type="KLDiscretLossWithWeight",
                use_target_weight=True,
                beta=10.0,
                label_softmax=True,
            ),
            # Structural prior à la theia's pose_structure_loss: root + relative
            # + bone, masked Smooth-L1 in normalised [0,1] space.  Fed via
            # soft-argmax in RTMW3DHead so gradient flows to cls_x/y/z (unlike
            # the argmax-based BoneLoss it replaces).
            dict(
                type="Pose3DStructureLoss",
                root_weight=1.0,
                relative_weight=2.0,
                bone_weight=0.5,
                beta=0.05,
                loss_weight=1.0,
            ),
        ],
        decoder=codec,
    ),
    test_cfg=dict(flip_test=False),
)

# ── Data pipelines ─────────────────────────────────────────────────────────
_backend = dict(backend="local")

train_pipeline = [
    dict(type="LoadImage", backend_args=_backend),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    # Keep keypoints_3d in sync with the 2D flip (swap L/R + mirror X).
    dict(type="Flip3DKeypoints"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform", scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage", backend_args=_backend),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage", backend_args=_backend),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="Flip3DKeypoints"),
    dict(type="RandomHalfBody"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.0,
        scale_factor=[0.5, 1.5],
        rotate_factor=90,
    ),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# ── Dataset paths ──────────────────────────────────────────────────────────
# Override data_root when invoking training:
#   mim train mmpose ... --cfg-options data_root=<absolute_path>
data_root = "data/GT"

# metainfo is resolved by GTJsonDataset.METAINFO (absolute path), so we don't
# pass it here — a relative from_file would break under `mim train`'s cwd.
_common = dict(type="GTJsonDataset", data_root=data_root)

# ── Dataloaders ────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        **_common,
        ann_file=f"{data_root}/splits/train.txt",
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        **_common,
        ann_file=f"{data_root}/splits/val.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        **_common,
        ann_file=f"{data_root}/splits/test.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

# ── Evaluators ─────────────────────────────────────────────────────────────
val_evaluator = [
    dict(type="SimpleMPJPE", mode="mpjpe"),
    dict(type="SimpleMPJPE", mode="p-mpjpe"),
]
test_evaluator = val_evaluator

# ── Hooks ──────────────────────────────────────────────────────────────────
custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
    # Dump GT-vs-Pred comparison images each time we validate.
    dict(type="Pose3DVisualizationHook", num_samples=4, interval=1),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", save_best="MPJPE", rule="less", max_keep_ckpts=3
    )
)
