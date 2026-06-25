"""RTMPose3D-L finetune config — v3: chống overfit-sớm (hội tụ tại epoch ~30).

Quan sát từ 2 run trước (pose24, pose24_v2): val MPJPE luôn hội tụ ~epoch 30
dù LR/decay timing khác nhau — đây là giới hạn từ quy mô dataset (16k ảnh),
không phải do LR đẩy quá nhanh. Đẩy điểm hội tụ này xa hơn cần làm model khó
"học thuộc" train set hơn sau khi đã nắm hết nguyên lý chung:

  1. Regularization mạnh hơn: weight_decay 0.05→0.1, dropout 0.1→0.2.
  2. Augmentation mạnh hơn: range scale/rotate rộng hơn, occlusion nhiều hơn.
  3. LR thấp hơn + decay sớm hơn (hỗ trợ, không phải nguyên nhân chính).

Dùng:
    PYTHONPATH=src uv run python tools/train.py \
        src/pose24/configs/rtmw3d_l_finetune_pose24_v3.py \
        --work-dir work_dirs/pose24_v3 --amp
"""

_base_ = ["rtmw3d_l_finetune_pose24.py"]

# ── Runtime ────────────────────────────────────────────────────────────────
max_epochs = 100
stage2_num_epochs = 10
base_lr = 2e-5  # giảm thêm từ 5e-5 → 2e-5
num_keypoints = 24
randomness = dict(seed=2024)
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# ── Optimizer: regularization mạnh hơn ──────────────────────────────────────
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.1),  # 0.05 → 0.1
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# Decay sớm hơn nữa: bắt đầu epoch 10 (thay 20).
param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=10,
        end=max_epochs,
        T_max=90,  # T_max = end - begin = 100 - 10
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
auto_scale_lr = dict(base_batch_size=512)

# ── Codec (giữ nguyên từ base) ───────────────────────────────────────────────
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

# ── Backbone pretrained weights (giữ nguyên) ────────────────────────────────
_backbone_ckpt = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth"
)

# ── Model: dropout cao hơn ───────────────────────────────────────────────────
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
            dropout_rate=0.2,  # 0.1 → 0.2
            drop_path=0.1,  # 0.0 → 0.1 (thêm stochastic depth nhẹ)
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=[
            dict(
                type="KLDiscretLossWithWeight",
                use_target_weight=True,
                beta=10.0,
                label_softmax=True,
            ),
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

# ── Data pipelines: augmentation mạnh hơn ───────────────────────────────────
_backend = dict(backend="local")

train_pipeline = [
    dict(type="LoadImage", backend_args=_backend),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="Flip3DKeypoints"),
    dict(type="RandomHalfBody"),
    # scale/rotate range rộng hơn: [0.6,1.4]→[0.5,1.6], 80°→90°
    dict(type="RandomBBoxTransform", scale_factor=[0.5, 1.6], rotate_factor=90),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.2),  # 0.1 → 0.2
            dict(type="MedianBlur", p=0.2),  # 0.1 → 0.2
            dict(
                type="CoarseDropout",
                max_holes=2,  # 1 → 2 vùng che
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.15,
                min_width=0.15,
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
            dict(type="Blur", p=0.15),
            dict(type="MedianBlur", p=0.15),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# ── Dataset paths (giữ nguyên) ───────────────────────────────────────────────
data_root = "data/GT"
_common = dict(type="GTJsonDataset", data_root=data_root)

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

# ── Evaluators (giữ nguyên) ──────────────────────────────────────────────────
val_evaluator = [
    dict(type="SimpleMPJPE", mode="mpjpe"),
    dict(type="SimpleMPJPE", mode="p-mpjpe"),
]
test_evaluator = val_evaluator

# ── Hooks: thêm EarlyStoppingHook ────────────────────────────────────────────
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
    dict(type="Pose3DVisualizationHook", num_samples=4, interval=1),
    # Tự dừng nếu val MPJPE không cải thiện sau 6 lần val liên tiếp
    # (6 * val_interval=5 = 30 epoch không cải thiện) — tránh lãng phí thời
    # gian train tiếp khi đã chắc chắn overfit.
    dict(
        type="EarlyStoppingHook",
        monitor="MPJPE",
        rule="less",
        patience=6,
        min_delta=0.0001,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", save_best="MPJPE", rule="less", max_keep_ckpts=3
    )
)
