# Training va Evaluation

## Config chinh

File: `src/pose24/configs/rtmw3d_l_finetune_pose24.py`

Runtime:

- `max_epochs = 100`
- `stage2_num_epochs = 10`
- `base_lr = 5e-5`
- `num_keypoints = 24`
- random seed: `2024`
- validate moi 5 epoch: `train_cfg.val_interval = 5`

Optimizer:

- `AdamW`
- weight decay `0.05`
- `auto_scale_lr.base_batch_size = 512`

Scheduler:

- Linear warmup 500 iter, start factor `0.01`.
- Cosine annealing tu epoch 20 den 100.
- eta min = `base_lr * 0.05`.

## Model config

Backbone:

- `CSPNeXt`, arch `P5`
- pretrained RTMPose-L checkpoint tu OpenMMLab URL
- neck `CSPNeXtPAFPN`

Head:

- `RTMW3DHead`
- `out_channels = 24`
- input size `(288,384,288)`
- feature map size tu input chia 32
- final kernel size 7

Loss:

- `KLDiscretLossWithWeight`
- `Pose3DStructureLoss`

Test config:

- `flip_test=False`

## Pipeline train

`train_pipeline`:

1. `LoadImage`
2. `GetBBoxCenterScale`
3. `RandomFlip(horizontal)`
4. `Flip3DKeypoints`
5. `RandomHalfBody`
6. `RandomBBoxTransform(scale_factor=[0.6,1.4], rotate_factor=80)`
7. `TopdownAffine(input_size=(288,384))`
8. `YOLOXHSVRandomAug`
9. `Albumentation`: Blur, MedianBlur, CoarseDropout
10. `GenerateTarget(encoder=codec)`
11. `PackPoseInputs`

`train_pipeline_stage2`:

- switch o `max_epochs - stage2_num_epochs`
- transform bbox manh hon: scale `[0.5,1.5]`, rotate `90`
- bo HSV/CoarseDropout, chi giu Blur/MedianBlur

## Dataloader

- Train: batch size 32, num workers 8, shuffle.
- Val/Test: batch size 32, num workers 8, no shuffle, no drop_last.
- Dataset type: `GTJsonDataset`.
- Data root default: `data/GT`.

## Hooks

Custom hooks:

- `EMAHook`, momentum `0.0002`.
- `mmdet.PipelineSwitchHook`.
- `Pose3DVisualizationHook`, xuat 4 sample moi validation.

Checkpoint hook:

- `save_best="MPJPE"`
- `rule="less"`
- `max_keep_ckpts=3`

## Lenh train

Qua Makefile:

```bash
make train
make train-resume
make train-multi-gpu
```

Truc tiep:

```bash
PYTHONPATH=src uv run python tools/train.py \
  src/pose24/configs/rtmw3d_l_finetune_pose24.py \
  --work-dir work_dirs/pose24_v2 \
  --amp \
  --cfg-options data_root=data/GT
```

Resume:

```bash
PYTHONPATH=src uv run python tools/train.py \
  src/pose24/configs/rtmw3d_l_finetune_pose24.py \
  --work-dir work_dirs/pose24_v2 \
  --amp \
  --resume \
  --cfg-options data_root=data/GT
```

## Lenh eval

Makefile default:

```bash
make eval
```

Truc tiep:

```bash
PYTHONPATH=src uv run python tools/eval.py work_dirs/pose24_v2/best_MPJPE_epoch_15.pth --split val
```

Metric:

- `SimpleMPJPE(mode="mpjpe")`
- `SimpleMPJPE(mode="p-mpjpe")`

## Checkpoint va log hien co

Dung luong:

| Thu muc | So file | Dung luong |
|---|---:|---:|
| `work_dirs/pose24` | 176 | 3,798.9 MiB |
| `work_dirs/pose24_v2` | 27 | 1,841.4 MiB |
| `work_dirs/original_rtmw3d` | 1 | 220.1 MiB |

`work_dirs/pose24/last_checkpoint`:

```text
/home/angle/jupyterlab/DEV/POSE3D/work_dirs/pose24/epoch_200.pth
```

`work_dirs/pose24_v2/last_checkpoint`:

```text
/home/angle/jupyterlab/DEV/POSE3D/work_dirs/pose24_v2/epoch_15.pth
```

Trang thai 2026-06-24: training dang resume tiep tu `epoch_15.pth` (log moi
`work_dirs/pose24_v2/20260624_223048/`), dang o epoch ~19/100, `eta` ~19-20 gio.
Da phai resume lai 2 lan vi: (1) server bi sap nghi do OOM khi chay demo
Streamlit song song voi training, (2) venv thieu `albumentations` sau khi
moi truong bi thay doi boi lenh cai dat khac (xem risk #9 trong
`06_management_notes.md`). Khong mat checkpoint nao vi `--resume` doc lai
dung `last_checkpoint`.

Checkpoint dang co tren disk:

- `work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth`
- `work_dirs/pose24/best_MPJPE_epoch_30.pth`
- `work_dirs/pose24/best_MPJPE_epoch_100.pth`
- `work_dirs/pose24/epoch_95.pth`
- `work_dirs/pose24/epoch_180.pth`
- `work_dirs/pose24/epoch_190.pth`
- `work_dirs/pose24/epoch_200.pth`
- `work_dirs/pose24_v2/best_MPJPE_epoch_15.pth`
- `work_dirs/pose24_v2/epoch_10.pth`
- `work_dirs/pose24_v2/epoch_15.pth`

## Metric tu log

Run `work_dirs/pose24_v2/20260615_114044`:

| Epoch | MPJPE | P-MPJPE |
|---:|---:|---:|
| 5 | 0.390312 | 0.090729 |
| 10 | 0.378787 | 0.074111 |
| 15 | 0.377451 | 0.070440 |

Best duoc log: `best_MPJPE_epoch_15.pth`, MPJPE 0.3775.

Run `work_dirs/pose24/20260614_150720`:

- Best sớm trong log: epoch 30, MPJPE 0.375212, P-MPJPE 0.066444.
- Sau epoch 30, MPJPE val tang dan len ~0.385495 tai epoch 200.
- Dieu nay goi y overfit/metric degrade sau epoch 30 trong run dai.

Mot so moc:

| Epoch | MPJPE | P-MPJPE |
|---:|---:|---:|
| 5 | 0.381700 | 0.078937 |
| 15 | 0.377157 | 0.068395 |
| 30 | 0.375212 | 0.066444 |
| 100 | 0.378997 | 0.077997 |
| 200 | 0.385495 | 0.093312 |

## Ghi chu ve torch load

`tools/train.py`, `tools/eval.py` va `demo/app.py` deu can xu ly PyTorch >= 2.6
vi mac dinh `torch.load(weights_only=True)` co the loi voi checkpoint MMEngine.
Code hien tai dung `weights_only=False` cho checkpoint tin cay local.

