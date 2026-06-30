# POSE3D Project Notes

Thu muc nay la ban tong hop quan tri du an POSE3D tai thoi diem ra soat.
No khong thay the code goc; muc tieu la gom cac thong tin quan trong ve code,
data, checkpoint, cach chay va cac diem can chu y vao mot noi de de theo doi.

## Muc luc

- [01_overview.md](01_overview.md) - buc tranh tong quan, cau truc repo, module chinh.
- [02_code_architecture.md](02_code_architecture.md) - kien truc code, dataset, codec, model, loss.
- [03_data_inventory.md](03_data_inventory.md) - thong ke data, schema label, split, kich thuoc.
- [04_training_eval.md](04_training_eval.md) - config train, checkpoint, log, metric.
- [05_demo_tools_tests.md](05_demo_tools_tests.md) - demo Streamlit, tools CLI, test suite.
- [06_management_notes.md](06_management_notes.md) - ghi chu quan ly, rui ro, viec nen lam tiep.
- [07_environment_build.md](07_environment_build.md) - stack da verify va cach build `mmcv`.

## Tom tat nhanh

POSE3D la du an fine-tune RTMPose3D cho bo skeleton POSE-24, duoc rut tu
MHR70/SAM-3D-Body. Code chinh nam trong `src/pose24`, dung ecosystem
OpenMMLab (`mmpose`, `mmengine`, `mmcv`, `mmdet`) va co demo Streamlit tai
`demo/app.py`.

Data chinh nam trong `data/GT`:

- `images`: 18,752 anh.
- `labels`: 18,752 file JSON ground truth.
- `splits`: train/val/test = 15,939 / 1,875 / 938.
- `labels_backup` va `combined` co cung 18,752 mau, nen can xem la data/phien ban
  phu tro khi quan ly dung luong.

Checkpoint/log nam trong `work_dirs`:

- `work_dirs/pose24`: run dai den epoch 200, best hien con tren disk:
  `best_MPJPE_epoch_30.pth` va `best_MPJPE_epoch_100.pth`.
- `work_dirs/pose24_v2`: run den epoch 15, best hien tai:
  `best_MPJPE_epoch_15.pth`.
- `work_dirs/original_rtmw3d`: checkpoint goc `rtmw3d-l_cocktail14.pth`.

## Lenh hay dung

```bash
make splits
make train
make train-resume
make eval
make visualize
make test
make demo
```

Neu chay truc tiep, thuong can:

```bash
PYTHONPATH=src uv run python tools/train.py src/pose24/configs/rtmw3d_l_finetune_pose24.py --work-dir work_dirs/pose24_v2 --amp --cfg-options data_root=data/GT
```

