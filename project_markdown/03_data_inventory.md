# Data Inventory

## Thu muc data chinh

Root: `data/GT`

Thong ke hien tai:

| Thu muc | So file | Dung luong |
|---|---:|---:|
| `data/GT/images` | 18,752 | 6,485.6 MiB |
| `data/GT/labels` | 18,752 | 1,096.7 MiB |
| `data/GT/labels_backup` | 18,752 | 966.5 MiB |
| `data/GT/combined/images` | 18,752 | 6,485.6 MiB |
| `data/GT/combined/labels` | 18,752 | 1,096.7 MiB |

Tong file trong `data/GT` cap maxdepth 2: 56,260 file.

Luu y quan ly dung luong: `images` va `combined/images` dang co cung so luong
va dung luong. Neu day la duplicate, can thong nhat vai tro de tranh ton o dia.

## Split

Files:

- `data/GT/splits/train.txt`: 15,939 mau.
- `data/GT/splits/val.txt`: 1,875 mau.
- `data/GT/splits/test.txt`: 938 mau.
- Tong: 18,752 mau.

Ti le gan dung: train 85%, val 10%, test 5%.

Script tao split: `tools/make_splits.py`

Mac dinh:

- train fraction: `0.85`
- val fraction: `0.10`
- test la phan con lai
- seed: `42`

Lenh:

```bash
make splits
```

hoac:

```bash
PYTHONPATH=src uv run python tools/make_splits.py --data-root data/GT
```

## Schema JSON label

Moi file `data/GT/labels/<id>.json` tuong ung 1 anh.

Mau da doc:

- file: `pinterest_pinterest_132785889008574419_3081231486.json`
- `status`: `ok`
- `keypoint_schema`: `MHR70`
- `camera_coordinate_system`: `x_right_y_down_z_forward_m`
- `prediction.image_size`: vi du `{width: 864, height: 1536}`
- `keypoints_3d_cam_m`: 70 x 3
- `pred_keypoints_2d_px`: 70 x 2

Field quan trong:

| Field | Vai tro |
|---|---|
| `keypoints_3d_cam_m` | 70 keypoint 3D trong camera frame, don vi met |
| `keypoints_3d_root_midhip_m` | 3D root-relative theo midhip |
| `pred_keypoints_2d_px` | 70 keypoint 2D pixel |
| `bbox_xyxy_normalized` | bbox normalized `[x1,y1,x2,y2]` |
| `prediction.image_size` / `image_size` | kich thuoc anh |
| `focal_length_px` | focal length pixel, dung verify projection |
| `midhip_translation_cam_m` | vi tri midhip trong camera frame |
| `distance_sam_midhip_z_m` | depth cua midhip |
| `distance_sam_midhip_euclidean_m` | khoang cach euclidean den midhip |

Dataset runtime hien dung truc tiep:

- `keypoints_3d_cam_m`
- `pred_keypoints_2d_px`
- `bbox_xyxy_normalized`
- `prediction.image_size` hoac `image_size`
- `image_path` chi la hint de tim anh; dataset van tim theo sample id neu can.

## He toa do

Camera frame:

- X duong sang phai.
- Y duong xuong duoi.
- Z duong di xa camera.
- Don vi met.

He qua kiem tra nhanh: trong y_down, `nose.Y < hip.Y` neu nguoi dung thang.
Test `tests/test_dataset.py` kiem tra dieu nay tren 20 mau dau.

## Anh va label hop le

`GTJsonDataset` bo qua mau neu:

- JSON khong doc duoc hoac khong phai dict.
- `status` khac `ok`.
- thieu/khong du 70 keypoint 3D.
- thieu/khong du 70 keypoint 2D.
- thieu image size hop le.
- bbox sai hoac rong.
- khong tim thay anh tu `images`.

## Data backup/combined

`labels_backup` co cung 18,752 JSON. `combined` co `images` va `labels` cung
18,752 mau. Trong code hien tai, config train dung `data/GT/images` va
`data/GT/labels`, khong dung truc tiep `combined`.

Nen xem:

- `data/GT/labels`: label runtime chinh.
- `data/GT/images`: image runtime chinh.
- `labels_backup`: backup.
- `combined`: ban tap hop/duplicate hoac staging, can xac nhan quy tac cap nhat.

