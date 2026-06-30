# Kien Truc Code

## Package `pose24`

`src/pose24` la package runtime chinh. Khi import `pose24`, cac module custom
duoc register vao registry cua MMPose/MMEngine, gom dataset, transform, codec,
model, loss va hook.

Tong so dong cac file chinh da ra soat: khoang 2,474 dong, chua tinh config/test
phu.

## Keypoint schema

File: `src/pose24/keypoints.py`

Bo POSE-24 gom:

- 17 diem COCO body dau tien.
- 7 diem anatomical bo sung: olecranon, cubital fossa, acromion, neck.
- Root joints: `(11, 12)` tuong ung `left_hip`, `right_hip`.
- Skeleton co 24 edge.
- Co `FLIP_INDICES` de doi trai/phai khi horizontal flip.
- Co `JOINT_PARENTS` phuc vu bone/structure loss.

Mapping MHR70 sang POSE-24:

```text
POSE24 idx -> MHR70 idx
0 nose -> 0
1 left_eye -> 1
2 right_eye -> 2
3 left_ear -> 3
4 right_ear -> 4
5 left_shoulder -> 5
6 right_shoulder -> 6
7 left_elbow -> 7
8 right_elbow -> 8
9 left_wrist -> 62
10 right_wrist -> 41
11 left_hip -> 9
12 right_hip -> 10
13 left_knee -> 11
14 right_knee -> 12
15 left_ankle -> 13
16 right_ankle -> 14
17 left_olecranon -> 63
18 right_olecranon -> 64
19 left_cubital_fossa -> 65
20 right_cubital_fossa -> 66
21 left_acromion -> 67
22 right_acromion -> 68
23 neck -> 69
```

## Dataset

File: `src/pose24/datasets/gt_json_dataset.py`

Class: `GTJsonDataset(BaseCocoStyleDataset)`

Expected layout:

```text
data/GT/
├── images/<sample_id>.(jpg|jpeg|png|webp|bmp)
└── labels/<sample_id>.json
```

Dataset doc JSON va tra ve data theo style MMPose:

- `bbox`: normalized bbox trong JSON duoc doi sang pixel `(1,4)`.
- `keypoints`: 2D pixel keypoints, MHR70 -> POSE-24, shape `(1,24,2)`.
- `keypoints_3d`: 3D camera-frame keypoints, MHR70 -> POSE-24, shape `(1,24,3)`.
- `keypoints_visible`: hien tai gan all-one cho 24 diem.
- `img_path`, `bbox_score`, `category_id`, `iscrowd`.

Neu `ann_file` ton tai, dataset doc sample id tu split txt. Neu khong, dataset
quet toan bo `labels/*.json`.

## Transform quan trong

File: `src/pose24/datasets/transforms.py`

Class: `Flip3DKeypoints`

Ly do co transform nay: `RandomFlip` cua mmpose flip 2D keypoints nhung khong
dong bo `keypoints_3d`. Transform nay phai dat ngay sau `RandomFlip` de:

- reorder 3D keypoints theo `flip_indices`;
- mirror truc camera X (`x_right`) bang cach nhan `-1`;
- giu nguyen Y va Z.

Neu bo qua transform nay, cac mau flip se bi sai giam sat left/right depth.

## Codec SimCC 3D

File: `src/pose24/codecs/simcc_3d_label.py`

Class: `SimCC3DLabel`

Y tuong: moi keypoint duoc ma hoa thanh 3 phan phoi rieng biet:

- `keypoint_x_labels`
- `keypoint_y_labels`
- `keypoint_z_labels`

Tham so trong config hien tai:

- `input_size=(288, 384, 288)`
- `simcc_split_ratio=2.0`
- label size: X=576, Y=768, Z=576
- `sigma=(6.0, 6.93, 6.0)`
- `root_index=(11, 12)`
- `z_range=1.5`
- `normalize=False`

Z label duoc tinh root-relative:

```text
root_z = mean(z cua left_hip va right_hip)
z_relative = keypoint_z - root_z
z_bin = (z_relative / z_range + 1) * (input_depth / 2)
```

Decode tra ve:

- `keypoints`: x/y pixel trong input space va z root-relative met.
- `keypoints_simcc`: toa do bin/SimCC da chia ratio.
- `scores`: confidence theo keypoint.

## Model

File chinh:

- `src/pose24/models/pose_estimator.py`
- `src/pose24/models/rtmw3d_head.py`
- `src/pose24/models/loss.py`
- `src/pose24/models/structure_loss.py`

### `TopdownPoseEstimator3D`

Ke thua `TopdownPoseEstimator`, them buoc dua prediction ve camera-frame 3D.

Luong trong `add_pred_to_datasample`:

1. head decode ra keypoints trong input/crop space.
2. doi x/y ve full-image space bang `input_center`, `input_scale`, `input_size`.
3. lay depth = z root-relative + `root_z`.
4. back-project x/y pixel ve camera X/Y bang focal length.

`GTJsonDataset` doc `focal_length_px` + `image_size` rieng cho moi sample (file
`gt_json_dataset.py`, field JSON). Luu y: `focal_length_px` KHONG phai thong
so camera vat ly do duoc -- la gia tri SAM-3D-Body tu uoc luong luc tao
pseudo-GT (`pose3d_source: "sam3d"` trong JSON). RTMPose3D khong tu suy luan
ra focal length, chi nhan gia tri nay nhu hang so da biet de back-project.
Dong goi thanh `camera_param=[{f:[fx,fx], c:[w/2,h/2]}]` -> mmpose tu pack
vao `gt_instances.camera_params` qua `instance_mapping_table` cua
`SimCC3DLabel` (field so it `camera_param` -> so nhieu `camera_params`). Vi
vay nhanh dung `camera_params` per-sample la nhanh thuc te dung cho moi
sample train/val/eval.

Nhanh fallback (dung khi KHONG co `camera_params`, vi du anh upload trong demo
khong co GT) dung gia tri co dinh:

- focal length: `[2074.0, 2074.0]` (median tu GT)
- principal point: `c = [w/2, h/2]` tu `ori_shape=(h,w)` -- luu y `ori_shape`
  la `(H,W)` theo convention mmpose, phai tach ro truoc khi build `c`, khong
  duoc halving truc tiep `ori_shape` (se dao nguoc thanh `(cy,cx)`).

### `RTMW3DHead`

Head nhan 2 feature map tu neck:

- `enc_b`
- `enc_t`

Sau do:

- dung conv + PixelShuffle de tron feature top/bottom;
- flatten feature map;
- qua RTMCCBlock;
- du doan `pred_x`, `pred_y`, `pred_z` bang 3 linear head.

Output dim tu config:

- `cls_x.out_features = 576`
- `cls_y.out_features = 768`
- `cls_z.out_features = 576`

### Loss

`KLDiscretLossWithWeight`:

- ap dung KL discrete loss theo 3 truc x/y/z;
- co target weight rieng, z co the tat neu khong co z label;
- loss name: `loss_kld`.

`Pose3DStructureLoss` (file `src/pose24/models/structure_loss.py`):

Van de can giai: `BoneLoss` ban goc lay toa do bang `argmax` tren SimCC, ham nay
khong co dao ham (non-differentiable) nen gradient khong chay nguoc duoc ve
`cls_x/y/z`. `Pose3DStructureLoss` thay the bang ky thuat **soft-argmax**:

```text
soft_argmax(logits) = sum_i softmax(logits * beta)[i] * i / (L - 1)
```

`softmax` lam toa do tro thanh ky vong co trong so (weighted average), lien tuc
va co dao ham theo logits, nen gradient chay duoc tu loss nguoc ve het 3 head
`cls_x`, `cls_y`, `cls_z`. He so `beta` (mac dinh 10 trong head) lam phan phoi
softmax sac hon, gan voi argmax cung khi beta lon nhung van giu duoc gradient.

Loss gom 3 thanh phan, tat ca tren toa do normalized `[0,1]`, dung masked
Smooth L1 (L1 quanh 0, L2 gan 0, tranh outlier lam loss bung qua manh):

1. **Root loss**: smooth-L1 giua pred va target cua diem root (trung binh
   `left_hip`/`right_hip`, index `(11,12)`). Buoc model dat dung vi tri goc
   cua khung xuong truoc khi xet chi tiet.
2. **Relative loss**: smooth-L1 giua pred va target sau khi da tru di root
   (toa do tuong doi so voi root). Buoc tung khop dung vi tri so voi than
   nguoi, khong phu thuoc vi tri tuyet doi trong anh.
3. **Bone-length loss**: smooth-L1 giua do dai bone du doan va do dai bone
   thuc te, tinh tu cap diem lien ke trong `SKELETON_EDGES`. Buoc ty le cac
   doan xuong (vd dai dui, dai canh tay) giu dung ty le giai phau, tranh
   truong hop khop dung vi tri nhung do dai xuong bi keo dan/co lai bat
   thuong.

Loss name: `loss_struct`. Test `test_struct_loss_gradient_reaches_depth_head`
xac nhan gradient cua `loss_struct` chay duoc den depth head `cls_z` — diem
nay quan trong vi day la phan ma `BoneLoss` ban goc khong lam duoc.

## Visualization hook

File: `src/pose24/engine/hooks.py`

Hook `Pose3DVisualizationHook` xuat anh GT-vs-Pred moi lan validate, mac dinh
`num_samples=4`. Output nam duoi `work_dirs/<run>/vis/epoch_<n>/`.

## Visualization utilities

File: `src/pose24/visualization/draw.py`

Ham chinh `render_comparison` ve:

- panel anh 2D voi GT/pred overlay;
- panel skeleton 3D trong camera frame;
- mau GT/pred rieng de so sanh.

