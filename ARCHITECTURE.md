# RTMPose3D Finetune — Kiến Trúc & Thuật Toán Đầy Đủ

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Dữ liệu & Keypoint Schema](#2-dữ-liệu--keypoint-schema)
3. [Hệ toạ độ camera](#3-hệ-toạ-độ-camera)
4. [Codec SimCC3DLabel](#4-codec-simcc3dlabel)
5. [Kiến trúc model](#5-kiến-trúc-model)
6. [Loss functions](#6-loss-functions)
7. [Soft-Argmax (differentiable decoder)](#7-soft-argmax-differentiable-decoder)
8. [Pipeline huấn luyện](#8-pipeline-huấn-luyện)
9. [Evaluation — MPJPE](#9-evaluation--mpjpe)
10. [Tóm tắt thay đổi so với RTMPose3D gốc](#10-tóm-tắt-thay-đổi-so-với-rtmpose3d-gốc)

---

## 1. Tổng quan hệ thống

```
Input image (RGB)
        │
        ▼
  Data Pipeline
  (crop + augment)
        │
        ▼
  CSPNeXt Backbone   ←── pretrained RTMPose-L
        │
   [enc_b, enc_t]      (stride-16, stride-32 feature maps)
        │
        ▼
  CSPNeXtPAFPN Neck
        │
   [feat_b, feat_t]    (1024-ch at each level)
        │
        ▼
   RTMW3DHead
        │
   (pred_x, pred_y, pred_z)   SimCC logits per keypoint
        │
   ┌────┴────────────────────────┐
   │ Loss path (train)           │ Decode path (val/test)
   │                             │
   │  KLDiscretLoss (x,y,z)      │  argmax → (u,v,d) bins
   │  Pose3DStructureLoss        │  → (x,y,z) camera coords
   │    via soft-argmax          │
   └─────────────────────────────┘
```

---

## 2. Dữ liệu & Keypoint Schema

### 2.1 MHR70 → POSE-24

Dữ liệu GT là skeleton MHR70 (70 keypoints). Ta chọn **24 keypoints** phần body:

| POSE-24 idx | Tên | MHR70 idx |
|:-----------:|-----|:---------:|
| 0  | nose              | 0  |
| 1  | left_eye          | 1  |
| 2  | right_eye         | 2  |
| 3  | left_ear          | 3  |
| 4  | right_ear         | 4  |
| 5  | left_shoulder     | 5  |
| 6  | right_shoulder    | 6  |
| 7  | left_elbow        | 7  |
| 8  | right_elbow       | 8  |
| 9  | left_wrist        | 62 |
| 10 | right_wrist       | 41 |
| 11 | **left_hip** ← root  | 9  |
| 12 | **right_hip** ← root | 10 |
| 13 | left_knee         | 11 |
| 14 | right_knee        | 12 |
| 15 | left_ankle        | 13 |
| 16 | right_ankle       | 14 |
| 17 | left_olecranon    | 63 |
| 18 | right_olecranon   | 64 |
| 19 | left_cubital_fossa | 65 |
| 20 | right_cubital_fossa | 66 |
| 21 | left_acromion     | 67 |
| 22 | right_acromion    | 68 |
| 23 | neck              | 69 |

**Root joints**: keypoint 11 (left_hip) và 12 (right_hip). Pelvis center = mean của hai joint này.

### 2.2 Skeleton Edges (24 bones)

```
Left leg:      15─13─11
Right leg:     16─14─12
Hips:          11─12
Torso:         5─11, 6─12, 5─6
Left arm:      5─7─9,   7─17 (olecranon)
Right arm:     6─8─10,  8─18 (olecranon)
Head:          0─1─3,   0─2─4
Neck:          23─5,    23─6
Acromion:      5─21,    6─22
```

### 2.3 Flip mapping

Horizontal flip: mỗi keypoint bên trái ↔ bên phải tương ứng.
VD: `FLIP_INDICES[5] = 6` (left_shoulder ↔ right_shoulder).

---

## 3. Hệ toạ độ camera

GT JSON dùng hệ `x_right_y_down_z_forward_meters`:

```
        Z (depth, vào màn hình)
       /
      /
     O──────► X (sang phải)
     │
     │
     ▼ Y (xuống)
```

- **Y dương = xuống** (khác với display convention thông thường)
- Đơn vị: **mét**
- Khi visualize 3D: map `(X, Z, −Y)` để người đứng thẳng trên màn hình

**Lỗi Y-inversion trong GT ban đầu**: 18 752 file JSON có trục Y bị đảo ngược (`y_down` bị lưu là `y_up`). Đã sửa bằng nhân tất cả `keypoints_3d[:, 1] *= -1` trước khi train.

---

## 4. Codec SimCC3DLabel

### 4.1 Ý tưởng

**SimCC = Simple Coordinate Classification.** Thay vì hồi quy thẳng toạ độ (1 số thực
khó tối ưu) hoặc dùng heatmap 3D (tốn bộ nhớ O(W×H×D)), SimCC biến **bài toán định vị
thành bài toán phân loại**: chia mỗi trục thành nhiều "bin" (ô) rời rạc, rồi hỏi
"keypoint nằm ở bin nào?". Mỗi keypoint được mã hoá thành **3 phân phối xác suất độc
lập** — một cho mỗi trục:

```
keypoint k  →  (μ_x, μ_y, μ_z)  →  L_x[i] = Gaussian(i; μ_x, σ_x)   (vector dài W_label)
                                     L_y[j] = Gaussian(j; μ_y, σ_y)   (vector dài H_label)
                                     L_z[d] = Gaussian(d; μ_z, σ_z)   (vector dài D_label)
```

**Vì sao tách 3 trục độc lập?**
- Bộ nhớ: `O(K × (W + H + D))` thay vì `O(K × W × H × D)`. Với input này:
  `24 × (576+768+576) = 46k` số, thay vì `24 × 576×768×576 ≈ 6 tỷ` số. Giảm ~130 000 lần.
- Tốc độ: 3 lớp Linear nhỏ thay vì conv 3D nặng.
- Đánh đổi: giả định x, y, z độc lập (mất tương quan giữa các trục) — `Pose3DStructureLoss`
  (§6.2) bù lại bằng cách thêm ràng buộc hình học.

**Vì sao dùng Gaussian label (mềm) thay vì one-hot (cứng)?** Label cứng (chỉ bin đúng = 1,
còn lại = 0) phạt nặng cả khi đoán lệch 1 bin. Gaussian "mềm" cho bin lân cận cũng có giá
trị > 0 → mô hình học được "gần đúng vẫn tốt", gradient mượt, hội tụ nhanh hơn. `σ` quyết
định độ "mềm": σ lớn = khoan dung hơn nhưng kém sắc nét.

### 4.2 Tham số

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `input_size` | `(288, 384, 288)` | W × H × D (px) |
| `simcc_split_ratio` | `2.0` | label bins = pixel × ratio |
| `sigma` | `(6.0, 6.93, 6.0)` | Gaussian σ theo (x, y, z) |
| `z_range` | `1.5` | nửa khoảng z (mét): z ∈ [−1.5, +1.5] |
| `root_index` | `(11, 12)` | left_hip + right_hip |

Label size:
```
W_label = W × ratio = 288 × 2 = 576 bins
H_label = H × ratio = 384 × 2 = 768 bins
D_label = D × ratio = 288 × 2 = 576 bins
```

### 4.3 Encode — Bước 1: Chuẩn hoá Z

Cho mẫu thứ `n`, `k`-th keypoint:

```
root_z_n = mean(keypoints_3d[n, {11,12}, 2])          # pelvis Z

z_relative_nk = keypoints_3d[n, k, 2] − root_z_n     # root-relative depth

z_bin_nk = (z_relative_nk / z_range + 1) × (D / 2)
         = (z_relative_nk / 1.5 + 1) × 144
```

Mapping:
- `z_relative = −1.5 m` → bin `0`
- `z_relative =  0.0 m` → bin `144`
- `z_relative = +1.5 m` → bin `288`

### 4.4 Encode — Bước 2: Gaussian label

Toạ độ pixel → bin:
```
μ_x = round(x_pixel × 2.0)
μ_y = round(y_pixel × 2.0)
μ_z = z_bin   (từ bước trên)
```

Gaussian label (nếu keypoint visible):
```
L_x[i] = exp(−(i − μ_x)² / (2σ_x²)),  i = 0..W_label−1
L_y[j] = exp(−(j − μ_y)² / (2σ_y²)),  j = 0..H_label−1
L_z[d] = exp(−(d − μ_z)² / (2σ_z²)),  d = 0..D_label−1
```

Nếu `normalize=False` (config hiện tại): giữ nguyên peak = 1.
Nếu `normalize=True`: chia cho `σ√(2π)`.

### 4.5 Decode — SimCC argmax

Trong val/inference:
```
û = argmax_i L_x[i]
v̂ = argmax_j L_y[j]
d̂ = argmax_d L_z[d]

x_pixel = û / simcc_split_ratio = û / 2
y_pixel = v̂ / simcc_split_ratio = v̂ / 2
z_rel   = (d̂ / (D/2) − 1) × z_range
        = (d̂ / 144 − 1) × 1.5  [mét]
```

### 4.6 Ví dụ số end-to-end (1 keypoint)

Giả sử `left_knee` của 1 người: toạ độ pixel `(140.0, 250.0)`, depth thật `z = 2.90 m`,
pelvis ở `root_z = 3.10 m`.

**ENCODE (chuẩn bị label cho training):**
```
1. z root-relative:  z_rel = 2.90 − 3.10 = −0.20 m   (đầu gối gần camera hơn pelvis 20cm)
2. z → bin:          μ_z = (−0.20/1.5 + 1) × 144 = (−0.1333 + 1) × 144 ≈ 124.8 → 125
3. x,y → bin:        μ_x = round(140.0 × 2) = 280
                     μ_y = round(250.0 × 2) = 500
4. Gaussian:         L_x = đỉnh tại bin 280 (σ=6),  L_z = đỉnh tại bin 125 (σ=6) ...
```

**DECODE (lấy lại toạ độ từ output model):**
```
Model xuất pred_z (576 bins). argmax rơi vào bin 125:
  z_rel = (125/144 − 1) × 1.5 = (0.868 − 1) × 1.5 = −0.198 m   ✓ khớp lại
  x_pixel = 280/2 = 140,  y_pixel = 500/2 = 250                 ✓
```

Sai số lượng tử của z: 1 bin = `1.5/144 × 2 ≈ 0.0208 m` ≈ **2 cm/bin**. Đây là giới hạn
độ phân giải depth của codec — muốn mịn hơn phải tăng `D` hoặc giảm `z_range`.

> ⚠️ **Quan trọng:** x, y vào model là **toạ độ pixel ảnh**, KHÔNG phải mét. Chỉ z là
> mét (root-relative). Việc quy đổi pixel → mét xảy ra **sau** decode, bằng back-projection
> với focal length (xem §9 và §10.9). Đây là lý do x,y học được từ ảnh còn z thì khó hơn.

---

## 5. Kiến trúc model

### 5.1 Backbone: CSPNeXt-L

Pretrained: `rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192`

| Block | Channels | Stride |
|-------|----------|--------|
| Stem  | 64       | /4     |
| Stage 1 | 128    | /8     |
| Stage 2 | 256    | /16    |
| Stage 3 | 512    | /32    |
| Stage 4 | 1024   | /32    |

Output: `enc_b` (256-ch, /16), `enc_t` (1024-ch, /32)

### 5.2 Neck: CSPNeXtPAFPN

```
in_channels  = [256, 512, 1024]
out_channels = None  (giữ nguyên)
out_indices  = (1, 2)   →  feat_b (512-ch), feat_t (1024-ch)
num_csp_blocks = 2
```

Feature map size (input 288×384):
```
feat_b: 18×24×512   (stride-16: 288/16=18, 384/16=24)
feat_t:  9×12×1024  (stride-32: 288/32=9,  384/32=12)
```

### 5.3 Head: RTMW3DHead

Head nhận 2 feature map từ neck và biến chúng thành 3 phân phối SimCC. Có 3 khối:
multi-scale fusion → GAU → SimCC projection.

#### Vì sao cần multi-scale fusion (2 mức phân giải)?

Neck cho ra 2 feature map ở 2 độ phân giải:
- `feat_t` (9×12, stride-32) — **thô, ngữ cảnh rộng**: "đây là vai hay hông?", quan hệ
  toàn thân. Nhiều ngữ nghĩa nhưng định vị không sắc.
- `feat_b` (18×24, stride-16) — **mịn, định vị chính xác**: cạnh/biên rõ, giúp đặt
  keypoint đúng pixel. Nhưng "nông" về ngữ nghĩa.

Head kết hợp cả hai: dùng `feat_t` cho ngữ cảnh + `feat_b` cho độ chính xác → keypoint
vừa đúng khớp vừa đúng vị trí. `PixelShuffle(2)` phóng `feat_t` (1024-ch) lên cùng độ phân
giải 18×24 (đổi channel→spatial: 1024→256, HxW ×2) để ghép được với `feat_b`.

#### Multi-scale feature fusion

```
enc_t (1024, 9×12)
  │
  ├─► final_layer [Conv2d 1024→24, k=7, BN, ReLU]
  │       │
  │   flatten → (N, 24, 108)
  │       │
  │   mlp [ScaleNorm(108) → Linear(108, 128)]
  │       │
  │   feats_t (N, 24, 128)
  │
  └─► PixelShuffle(2) → (256, 18×24)
          │
      conv_dec [Conv2d 256→256, k=7, BN, ReLU]
          │
      cat với enc_b (512+256=768, 18×24)
          │
      final_layer2 [Conv2d 768→24, k=7, BN, ReLU]
          │
      flatten → (N, 24, 432)
          │
      mlp2 [ScaleNorm(432) → Linear(432, 128)]
          │
      feats_b (N, 24, 128)

feats = cat([feats_t, feats_b], dim=2) → (N, 24, 256)
```

**Flatten dims**:
```
fine (feat_t): 9 × 12 = 108
coarse (feat_b): 18 × 24 = 432
```

#### GAU (Gated Attention Unit)

```
feats (N, 24, 256)
   │
RTMCCBlock(
  in=24, hidden=256, out=256,
  s=128,               ← attention head dimension
  expansion_factor=2,  ← FFN expansion
  act_fn='SiLU',
  use_rel_bias=False,
  pos_enc=False,
  dropout=0.1
)
   │
feats_gau (N, 24, 256)
```

RTMCCBlock dùng **Gated Linear Attention** (GAU = Gated Attention Unit):
```
Z = Linear(feats)  →  (N, K, d_head)
gate = sigmoid(Linear(feats))
out = gate ⊙ Z         (⊙ = nhân từng phần tử)
```

**GAU làm gì ở đây?** Mỗi token là 1 keypoint (24 token, mỗi token vector 256-d).
Self-attention giữa các keypoint cho phép chúng "trao đổi thông tin": vị trí cổ tay
phụ thuộc khuỷu tay, đầu gối phụ thuộc hông... Cổng `gate` (sigmoid) lọc bớt nhiễu,
chỉ giữ tín hiệu hữu ích. GAU rẻ hơn self-attention chuẩn (gộp attention + FFN làm một).
`use_rel_bias=False, pos_enc=False` vì thứ tự keypoint cố định (không cần mã hoá vị trí).

#### SimCC Projection — 3 đầu phân loại

```
cls_x: Linear(256, 576)   →  pred_x (N, 24, 576)   ← logits cho 576 bin trục x
cls_y: Linear(256, 768)   →  pred_y (N, 24, 768)   ← 768 bin trục y
cls_z: Linear(256, 576)   →  pred_z (N, 24, 576)   ← 576 bin trục z (depth)
```

Mỗi `cls_*` là 1 lớp Linear biến vector 256-d của mỗi keypoint thành điểm số (logits)
cho từng bin. softmax(logits) = phân phối xác suất "keypoint ở bin nào". Lúc train so
với Gaussian GT (qua KL loss); lúc infer lấy argmax → toạ độ (§4.5).

**Tổng tham số head** ≈ 1.2M (không tính backbone). Lưu ý 3 đầu này chính là nơi gradient
của cả 2 loss đổ về (KL trực tiếp; structure loss qua soft-argmax — §7).

#### Toàn bộ shape qua head (input 288×384, batch N)
```
neck → feat_t (N,1024,9,12),  feat_b (N,512,18,24)
  → fusion → feats (N, 24, 256)
  → GAU    → feats_gau (N, 24, 256)
  → cls_x/y/z → pred_x (N,24,576), pred_y (N,24,768), pred_z (N,24,576)
```

---

## 6. Loss functions

Training dùng **2 loss bổ sung nhau** (xem [config](src/pose24/configs/rtmw3d_l_finetune_pose24.py)):

| Loss | Không gian | Vai trò | Vì sao cần |
|------|-----------|---------|-----------|
| `KLDiscretLossWithWeight` | bin SimCC (rời rạc) | **Driver chính** — định vị từng keypoint | Tín hiệu mạnh, sắc, hội tụ nhanh |
| `Pose3DStructureLoss` | toạ độ [0,1] (liên tục) | **Prior hình học** — quan hệ giữa các khớp | KL bỏ qua tương quan giữa khớp |

KL giỏi đặt từng điểm đúng chỗ nhưng "mù" về cấu trúc (có thể đặt đầu gối đúng pixel mà
cẳng chân dài bất thường). Structure loss bù đúng chỗ đó: ép độ dài xương, vị trí tương
đối hợp lý. Hai loss kéo mô hình từ 2 góc khác nhau → vừa chính xác vừa giải phẫu đúng.

### 6.1 KLDiscretLossWithWeight

Loss chính, tối ưu **phân phối SimCC** trên từng trục (x, y, z).

#### Công thức

```
criterion = KL divergence với label_softmax=True:
  p̂_k = softmax(pred_k × β),  β = 10
  p_k  = softmax(target_k × β)  (nếu label_softmax=True)

loss_axis = Σ_i p_k[i] × log(p_k[i] / p̂_k[i])    [per keypoint]
          ≈ cross_entropy(pred_k × β, p_k)
```

#### Forward

```
For each axis a ∈ {x, y, z}:
    weight_a:  (N, K)  — per-axis keypoint weight
               (weight_z = 0 nếu không có z label)

    pred_a reshape (N×K, L_a)
    target_a reshape (N×K, L_a)
    weight_a reshape (N×K,)

    t_loss = KL(pred_a, target_a)  ×  weight_a   →  (N×K,)

total_loss = sum(t_loss_x + t_loss_y + t_loss_z) / K
```

**Ý nghĩa**: khuyến khích phân phối predicted SimCC trùng với Gaussian GT.

---

### 6.2 Pose3DStructureLoss (custom, theia-inspired)

Loss bổ sung, tối ưu **hình học 3D** trong không gian toạ độ normalised [0, 1].

#### Động lực

`KLDiscretLoss` tối ưu từng keypoint độc lập → không encode quan hệ spatial giữa các joint.
`Pose3DStructureLoss` thêm 3 prior về cấu trúc cơ thể người.

#### Đầu vào

```
pred   (N, K, 3)  — soft-argmax coords trong [0, 1]
target (N, K, 3)  — GT soft-argmax coords (detach)
weight (N, K)     — visibility mask (0 hoặc 1)
```

#### Bước 1: Root term

```
pelvis_pred   = mean(pred[:, {11,12}, :],   dim=1)   →  (N, 1, 3)
pelvis_target = mean(target[:, {11,12}, :], dim=1)   →  (N, 1, 3)

loss_root = smooth_l1(pelvis_pred, pelvis_target, β=0.05)
```

Không masked (pelvis luôn visible trong data).

Smooth-L1 (Huber loss):
```
smooth_l1(x, β) = { x²/(2β)       nếu |x| < β
                  { |x| − β/2     nếu |x| ≥ β
```

#### Bước 2: Relative term

```
pred_rel   = pred   − pelvis_pred     (N, K, 3)
target_rel = target − pelvis_target   (N, K, 3)

loss_rel = masked_smooth_l1(pred_rel, target_rel, weight, β=0.05)
```

Masked smooth-L1:
```
loss_elem = smooth_l1(pred_rel[n,k,:], target_rel[n,k,:], β)  →  (N, K, 3)
loss_rel  = Σ_{n,k} weight[n,k] × Σ_d loss_elem[n,k,d]
          / max(Σ_{n,k} weight[n,k] × 3, 1)
```

#### Bước 3: Bone term

Với mỗi edge `(s, e)` trong `SKELETON_EDGES` (24 edges):

```
pred_bone_len   = ‖pred_rel[:, s, :]   − pred_rel[:, e, :]‖₂    (N, 24)
target_bone_len = ‖target_rel[:, s, :] − target_rel[:, e, :]‖₂  (N, 24)

edge_mask = weight[:, s] × weight[:, e]   (AND của 2 endpoints)

loss_bone = masked_smooth_l1(pred_bone_len, target_bone_len, edge_mask, β=0.05)
```

#### Tổng loss

```
L_struct = loss_weight × (
    root_weight     × loss_root   +    [1.0]
    relative_weight × loss_rel    +    [2.0]
    bone_weight     × loss_bone        [0.5]
)
         = 1.0 × (1.0 × L_root + 2.0 × L_rel + 0.5 × L_bone)
```

#### Tổng loss huấn luyện

```
L_total = L_KLD + L_struct
        ≈ 0.55 + 0.37  (epoch 1, batch đầu)
        → 0.28 + 0.06  (epoch 3)
```

#### ⚠️ Hạn chế đã biết: trộn đơn vị trong không gian [0,1]

`Pose3DStructureLoss` hiện tính trong không gian **soft-argmax [0,1]** (mỗi trục chia cho
chiều dài bin của nó). Hệ quả: 1 đơn vị normalised của mỗi trục ứng với khoảng cách vật lý
**khác nhau**:

| Trục | 1.0 normalized = | ≈ mét (ở depth ~2.5m, f=2074) |
|------|------------------|-------------------------------|
| x | 288 px | ~0.35 m |
| y | 384 px | ~0.46 m |
| z | z_range×2 = 3.0 m | **3.0 m** |

→ Trong `bone_len = √(dx²+dy²+dz²)`, thành phần **z lấn át** x,y (~8×). Bone/relative loss
vì thế thiên về depth, ít ràng buộc hình học ngang/dọc. theia tránh được vì tính trong
**mét thật**. Hướng nâng cấp (tuỳ chọn): back-project soft-argmax → mét rồi mới tính
structure loss (đã phân tích, chưa áp dụng — giữ bản [0,1] cho đơn giản và ổn định).

---

## 7. Soft-Argmax (differentiable decoder)

### 7.1 Vấn đề với argmax thông thường

```
x̂ = argmax_i pred_x[k, i]
```

`argmax` không có gradient → `∂x̂/∂pred_x = 0` everywhere → loss dùng `x̂` không backprop được!

### 7.2 Soft-argmax

Thay thế bằng **kỳ vọng có trọng số** sau khi sharpening:

```
p_i = softmax(pred_x[k, :] × β),   β = 10.0   (soft_argmax_beta)
x̂_soft = Σ_i p_i × i
```

Normalize về [0, 1]:
```
x̂_norm = x̂_soft / (L − 1)     L = W_label = 576
```

### 7.3 Gradient flow

```
∂x̂_norm/∂pred_x[k, i] = 1/(L-1) × Σ_j (∂p_j/∂logit_i) × j
                        ≠ 0  (vì softmax có gradient liên tục)
```

→ Gradient lan ngược qua `cls_x`, `cls_y`, `cls_z` về backbone.

### 7.4 Xử lý Z không có label

Các sample không có z label (2D-only data):
```
with_z_label[n] = False  →  z̃_norm[n, :] = 0.5  (trung tâm, không contribute loss)
```

---

## 8. Pipeline huấn luyện

### 8.1 Data augmentation (train)

```
LoadImage
  │
GetBBoxCenterScale          ← crop region từ bbox
  │
RandomFlip(horizontal, p=0.5)
  │
Flip3DKeypoints             ← sync keypoints_3d với 2D flip (swap L/R + mirror X)
  │                            BẮT BUỘC — xem §10.8
RandomHalfBody              ← đôi khi chỉ crop upper/lower body
  │
RandomBBoxTransform
  scale ∈ [0.6, 1.4]
  rotate ∈ [−80°, +80°]
  │
TopdownAffine(288×384)      ← warp về fixed size
  │
YOLOXHSVRandomAug           ← HSV color jitter
  │
Albumentation:
  Blur(p=0.1)
  MedianBlur(p=0.1)
  CoarseDropout(p=1.0,       ← random occlusion patch
    max_holes=1, 20-40% area)
  │
GenerateTarget(encoder=SimCC3DLabel)
  │
PackPoseInputs
```

**Từng augment làm gì** (mục tiêu chung: model bền với biến thể thật, không học thuộc):

| Transform | Tác dụng | Vì sao |
|-----------|----------|--------|
| `GetBBoxCenterScale` | bbox → tâm + scale crop | chuẩn bị cho affine |
| `RandomFlip` | lật ngang ảnh + swap khớp L/R | tăng đôi dữ liệu, học đối xứng |
| `Flip3DKeypoints` | sync keypoints_3d với flip | **bắt buộc** — nếu thiếu → loạn depth L/R (§10.8) |
| `RandomHalfBody` | đôi khi chỉ crop nửa thân | bền với ảnh bị cắt cụt |
| `RandomBBoxTransform` | scale/xoay ngẫu nhiên | bền với khoảng cách & nghiêng camera |
| `TopdownAffine` | warp về 288×384 cố định | model cần input size cố định |
| `YOLOXHSVRandomAug` | đổi màu/sáng (HSV) | bền với điều kiện ánh sáng |
| `Albumentation` Blur/Dropout | mờ + che ngẫu nhiên 20-40% | bền với mờ + che khuất (occlusion) |
| `GenerateTarget` | encode keypoint → SimCC label | tạo nhãn huấn luyện (§4) |
| `PackPoseInputs` | gói thành tensor + data sample | format cho model |

**Stage 2** (10 epoch cuối, `max_epochs − stage2_num_epochs`): tắt YOLOXHSVRandomAug,
tăng random scale lên [0.5, 1.5] + rotate 90°. Ý tưởng: cuối quá trình, giảm jitter màu
(model đã quen) nhưng tăng biến đổi hình học để tinh chỉnh độ bền. `Flip3DKeypoints` giữ.

### 8.2 Optimizer & Learning-rate schedule

```
AdamW(lr=5e-5, weight_decay=0.05)

Warm-up: LinearLR  start_factor=0.01 → 1.0,  500 iterations đầu
Decay:   CosineAnnealingLR, eta_min = lr×0.05,  begin=epoch 20 → end=epoch 100
         T_max = end − begin = 80  (LR chạm đáy đúng lúc kết thúc)
```

**Vì sao có warmup?** Đầu training trọng số head khởi tạo ngẫu nhiên, gradient lớn/nhiễu.
Nếu LR cao ngay → cập nhật giật, dễ phá hỏng backbone pretrained. Warmup tăng LR từ
`0.01×lr` lên `lr` trong 500 iter đầu để "khởi động êm".

**Vì sao có cosine decay?** Sau warmup, LR cao giúp học nhanh; nhưng gần điểm tối ưu cần
LR nhỏ để "lắng" vào, không dao động. Cosine giảm LR mượt theo hình cos từ `lr` xuống
`eta_min`:
```
LR(t) = eta_min + (lr − eta_min) × (1 + cos(π·t/T_max)) / 2
```

**Decay SỚM (begin=20) — quan trọng:** Phiên bản đầu để decay bắt đầu ở `max_epochs/2`
(epoch 100), nhưng mô hình hội tụ từ ~epoch 25-30 và **LR cao phẳng quá lâu đẩy nó vào
overfit** (xem §8.5). Đổi `begin=20` để LR giảm ngay sau khi hội tụ → tinh chỉnh nhẹ thay
vì "nghiền" tiếp.

### 8.2.1 EMA (Exponential Moving Average)

```
EMAHook(ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True)
```
Duy trì 1 bản sao trọng số "trung bình trượt" song song với trọng số đang train. Mỗi step:
`θ_ema = (1−m)·θ_ema + m·θ`. Bản EMA mượt hơn, ít nhiễu hơn bản gốc → thường cho val tốt
hơn. Checkpoint lưu cả hai; eval dùng bản EMA.

### 8.3 Hyperparameters

| Config | Giá trị |
|--------|---------|
| Epochs | 100 (`max_epochs`) |
| base_lr | 5e-5 |
| Batch size | 32 |
| num_workers | 8 |
| val_interval | mỗi 5 epoch |
| AMP | bật (`AmpOptimWrapper`) — `--amp` |
| EMA | momentum=0.0002 |
| Max checkpoints | 3 (best + 2 gần nhất) |

### 8.4 Checkpoint strategy

```
save_best = 'MPJPE'   ← chỉ lưu best_ khi MPJPE THẤP HƠN kỷ lục cũ
rule = 'less'         ← MPJPE càng nhỏ càng tốt
max_keep_ckpts = 3    ← chỉ giữ 3 checkpoint định kỳ gần nhất
```
Vì `val_interval=5`, MPJPE chỉ được tính (và `best_` chỉ cập nhật) mỗi 5 epoch.

### 8.5 Bài học overfit (lần train 200-epoch đầu)

Lần train đầu (200 epoch, lr 1e-4, decay từ epoch 100) cho thấy chữ ký overfit kinh điển:

| Epoch | TRAIN loss | VAL MPJPE | |
|-------|-----------|-----------|---|
| 30 | 0.198 | **0.3752** | ✅ best |
| 100 | 0.160 | 0.378 | |
| 200 | **0.105** ↓ | **0.3855** ↑ | ❌ |

Train loss giảm 47% nhưng VAL MPJPE **tăng** từ epoch 30 → hai đường phân kỳ. Mô hình học
thuộc train set thay vì tổng quát hoá. Nguyên nhân: LR cao phẳng quá lâu + 16k ảnh là ít
cho model lớn + không early-stop.

**Khắc phục (config hiện tại):** `max_epochs=100`, `lr=5e-5`, **decay sớm từ epoch 20**
(§8.2). Tuỳ chọn thêm: `weight_decay 0.05→0.1`, `dropout 0.1→0.2`, `EarlyStoppingHook`.

---

## 9. Evaluation — MPJPE

### 9.1 Định nghĩa

**MPJPE** (Mean Per-Joint Position Error):
```
MPJPE = (1/N) × (1/K) × Σ_n Σ_k w_{n,k} × ‖p̂_{n,k} − p_{n,k}‖₂
```

Đơn vị: **mm** (metric 3D, camera space).

### 9.2 Aligned MPJPE (P-MPJPE)

Căn chỉnh rigid (Procrustes alignment) trước khi đo:
```
p̂'_{n} = R × p̂_{n} × s + t    (R: rotation, s: scale, t: translation)
P-MPJPE = MPJPE(p̂'_{n}, p_{n})
```

P-MPJPE loại bỏ lỗi global rotation/scale → đo độ chính xác hình học tương đối.

### 9.3 Chuẩn hoá gốc

Trước khi tính MPJPE, cả pred và GT đều được trừ pelvis:
```
p̂_rel_{n,k} = p̂_{n,k} − mean(p̂_{n, {11,12}})
p_rel_{n,k}  = p_{n,k}  − mean(p_{n, {11,12}})

MPJPE = mean over visible joints of ‖p̂_rel − p_rel‖₂
```

### 9.4 Training-time MPJPE proxy

Trong `loss()`, tính thêm `mpjpe` ở **SimCC space** (bins, không phải mm) làm accuracy monitor:
```
pred_coord = argmax(pred_simcc) / simcc_split_ratio   [pixels]
gt_coord   = argmax(gt_simcc)   / simcc_split_ratio   [pixels]
mpjpe_proxy = mean ‖pred_coord − gt_coord‖₂           [pixels]
```

Giá trị này có đơn vị pixel, dùng để theo dõi xu hướng — không phải metric chính.

---

## 10. Tóm tắt thay đổi so với RTMPose3D gốc

### 10.1 Module ported từ `clone_code/` (không sửa logic)

| Module | File gốc | File mới |
|--------|----------|----------|
| `SimCC3DLabel` | `clone_code/pose3d/rtmpose3d/codec` | `src/pose24/codecs/simcc_3d_label.py` |
| `RTMW3DHead` | `clone_code/pose3d/rtmpose3d/head` | `src/pose24/models/rtmw3d_head.py` |
| `KLDiscretLossWithWeight` | `clone_code/pose3d/rtmpose3d/loss` | `src/pose24/models/loss.py` |
| `TopdownPoseEstimator3D` | `clone_code/pose3d/rtmpose3d/estimator` | `src/pose24/models/pose_estimator.py` |

### 10.2 Thay đổi trong `SimCC3DLabel.encode()`

**Thêm `lifting_target_visible`** vào return dict:
```python
# Cần thiết cho SimpleMPJPE.process()
lifting_target_visible = keypoints_visible.copy()
```

Không có field này → `KeyError` khi val.

### 10.3 Thay đổi trong `GTJsonDataset` — registry fix

Pipeline trong stock mmpose dùng `mmengine.Compose` → tìm transform trong scope `mmengine::transform` (không thấy `LoadImage` của mmpose).

**Fix**: build pipeline thủ công dùng `mmpose.registry.TRANSFORMS`:
```python
super().__init__(pipeline=[], ...)   # tắt auto-build
self.pipeline = Compose(
    [TRANSFORMS.build(t) if isinstance(t, dict) else t
     for t in pipeline]
)
```

### 10.4 Thêm mới: `Pose3DStructureLoss`

Thay thế `BoneLoss` từ clone_code (argmax-based, zero gradient).

| | BoneLoss (cũ) | Pose3DStructureLoss (mới) |
|--|--|--|
| Coordinates | argmax bins (non-diff) | soft-argmax [0,1] (diff) |
| Gradient | **không có** | **có** |
| Terms | bone length only | root + relative + bone |
| Masking | không | theo visibility |
| Units | bins (~480) | normalised [0,1] |
| Typical value | **482** (không giảm) | **~0.37** (giảm từ epoch 1) |

### 10.5 Thêm mới: `_soft_decode()` trong `RTMW3DHead`

Khi `loss_.requires_coords == True`:
```python
pred_coords = self._soft_decode(pred_x, pred_y, pred_z, with_z_labels)
gt_coords   = self._soft_decode(gt_x,   gt_y,   gt_z,   with_z_labels).detach()
loss = loss_(pred_coords, gt_coords, keypoint_weights_)
```

`gt_coords` được `.detach()` — GT không cần gradient.

### 10.6 Thêm mới: `Pose3DVisualizationHook`

Sau mỗi val epoch, render 4 samples:
- **Trái**: ảnh gốc với 2D keypoints overlay (GT=xanh, Pred=đỏ)
- **Phải**: 3D skeleton comparison (GT=xanh, Pred=đỏ, view từ phía trước)

Lưu tại: `work_dirs/pose24/vis/epoch_{N}/sample_{i}.png`

3D display convention: map `(X, Z, −Y)` để người đứng thẳng (Y-down camera → Y-up display).

### 10.7 Keypoints: 24 thay vì 133/17

Stock RTMPose3D dùng 133-keypoint wholebody hoặc 17-keypoint body.
Project này dùng 24-keypoint custom (17 COCO body + 7 anatomical: olecranon, cubital fossa, acromion, neck).

### 10.8 Thêm mới: `Flip3DKeypoints` — sync keypoints_3d khi flip

**Triệu chứng (bản train 100-epoch đầu):** 2D overlay khớp ảnh, nhưng 3D skeleton lật
trái/phải theo chiều sâu (chân trái GT giơ ra trước → model đặt depth vào chân phải).

**Nguyên nhân:** mmpose `RandomFlip` chỉ flip `results['keypoints']` (2D: mirror x +
swap L/R qua `flip_indices`), **không đụng `keypoints_3d`**. Codec lấy z-label từ
`keypoints_3d[...,2]` theo cùng thứ tự joint đã swap → với ~50% sample bị flip lúc
train, depth của trái/phải bị gán ngược cho nhau. Chỉ z bị ảnh hưởng (x,y lấy từ 2D
đúng) nên 2D nhìn vẫn đúng.

**Fix:** custom transform [`Flip3DKeypoints`](src/pose24/datasets/transforms.py) đặt
ngay sau `RandomFlip` (xem §8.1), đọc cờ `flip`/`flip_direction`/`flip_indices` mà
RandomFlip ghi vào results rồi áp dụng cho `keypoints_3d`:
```
khi flip horizontal:
  1. reorder joint theo flip_indices (swap L↔R)
  2. negate camera X (x_right → mirror);  Y, Z giữ nguyên
```
Kiểm chứng: sau flip, `new_left_hip.z == old_right_hip.z` và `new_left_hip.x ==
−old_right_hip.x`. Regression test: [tests/test_flip3d.py](tests/test_flip3d.py).

⚠️ Bug này làm **hỏng trọng số** (model học sai L/R suốt training) → **bắt buộc train
lại**, không sửa được ở post-process.

### 10.9 Sửa focal length fallback: `f = 1145 → 2074`

**Triệu chứng:** MPJPE ≈ 746 mm rất cao, nhưng P-MPJPE ≈ 91 mm tốt. Gap lớn = lỗi
scale thuần (Procrustes của P-MPJPE khử scale nên không thấy). Trên demo: skeleton
Pred to gấp ~2x GT.

**Nguyên nhân:** `add_pred_to_datasample` back-project pixel → camera space bằng
`X_cam = (u − cx)/f · Z_cam`. `GTJsonDataset` không có `camera_params` per-sample nên
rơi vào nhánh fallback dùng `f = 1145` (default RTMPose3D). Nhưng SAM-3D-Body tạo GT
với `f ≈ 2074 px` (median, recover từ tương ứng 2D↔3D). f nhỏ hơn 1.8x → X,Y back-proj
lớn hơn 1.8x.

**Fix:** fallback ở [pose_estimator.py](src/pose24/models/pose_estimator.py) đổi sang
`f = 2074`. Đây là bước **decode/eval**, không nằm trong forward/loss/backprop →
**không cần train lại** cho riêng lỗi này (chỉ ảnh hưởng MPJPE và visualization).

| Bug | Ảnh hưởng | Cần train lại? |
|-----|-----------|:--------------:|
| 10.8 Flip L/R | trọng số học sai depth L/R | **Có** |
| 10.9 Focal length | chỉ bước decode → MPJPE/scale | Không |

---

## Quick reference — dimensions

```
Image input:      (N, 3, 384, 288)     [H×W after TopdownAffine]
Backbone out:     enc_b (N, 512, 24, 18)
                  enc_t (N, 1024, 12, 9)
Head feats:       (N, 24, 256)         [24 keypoints, 256-dim each]
SimCC logits:     pred_x (N, 24, 576)
                  pred_y (N, 24, 768)
                  pred_z (N, 24, 576)
Soft-argmax out:  (N, 24, 3)           [x,y,z in [0,1]]
GT labels:        keypoint_x_labels (N, 24, 576)
                  keypoint_y_labels (N, 24, 768)
                  keypoint_z_labels (N, 24, 576)
                  keypoint_weights  (N, 24)
                  lifting_target    (N, 24, 3)  [camera coords, metres]
```
