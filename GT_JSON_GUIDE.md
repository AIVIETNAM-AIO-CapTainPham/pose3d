# Hướng dẫn đọc file JSON Ground Truth (Pose 3D)

Mỗi file trong `data/GT/labels/` tương ứng với một ảnh. Dưới đây là giải thích từng field,
lấy ví dụ từ file `0a5wL8Bsnx.json`.

---

## Hệ tọa độ camera

```
camera_coordinate_system: "x_right_y_down_z_forward_m"
```

Đây là hệ tọa độ **camera chuẩn** trong computer vision:

```
         ● Camera
         │
    ─────┼──────→ X (sang phải)
         │
         ↓ Y (xuống dưới)

         Z (đi vào màn hình, ra xa camera)
```

- **X**: dương → sang phải
- **Y**: dương → xuống dưới (ngược với toán học thông thường)
- **Z**: dương → xa camera hơn (độ sâu)
- **Đơn vị**: mét

**Hệ quả:** Nếu người đứng thẳng nhìn vào camera:
- Nose có Y **âm hoặc nhỏ** hơn hip (vì nose ở trên → ít "xuống dưới" hơn)
- Z của tất cả keypoint xấp xỉ bằng khoảng cách người đến camera

---

## Các field chính

### `keypoints_3d_cam_m` — Keypoint 3D trong camera frame *(field quan trọng nhất)*

```json
"keypoints_3d_cam_m": [
  [-0.033, -0.083,  1.922],   // keypoint 0: nose
  ...                          // 70 keypoints tổng cộng
  [-0.076,  0.520,  2.062],   // keypoint 10: right_hip
]
```

- Shape: **(70, 3)** — 70 keypoints theo schema MHR70, mỗi keypoint là `[X, Y, Z]`
- Đơn vị: **mét**, trong hệ tọa độ camera
- Đây là field được dùng trực tiếp để train

**Đọc ví dụ trên:**
- Nose: X=-0.033 (hơi lệch trái), Y=-0.083 (0.083m **phía trên** trục quang học camera), Z=1.922m (cách camera ~1.9m)
- Right hip: X=-0.076, Y=0.520 (0.520m **phía dưới** trục quang học camera), Z=2.062m

"Trục quang học camera" là đường Y=0 — tưởng tượng như đường nằm ngang đi qua tâm ống kính.
Nose có Y=-0.083, hip có Y=+0.520. Trong hệ y_down, Y nhỏ hơn = cao hơn trong thực tế,
nên nose (-0.083) **cao hơn** hip (+0.520) → đúng về giải phẫu ✓

---

### `keypoints_3d_root_midhip_m` — Keypoint 3D tương đối so với gốc midhip

```json
"keypoints_3d_root_midhip_m": [
  [-0.019, -0.601, -0.173],   // nose, tính từ điểm giữa 2 hip
  ...
]
```

- Shape: **(70, 3)**, cùng đơn vị mét
- Gốc tọa độ (0,0,0) = điểm giữa left_hip và right_hip
- Dùng để train phần **cấu trúc tư thế** (pose structure), độc lập với vị trí trong không gian

**Đọc ví dụ:** Nose ở -0.601m theo Y tính từ hip → tức là nose cao hơn hip khoảng 60cm ✓

**Quan hệ:**
```
keypoints_3d_root_midhip_m = keypoints_3d_cam_m - midhip_translation_cam_m
```

---

### `midhip_translation_cam_m` — Vị trí điểm giữa 2 hip trong camera frame

```json
"midhip_translation_cam_m": [-0.013, 0.518, 2.095]
```

- 1 vector `[X, Y, Z]` duy nhất
- Là trung bình cộng của left_hip và right_hip trong camera frame
- Dùng làm **gốc tọa độ** cho `keypoints_3d_root_midhip_m`

**Đọc:** Điểm giữa 2 hip nằm ở: hơi lệch trái (-0.013m), 0.518m phía dưới trục quang học camera, cách camera 2.095m

---

### `distance_sam_midhip_z_m` và `distance_sam_midhip_euclidean_m` — Khoảng cách người đến camera

```json
"distance_sam_midhip_z_m": 2.095,
"distance_sam_midhip_euclidean_m": 2.159
```

Cả hai đều đo từ camera đến điểm **giữa 2 hip** (midhip):

| Field | Cách tính | Ý nghĩa |
|---|---|---|
| `_z_m` | Chỉ lấy thành phần Z của midhip | Độ sâu thẳng (depth) |
| `_euclidean_m` | `sqrt(X² + Y² + Z²)` của midhip | Khoảng cách thực tế trong không gian 3D |

Hai giá trị gần nhau khi người đứng gần trục quang học của camera (X, Y nhỏ).
Khác nhau nhiều khi người đứng lệch nhiều so với tâm ảnh.

---

### `bbox_xyxy_normalized` — Bounding box người trong ảnh

```json
"bbox_xyxy_normalized": [0.181, 0.304, 0.771, 0.999]
```

Format: `[x1, y1, x2, y2]` — tọa độ **đã normalize** về khoảng [0, 1] theo chiều rộng/cao ảnh:

```
(x1, y1) ─────────┐
    │              │
    │   PERSON     │
    │              │
    └─────────── (x2, y2)
```

**Đọc ví dụ:**
- x1=0.181 → cạnh trái box cách lề trái ảnh 18.1%
- y1=0.304 → cạnh trên box ở 30.4% từ đỉnh ảnh
- x2=0.771 → cạnh phải ở 77.1%
- y2=0.999 → cạnh dưới ở gần đáy ảnh (99.9%)

---

### `pred_keypoints_2d_px` — Keypoint 2D trong ảnh (pixel)

```json
"pred_keypoints_2d_px": [
  [680.85, 800.60],   // nose: cột 681, dòng 801
  ...
]
```

- Shape: **(70, 2)** — 70 keypoints, mỗi cái là `[pixel_x, pixel_y]`
- Đây là **2D projection** của keypoint 3D lên ảnh
- Dùng để verify: nếu 3D đúng, chiếu xuống phải ra pixel này

**Công thức chiếu:**
```
pixel_x = focal_length * X / Z + cx
pixel_y = focal_length * Y / Z + cy

(cx = width/2, cy = height/2)
```

---

### `focal_length_px` — Tiêu cự camera

```json
"focal_length_px": 2305.12
```

- Đơn vị: **pixel**
- Dùng trong công thức chiếu 3D → 2D ở trên
- Giá trị này lớn → camera zoom in / góc nhìn hẹp
- ⚠️ **Không phải EXIF/thông số camera vật lý đo được** — là giá trị do
  SAM-3D-Body **ước lượng** lúc tạo pseudo-GT (`pose3d_source: "sam3d"`).
  RTMPose3D không tự suy luận ra `f`, chỉ tiêu thụ giá trị này như hằng số đã
  biết khi back-project 2D→3D. Phân phối ước lượng này rất rộng giữa các
  sample (~775–6900px, std~764) — xem `ARCHITECTURE.md` §10.9.

---

### `keypoints_3d_sam_raw` và `pred_cam_t_m` — Dữ liệu gốc trước khi transform

```json
"keypoints_3d_sam_raw": [[-0.015, -1.517, -0.181], ...],
"pred_cam_t_m": [-0.018, 1.434, 2.103]
```

- `sam_raw`: Output thô từ model SAM-3D, chưa có translation
- `pred_cam_t_m`: Vector dịch chuyển từ gốc SAM về camera frame

**Quan hệ:**
```
keypoints_3d_cam_m = keypoints_3d_sam_raw + pred_cam_t_m
```

Đây là nguồn gốc của bug cũ: formula cũ dùng thêm axis remap sai → Y bị invert.

---

### Các field phụ

| Field | Giải thích |
|---|---|
| `id` | Tên file (không có extension) |
| `status` | `"ok"` = file hợp lệ |
| `pose3d_source` | `"sam3d"` = nguồn data từ SAM-3D-Body |
| `keypoint_schema` | `"MHR70"` = dùng 70 keypoint theo chuẩn MHR |
| `nose_depth_m` | Z của mũi (tiện dùng, bằng `keypoints_3d_cam_m[0][2]`) |
| `torso_center_cam_m` | Trung bình 4 điểm: 2 vai + 2 hip |
| `axis_mapping` | Ghi lại công thức đã dùng để tính `keypoints_3d_cam_m` |
| `elapsed_s` | Thời gian xử lý file này (giây) |

---

## Tóm tắt các field theo mục đích sử dụng

| Dùng để làm gì | Field |
|---|---|
| Train pose structure | `keypoints_3d_cam_m`, `keypoints_3d_root_midhip_m` |
| Train bbox | `bbox_xyxy_normalized` |
| Train distance | `distance_sam_midhip_z_m` |
| Verify tính đúng đắn | `pred_keypoints_2d_px` + `focal_length_px` |
| Không dùng để train | `keypoints_3d_sam_raw`, `pred_cam_t_m`, `elapsed_s`, `image_path` |

---

## Cách nhanh để kiểm tra 1 file có đúng không

```python
import json, math, numpy as np

with open("labels/sample.json") as f:
    d = json.load(f)

kpts = np.array(d["keypoints_3d_cam_m"])
px2d = np.array(d["pred_keypoints_2d_px"])
focal = d["focal_length_px"]
# lấy image size từ prediction hoặc mở ảnh
cx, cy = 720, 900  # ví dụ ảnh 1440x1800

errors = [
    math.hypot(focal * x / z + cx - p[0], focal * y / z + cy - p[1])
    for (x, y, z), p in zip(kpts, px2d) if z > 0
]
print(f"Median reprojection error: {np.median(errors):.4f} px")
# Nếu < 1px → đúng. Nếu > 100px → sai.
```
