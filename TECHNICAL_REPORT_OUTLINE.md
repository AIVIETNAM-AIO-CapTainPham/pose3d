# Cấu trúc Technical Report — RTMPose3D Finetune cho POSE-24

> File này là **khung sườn** (outline) để viết technical report hoàn chỉnh.
> Mỗi mục có: mục tiêu của phần đó, nội dung cần điền, và **nguồn lấy dữ liệu**
> (đã có sẵn trong repo, chỉ cần tổng hợp lại — không cần đo lại từ đầu).
>
> **Nguyên tắc nội dung**: report chỉ trình bày *vấn đề kỹ thuật của thiết kế
> gốc* và *giải pháp/thay đổi kiến trúc tương ứng*. Không kể lại quá trình vận
> hành/debug cụ thể (vd lỗi dữ liệu cá nhân gặp khi train, log thử-sai) — những
> nội dung đó đã được dọn vào `project_markdown/06_management_notes.md` để
> tham khảo nội bộ, không đưa vào report.

---

## 0. Title page / Abstract

- Tên project, tác giả, ngày.
- Abstract (150–250 từ): bài toán (3D body pose estimation cho 24 keypoint,
  gồm 17 COCO body + 7 điểm giải phẫu), nền tảng (finetune RTMPose3D-L), thay
  đổi kiến trúc chính, kết quả MPJPE/P-MPJPE đạt được.

**Nguồn**: `README.md`, `project_markdown/01_overview.md`.

---

## 1. Giới thiệu (Introduction)

1.1. Bối cảnh & động lực
   - Vì sao cần 3D pose riêng cho 24 keypoint (body + landmark giải phẫu) thay
     vì dùng model wholebody 133-keypoint có sẵn.
   - Ứng dụng dự kiến (nếu có): phân tích vận động, tư thế lâm sàng, v.v.

1.2. Bài toán đặt ra
   - Input: ảnh person-crop. Output: 24 keypoint 3D (x, y pixel + z depth
     root-relative), sau đó back-project ra camera space (mét).
   - Đặc điểm dữ liệu: nhãn 3D là pseudo-GT từ SAM-3D-Body (không phải mocap
     thật) — nêu như đặc điểm bài toán, không phải lỗi.

1.3. Đóng góp chính (Contributions) — liệt kê ngắn, triển khai chi tiết ở §4–§5:
   - Thu hẹp + mở rộng schema keypoint (133→24, thêm 7 điểm giải phẫu).
   - Thay `BoneLoss` (non-differentiable) bằng `Pose3DStructureLoss`
     (soft-argmax, differentiable tới nhánh depth).
   - Bổ sung transform đồng bộ nhãn 3D khi augment (flip ảnh).
   - Calibrate lại `z_range` (codec) và focal length fallback (decode) theo
     thống kê thật của dataset, thay cho giá trị default generic.
   - Điều chỉnh lịch trình training (epoch/LR) phù hợp quy mô dataset.

**Nguồn**: `project_markdown/01_overview.md`.

---

## 2. Related Work (tuỳ chọn — bỏ qua nếu report nội bộ không cần)

- RTMPose / RTMPose3D (paper gốc, repo `b-arac/rtmpose3d`).
- SimCC (Simple Coordinate Classification) — vì sao chọn classification thay
  vì regression cho keypoint localization.
- Kỹ thuật structure-loss (root + relative + bone term) áp dụng trong project
  — trình bày như **kỹ thuật của mình**, không cần trích dẫn nguồn ngoài.

**Nguồn**: `ARCHITECTURE.md` §6 (Loss functions).

---

## 3. Dataset

3.1. Nguồn dữ liệu
   - SAM-3D-Body pseudo-3D labels, nguồn ảnh.
   - Số lượng: tổng số ảnh/label, train/val/test split.

3.2. Format nhãn gốc (MHR70) → POSE-24
   - Bảng mapping 24 keypoint ← MHR70 index.
   - Lý do chọn 7 điểm giải phẫu bổ sung (olecranon, cubital fossa, acromion,
     neck) — nếu có lý do ứng dụng cụ thể, ghi vào đây.

3.3. Hệ toạ độ & đơn vị
   - Camera frame `x_right_y_down_z_forward`, đơn vị mét.
   - Quy ước: Y dương = xuống (khác convention hiển thị thông thường) — nêu
     như đặc tả hệ toạ độ, phần ảnh hưởng tới hiển thị nói ở §4 (visualization).

**Nguồn**: `ARCHITECTURE.md` §2–§3; `project_markdown/03_data_inventory.md`;
`GT_JSON_GUIDE.md`.

---

## 4. Phương pháp (Methodology)

> Đây là phần dài nhất và quan trọng nhất — trình bày **tuần tự theo pipeline
> forward**, mỗi khối nêu rõ: thiết kế gốc (RTMPose3D-L cocktail14) → vấn đề
> kỹ thuật (nếu có) → thay đổi áp dụng cho bài toán POSE-24.

4.1. Kiến trúc tổng thể (Overview diagram)
   - Sơ đồ: Image → Backbone → Neck → Head → SimCC logits → Decode.

4.2. Backbone — CSPNeXt-L
   - Giữ nguyên kiến trúc gốc (không thay đổi).
   - Bảng channel/stride theo stage.

4.3. Neck — CSPNeXtPAFPN
   - Giữ nguyên, mô tả vai trò (multi-scale feature fusion cho 2 nhánh
     fine/coarse).

4.4. Head — RTMW3DHead
   - Cơ chế multi-scale fusion (PixelShuffle + concat 2 nhánh).
   - GAU (Gated Attention Unit) — self-attention giữa các keypoint.
   - 3 đầu phân loại `cls_x/y/z` — **thay đổi**: `out_channels` 133→24 (theo
     schema mới ở §3.2).

4.5. Codec — SimCC3DLabel
   - Nguyên lý SimCC: định vị = phân loại (classification thay vì regression).
   - Công thức encode/decode (x, y theo pixel; z root-relative → bin).
   - **Vấn đề thiết kế gốc**: `z_range` default (2.1745 m) được hiệu chỉnh cho
     phân phối depth của dataset cocktail14 (đa nguồn, sải rộng); với phạm vi
     depth thực tế của POSE-24 (femur/torso ~0.4–0.5 m) thì default này khiến
     phần lớn bin SimCC trên trục z không được sử dụng — độ phân giải lượng tử
     hoá hiệu dụng cho depth bị giảm.
   - **Thay đổi**: `z_range` → 1.5 m, calibrate theo thống kê depth thật của
     POSE-24, tăng độ phân giải hiệu dụng trên trục z.
   - Ví dụ số minh hoạ encode/decode 1 keypoint (copy từ ARCHITECTURE.md §4.6).

4.6. Loss functions
   - `KLDiscretLossWithWeight` — giữ nguyên, mô tả công thức KL trên 3 trục.
   - **Vấn đề thiết kế gốc**: `BoneLoss` lấy toạ độ bằng `argmax` trên phân
     phối SimCC — hàm này không có đạo hàm, nên gradient của loss không lan
     truyền ngược được tới các đầu phân loại (`cls_x/y/z`), đặc biệt là nhánh
     depth (`cls_z`); thành phần ràng buộc hình học (độ dài xương) thực chất
     không huấn luyện được phần dự đoán depth.
   - **Thay đổi**: thay `BoneLoss` bằng `Pose3DStructureLoss`, dùng kỹ thuật
     **soft-argmax** (toạ độ = kỳ vọng có trọng số của softmax(logits × β))
     để toàn bộ phép lấy toạ độ khả vi (differentiable). Loss gồm 3 hạng tử
     Smooth-L1 trên không gian normalised [0,1]: root (định vị pelvis),
     relative (vị trí khớp so với root), bone-length (tỉ lệ đoạn xương).
   - Hạn chế còn lại (nêu ngắn ở §7): các trục x/y/z có đơn vị vật lý khác
     nhau trong không gian [0,1] normalised, nên thành phần z có thể lấn át
     x/y trong bone-length loss.

4.7. Decode / Post-processing
   - `TopdownPoseEstimator3D`: back-project (u, v, Z_abs) → camera (X, Y, Z)
     bằng intrinsics: `X = (u − cx)/f × Z`.
   - **Vấn đề thiết kế gốc (vòng 1)**: focal length fallback (1145 px) là giá
     trị mặc định của RTMPose3D cho dataset cocktail14; áp dụng cho nguồn ảnh
     khác gây sai lệch hệ thống về tỉ lệ (scale) khi quy đổi toạ độ pixel
     sang camera space.
   - **Vấn đề thiết kế gốc (vòng 2)**: kể cả sau khi đổi fallback sang giá
     trị trung vị của dataset (2074 px), đây vẫn là 1 giá trị chung cho mọi
     sample — trong khi mỗi ảnh có `focal_length_px` riêng (ước lượng bởi
     SAM-3D-Body lúc tạo pseudo-GT, phân phối rất rộng ~775–6900px) đã có sẵn
     trong GT JSON nhưng bị bỏ qua. Sample có `f` ước lượng lệch xa trung vị
     vẫn cho MPJPE outlier do sai scale, không phản ánh chất lượng dự đoán
     keypoint thật.
   - **Thay đổi**: đọc `focal_length_px` per-sample từ GT, dùng đúng giá trị
     riêng của từng ảnh khi back-project — chỉ còn dùng giá trị trung vị làm
     fallback cho ảnh không có GT (vd ảnh ngoài qua demo).
   - **Lưu ý quan trọng cho phần Thảo luận/Hạn chế**: `focal_length_px` không
     phải thông số camera vật lý đo được — là ước lượng của SAM-3D-Body, nên
     độ chính xác phụ thuộc chất lượng ước lượng đó, không phải con số tuyệt
     đối đáng tin 100%.

**Nguồn**: `ARCHITECTURE.md` §1, §4, §5, §6, §7, §10 (đã viết chi tiết, có
công thức + ví dụ số — diễn giải lại theo văn phong report).

---

## 5. Data Pipeline & Training Setup

5.1. Augmentation pipeline
   - Bảng liệt kê từng transform + vai trò (xem ARCHITECTURE.md §8.1).
   - **Vấn đề thiết kế gốc**: `RandomFlip` (mmpose) chỉ áp dụng phép lật cho
     toạ độ 2D (mirror + đổi chỉ số trái/phải); trường `keypoints_3d` không
     được transform đồng bộ, nên sau khi lật ảnh, nhãn depth gắn với chỉ số
     khớp không còn tương ứng đúng bên trái/phải.
   - **Thay đổi**: bổ sung transform `Flip3DKeypoints` ngay sau `RandomFlip`,
     thực hiện lại đúng 2 phép biến đổi cho `keypoints_3d`: hoán đổi chỉ số
     theo `flip_indices` và lấy đối xứng trục camera X.

5.2. Optimizer & LR schedule
   - AdamW, warmup (LinearLR) + cosine decay — công thức (xem ARCHITECTURE.md
     §8.2).
   - **Thay đổi so với cấu hình gốc**: thời điểm bắt đầu cosine decay được
     đưa sớm hơn (epoch 20 so với mốc `max_epochs/2` mặc định), kết hợp giảm
     `base_lr` — điều chỉnh phù hợp với quy mô dataset nhỏ hơn cocktail14
     (chi tiết động lực ở §7.3/§8).

5.3. Hyperparameter cuối cùng (bảng)
   - max_epochs, base_lr, batch_size, val_interval, AMP, EMA — bảng so sánh
     cấu hình gốc (cocktail14) vs cấu hình POSE-24.

**Nguồn**: `ARCHITECTURE.md` §8; `project_markdown/04_training_eval.md`.

---

## 6. Đánh giá (Evaluation Metrics)

- Định nghĩa MPJPE (Mean Per-Joint Position Error).
- Định nghĩa P-MPJPE (sau Procrustes alignment) — loại bỏ sai số global
  rotation/scale/translation, dùng để tách bạch lỗi hình học tương đối khỏi
  lỗi định vị/scale tuyệt đối.
- Vai trò bổ trợ của 2 metric: chênh lệch lớn giữa MPJPE và P-MPJPE là dấu
  hiệu cho thấy lỗi nằm ở scale/vị trí tuyệt đối hơn là ở hình dạng tương đối.

**Nguồn**: `ARCHITECTURE.md` §9.

---

## 7. Kết quả thực nghiệm (Results)

7.1. Thiết lập đánh giá
   - Tập val/test (số lượng sample), checkpoint được dùng để báo cáo.

7.2. Kết quả định lượng (bảng số)
   - Bảng MPJPE/P-MPJPE theo epoch — copy từ log thật, không tự đặt số.
   - Kết quả cuối cùng (best checkpoint).

7.3. Ảnh hưởng của từng thay đổi (ablation, nếu có số liệu)
   - So sánh MPJPE/P-MPJPE trước/sau khi calibrate `z_range` và focal length
     (nếu có log riêng từng giai đoạn).
   - Quan sát về độ phân giải depth (mức độ tách biệt giữa các khớp theo
     trục z) trước/sau calibrate `z_range`.
   - Đường cong train-loss vs val-MPJPE theo epoch cho 2 cấu hình lịch trình
     LR (decay muộn vs decay sớm) — minh hoạ lý do điều chỉnh ở §5.2.

7.4. Kết quả định tính (qualitative)
   - Ảnh từ `Pose3DVisualizationHook` (GT vs Pred, 2D overlay + 3D skeleton).
   - Ảnh so sánh model gốc (133-kpt, body-17 subset) vs model POSE-24 (qua
     demo Streamlit), nếu dùng để minh hoạ.

7.5. So sánh với baseline (model gốc RTMPose3D-L cocktail14)
   - Bảng so sánh trên body-17 subset: model gốc vs model POSE-24, cùng metric.

**Nguồn**: log thật trong `work_dirs/*/`; ảnh trong `work_dirs/*/vis/`;
`project_markdown/04_training_eval.md` (đã có bảng MPJPE theo epoch).

---

## 8. Thảo luận (Discussion)

8.1. Vì sao depth cải thiện so với baseline gốc
   - Hai yếu tố thiết kế đóng góp: (a) `z_range` calibrate khớp phạm vi giá
     trị thật → tăng độ phân giải hiệu dụng của trục z trong codec; (b) nhãn
     depth nhất quán theo trái/phải trong toàn bộ augmentation (sau khi bổ
     sung `Flip3DKeypoints`) → tín hiệu huấn luyện cho trục z không bị triệt
     tiêu bởi nhãn mâu thuẫn.

8.2. Hạn chế còn tồn tại
   - Trộn đơn vị trong structure loss (không gian [0,1] khiến z lấn át x,y
     trong bone-length).
   - Dữ liệu là pseudo-GT (SAM-3D-Body), không phải mocap thật — kể cả sau
     khi dùng `focal_length_px` per-sample, đây vẫn là **ước lượng** của
     SAM-3D-Body, không phải thông số camera đo được; sai số gốc của ước
     lượng đó không loại bỏ được, chỉ tránh được lỗi cộng thêm do dùng 1 giá
     trị chung sai cho mọi sample.
   - Ảnh không có GT (vd ảnh ngoài qua demo) vẫn phải dùng 1 focal length
     fallback chung (giá trị trung vị dataset) — không có cách biết `f` thật
     của ảnh đó, nên toạ độ 3D tuyệt đối (mét) kém chính xác hơn so với ảnh
     trong tập train/val (có `focal_length_px` riêng).

8.3. Hướng cải thiện tiếp theo (Future work)
   - Tính structure loss trong không gian mét thật (back-project trước khi
     tính bone-length) thay vì [0,1] normalised.
   - Thêm cơ chế dừng sớm (early stopping) theo dõi MPJPE để tự động hoá việc
     chọn điểm dừng huấn luyện.

**Nguồn**: `ARCHITECTURE.md` mục "hạn chế đã biết" (§6.2, §8.5, §10.9, §10.10).

---

## 9. Kết luận (Conclusion)

- Tóm tắt 3–5 câu: đã finetune RTMPose3D-L cho 24-keypoint body + giải phẫu,
  với các thay đổi kiến trúc/thiết kế ở loss, codec, augmentation và decode;
  đạt MPJPE/P-MPJPE cụ thể (điền số liệu cuối cùng); nêu rõ giới hạn còn lại.

---

## 10. Phụ lục (Appendix) — tuỳ chọn

- A. Toàn bộ bảng tham số config cuối (copy nguyên từ file config).
- B. Danh sách test suite + mục đích từng test (từ
  `project_markdown/05_demo_tools_tests.md`).
- C. Hướng dẫn tái lập (reproduce): lệnh `make train`, `make eval`,
  `make demo` — từ Makefile.
- D. Kiến trúc thư mục project (từ `project_markdown/01_overview.md`).

---

## Bảng tổng hợp nhanh: phần nào lấy từ nguồn nào

| Mục report | Nguồn chính trong repo |
|---|---|
| §1 Giới thiệu | `project_markdown/01_overview.md` |
| §3 Dataset | `ARCHITECTURE.md` §2–§3, `GT_JSON_GUIDE.md`, `project_markdown/03_data_inventory.md` |
| §4 Phương pháp | `ARCHITECTURE.md` §1, §4–§7, §10 (công thức, ví dụ số, bảng shape) |
| §5 Pipeline & Training | `ARCHITECTURE.md` §8, `project_markdown/04_training_eval.md` |
| §6 Đánh giá | `ARCHITECTURE.md` §9 |
| §7 Kết quả | log thật `work_dirs/*/`, ảnh `work_dirs/*/vis/` |
| §8 Thảo luận | `ARCHITECTURE.md` §6.2, §8.5 (mục hạn chế) |
| §10 Phụ lục | `project_markdown/05_demo_tools_tests.md`, `01_overview.md`, `Makefile` |

> Gợi ý khi viết: với mỗi mục, mở file nguồn tương ứng, **diễn giải lại bằng
> văn phong report kỹ thuật** (vấn đề thiết kế → giải pháp → kết quả), không
> kể theo trình tự thời gian thao tác/debug. Chỉ giữ lại số liệu/công thức đã
> được xác nhận đúng trong repo — tránh suy diễn thêm số liệu chưa đo.

> **Cập nhật tài liệu liên quan**: các vấn đề kỹ thuật đã được mô tả đầy đủ ở
> `ARCHITECTURE.md` (§4.5, §6, §8.1, §10.8–§10.9) dưới dạng "vấn đề thiết kế →
> thay đổi", và `project_markdown/02_code_architecture.md` /
> `04_training_eval.md` đã đồng bộ theo đúng góc nhìn này. Phần "quá trình vận
> hành/sự cố cụ thể" (vd lỗi môi trường, OOM khi chạy đồng thời demo) chỉ nằm
> trong `project_markdown/06_management_notes.md` — không đưa vào report.
