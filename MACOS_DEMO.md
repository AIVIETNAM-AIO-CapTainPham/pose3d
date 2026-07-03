# Cài đặt & chạy demo từ đầu trên macos (`.venv`)

Hướng dẫn chạy `[demo/app.py](demo/app.py)` — so sánh GT vs model finetune POSE-24
(24 keypoint), overlay 2D + skeleton 3D.

Mọi lệnh `uv pip` / `uv run` bên dưới **phải chạy từ thư mục gốc repo** để cài vào
`.venv` (không dùng `pip` hệ thống).

## Yêu cầu

- Python **3.11+**
- `[uv](https://docs.astral.sh/uv/)`
- Ít nhất một checkpoint finetune trong `work_dirs/pose24/` (ví dụ
`best_MPJPE_epoch_30.pth`)

## 1. Tạo virtualenv + UI dependencies

```bash
cd /path/to/pose3d
uv sync
```

`uv sync` tạo `.venv` và cài streamlit, plotly, opencv, numpy, … (xem
`pyproject.toml`). **Stack ML (torch, mmcv, mmpose) không nằm trong** `uv sync` — cài
thủ công ở bước 2.

Kiểm tra đang dùng đúng venv:

```bash
uv run which python
# → .../pose3d/.venv/bin/python
```



## 2. ML stack — chọn theo máy

**Phiên bản khớp nhau (quan trọng):**


| Package    | Version                                           |
| ---------- | ------------------------------------------------- |
| `mmpose`   | 1.3.2                                             |
| `mmcv`     | **2.1.0** (`mmpose 1.3.x` yêu cầu `mmcv < 2.2.0`) |
| `mmengine` | 0.10.7                                            |
| `mmdet`    | 3.3.0                                             |




### CPU inference

Không có wheel `mmcv` cho macOS → build từ source (~15–30 phút). Inference chạy CPU
(chậm hơn GPU nhưng đủ demo).

```bash
# PyTorch CPU (pin 2.4.x cho tương thích mmcv)
uv pip install "torch==2.4.1" "torchvision==0.19.1"

# Công cụ build
uv pip install cython numpy "setuptools<70" wheel ninja psutil

# xtcocotools — build khớp numpy hiện tại (tránh lỗi dtype)
uv pip install xtcocotools --no-build-isolation --no-binary xtcocotools

# mmcv 2.1.0 — build CPU ops trên macOS
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 MAX_JOBS=4 \
  uv pip install "mmcv==2.1.0" --no-binary mmcv --no-build-isolation

# OpenMMLab
uv pip install "mmengine>=0.7.0" "mmdet>=3.0.0" "mmpose>=1.0.0" tqdm --no-build-isolation
```



## 3. Kiểm tra cài đặt

```bash
uv run python -c "
import sys, torch, cv2, mmcv, mmengine, mmpose
from xtcocotools.coco import COCO
print('python:', sys.executable)
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('mmcv:', mmcv.__version__, '| mmpose:', mmpose.__version__)
print('ok')
"

PYTHONPATH=src uv run python -c "import pose24; print('pose24 ok')"
```



## 4. Checkpoint

App đọc checkpoint từ `work_dirs/pose24/`:

```bash
ls work_dirs/pose24/*.pth
```

Copy file `.pth` finetune vào đó nếu train trên máy khác. Tùy chọn so sánh với
RTMPose3D gốc: đặt `work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth`.

## 5. Chạy demo

```bash
make demo
```

Hoặc:

```bash
PYTHONPATH=src uv run streamlit run demo/app.py --server.port 8252
```

Mở URL Streamlit in ra (thường `http://localhost:8252`).

### Sidebar (`demo/app.py`)

- **Input source**: dataset sample (val/test/train) hoặc upload ảnh.
- **Finetuned checkpoint**: chọn file `.pth` trong `work_dirs/pose24/`.
- **Compare with original RTMPose3D**: bật nếu có
`work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth`.

Kết quả: overlay 2D + skeleton 3D tương tác (plotly), GT (xanh) vs Pred (đỏ).

App tham chiếu RTMPose3D gốc (tải checkpoint trong UI): `[demo/rtm/](demo/rtm/)`.

## Xử lý lỗi thường gặp


| Lỗi                            | Cách xử lý                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| `No module named 'mmpose'`     | Cài bằng `uv pip` từ repo root, không dùng `pip` hệ thống                                   |
| `numpy.dtype size changed`     | `uv pip install xtcocotools --no-build-isolation --no-binary xtcocotools --force-reinstall` |
| `MMCV==2.2.0 is incompatible`  | Dùng `mmcv==2.1.0`, không cài 2.2.0 với `mmpose 1.3.x`                                      |
| `mmcv._ext` / symbol not found | mmcv build lệch torch — gỡ và build lại mmcv sau khi pin torch                              |
| `No finetuned checkpoints`     | Thêm file `.pth` vào `work_dirs/pose24/`                                                    |


