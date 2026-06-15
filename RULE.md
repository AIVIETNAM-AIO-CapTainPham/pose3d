# 3D Pose Estimation

Dự án nghiên cứu và xây dựng hệ thống ước lượng tư thế con người trong không gian 3D từ ảnh 2D thời gian thực.

# RTMPose3D Web

Streamlit UI để upload ảnh và chạy inference bằng [`b-arac/rtmpose3d`](https://github.com/b-arac/rtmpose3d).

## Models

| Config | Pose model | Speed | Accuracy |
|---|---|---|---|
| `config/rtmpose3d_l.json` | RTMW3D-L | faster | good |
| `config/rtmpose3d_x.json` | RTMW3D-X | slower | better |

Detector dùng chung: **RTMDet-M** (~50 MB). Pose checkpoint: L ~170 MB, X ~280 MB.

## Requirements

- Python 3.10–3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- CPU đủ để chạy (không cần GPU)

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd rtmpose_3d
make install-dev        # cài deps + pre-commit hook
make download-weights   # tải tất cả checkpoints (~500 MB)
```

> Nếu đã clone rồi mà thư mục `rtmpose3d/` rỗng: `git submodule update --init`

Để chỉ tải model L (nhẹ hơn):

```bash
uv run python -c "
import os, importlib.util
from pathlib import Path
ROOT = Path('.')
os.environ['RTMPOSE3D_CACHE_DIR'] = str(ROOT / 'weights')
spec = importlib.util.spec_from_file_location('dl', ROOT / 'rtmpose3d/rtmpose3d/weights/downloader.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.get_checkpoint_path('https://huggingface.co/rbarac/rtmpose3d/resolve/main/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', ROOT / 'weights')
m.get_checkpoint_path('https://huggingface.co/rbarac/rtmpose3d/resolve/main/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth', ROOT / 'weights')
"
```

## Chạy

```bash
make run
# hoặc trực tiếp:
uv run streamlit run app/streamlit_app.py
```

Mở `http://localhost:8501`.

Để dùng model X thay vì L, sửa `CONFIG_PATH` trong [app/streamlit_app.py](app/streamlit_app.py):

```python
CONFIG_PATH = ROOT / "config" / "rtmpose3d_x.json"
```

### Chạy trên LAN

```bash
uv run streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Mở `http://<IP-máy>:8501` từ thiết bị khác trên cùng mạng.

## GPU (tuỳ chọn)

Nếu có GPU NVIDIA với CUDA, cài MMCV wheel tương ứng rồi chạy:

```bash
RTMPOSE3D_DEVICE=cuda uv run streamlit run app/streamlit_app.py
```

hoặc sửa `"device"` trong file config thành `"cuda:0"`.

## Layout UI

```
┌──────────────────────────────────────────────────────┐
│  Input image          │  2D Overlay (skeleton)        │
├──────────────────────────────────────────────────────┤
│  3D Skeleton (xoay)   │  Keypoints table (17 joints)  │
└──────────────────────────────────────────────────────┘
```

Sidebar: bbox threshold · single/multi person · góc nhìn 3D · toggle raw JSON.

## Dev

```bash
make lint     # ruff check
make format   # ruff format
make clean    # xóa __pycache__, .cache, .ruff_cache
```

## Git Workflow

### Cấu trúc branch

```
main                        # tài liệu, ổn định
└── dev                     # tích hợp các feature
    └── feature/<tên>       # thêm tính năng mới
    └── fix/<tên>           # sửa bug
    └── chore/<tên>         # cập nhật config, docs,...
```

Ví dụ: `feature/data-preprocessing`, `fix/model-output-error`

### Các bước làm việc

```bash
# 1. Luôn cập nhật branch dev trước
git checkout dev
git pull origin dev

# 2. Tạo branch mới từ dev
git checkout -b feature/<tên-tính-năng>

# 3. Làm việc, sau đó commit
git add .
git commit -m "feat: mô tả ngắn thay đổi"

# 4. Push branch lên remote
git push origin feature/<tên-tính-năng>

# 5. Tạo Pull Request trên GitHub để merge vào dev
```

### Quy tắc commit message

| Prefix | Dùng khi |
|---|---|
| `feat:` | Thêm tính năng mới |
| `fix:` | Sửa bug |
| `docs:` | Cập nhật tài liệu |
| `chore:` | Thay đổi config, dependencies |
| `refactor:` | Refactor code |
