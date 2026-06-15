# POSE3D — Streamlit UI cho RTMPose3D

UI bằng **Streamlit** để chạy 3D pose estimation với [b-arac/rtmpose3d](https://github.com/b-arac/rtmpose3d).
Cho phép **truyền model (checkpoint đã fine-tune) vào** rồi inference trên ảnh, hiển thị 2D keypoints overlay + 3D skeleton.

App: demo/rtm/app.py

## Môi trường

Máy đích: **NVIDIA GB10 (Grace Blackwell, aarch64)**, CUDA driver 580, toolkit hệ thống CUDA 13.0.
Điểm mấu chốt: kiến trúc **aarch64** + GPU Blackwell (sm_120) nên phải dùng PyTorch `cu128`, và `mmcv`
không có wheel sẵn -> **build từ source** với một CUDA toolkit **12.8** (khớp major với torch cu128).

Quản lý môi trường bằng `uv` (`.venv` trong repo).

### 1. UI + PyTorch (cu128)

    uv sync
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Kiểm tra GPU:

    uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

### 2. CUDA toolkit 12.8 (để build mmcv)

Toolkit hệ thống là 13.0 (major mismatch với torch cu128 -> PyTorch chặn build).
Lấy nvcc 12.8 độc lập qua micromamba (không cần sudo):

    curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba

    MAMBA_ROOT_PREFIX=$HOME/micromamba ./bin/micromamba create -y -p $HOME/cuda128 \
      -c nvidia -c conda-forge \
      "cuda-nvcc=12.8" "cuda-cudart-dev=12.8" "cuda-cccl=12.8" "cuda-nvtx=12.8" \
      "libcusparse-dev=12" "libcublas-dev=12" "libcusolver-dev=11"

Trên aarch64 (sbsa) header lib nằm ở `$HOME/cuda128/targets/sbsa-linux/include`,
nên build mmcv phải thêm `CPATH` trỏ vào đó (xem bước 3). Build mmcv mất ~1 giờ.

### 3. Build mmcv từ source

    export CUDA_HOME=$HOME/cuda128
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
    export CPATH=$CUDA_HOME/targets/sbsa-linux/include:$CPATH
    export LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LIBRARY_PATH
    export MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"

    uv pip install "setuptools<70" wheel       # cung cấp pkg_resources cho build
    uv pip install "mmcv==2.2.0" --no-binary mmcv --no-build-isolation

### 4. mmpose / mmdet / rtmpose3d

xtcocotools (dep của mmpose) cần build, nên cài Cython trước rồi dùng `--no-build-isolation`:

    uv pip install cython numpy
    uv pip install xtcocotools --no-build-isolation
    uv pip install "mmengine>=0.7.0" "mmdet>=3.0.0" "mmpose>=1.0.0" tqdm --no-build-isolation
    uv pip install -e demo/rtm --no-deps --no-build-isolation

## Chạy app

Dùng script `run.sh` (đã set sẵn `LD_LIBRARY_PATH` tới CUDA 12.8 libs cho runtime):

    ./run.sh

Hoặc thủ công:

    export LD_LIBRARY_PATH=$HOME/cuda128/lib64:$HOME/cuda128/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
    uv run streamlit run demo/rtm/app.py

Trong sidebar:
- **📥 Tải model về**: clone checkpoint ngay trong app — từ *URL trực tiếp*,
  *HuggingFace Hub* (repo id + tên file), hoặc *checkpoint mặc định RTMPose3D*.
  File tải về cache `~/.cache/rtmpose3d/checkpoints/`, dùng được luôn qua mục
  *Đã tải về* bên dưới.
- **Model size**: `l` (large) hoặc `x` (extra large).
- **Pose checkpoint**: chọn *Mặc định* (auto-download), *Đã tải về* (từ mục tải
  ở trên), *Đường dẫn/URL* (trỏ checkpoint **fine-tune** của bạn), hoặc *Upload .pth*.
- **Pose config**: để trống dùng mặc định theo model size, hoặc trỏ config riêng.
- **Detector**: tùy chọn, mặc định auto-download RTMDet-M.
- **Device**: `cuda:0` hoặc `cpu`.

Bấm **Load model** -> upload ảnh -> **Chạy inference**. Kết quả: ảnh 2D overlay keypoints +
skeleton 3D tương tác (plotly).

## Dữ liệu

`data/GT/{images,labels}` — ảnh + nhãn ground-truth cho fine-tune.

## Cấu trúc thư mục

    POSE3D/
    ├── src/pose24/              # Package chính — bộ pose 24 keypoint
    │   ├── keypoints.py         # Định nghĩa 24 keypoint, tên, nối xương, cặp trái-phải
    │   ├── datasets/            # Đọc & tiền xử lý dữ liệu
    │   │   ├── gt_json_dataset.py  # Nạp nhãn GT (JSON) → 24 keypoint
    │   │   └── transforms.py       # Biến đổi/augment ảnh + keypoint khi train
    │   ├── codecs/              # Mã hoá/giải mã nhãn 3D (toạ độ ↔ dạng model học)
    │   ├── models/              # Định nghĩa mô hình
    │   │   ├── rtmw3d_head.py      # Đầu ra dự đoán keypoint 3D
    │   │   ├── pose_estimator.py   # Ghép thành mô hình ước lượng pose hoàn chỉnh
    │   │   ├── loss.py             # Hàm mất mát cơ bản
    │   │   └── structure_loss.py   # Mất mát giữ đúng cấu trúc/tỉ lệ khung xương
    │   ├── engine/hooks.py      # Tự xuất ảnh so sánh GT vs Pred trong lúc train
    │   ├── visualization/draw.py # Vẽ overlay 2D + skeleton 3D (GT vs Pred)
    │   └── configs/             # File cấu hình train/finetune
    │
    ├── tools/                   # Script chạy tay
    │   ├── make_splits.py       # Chia dữ liệu train/val/test
    │   ├── train.py             # Huấn luyện / finetune
    │   ├── eval.py              # Đánh giá checkpoint
    │   └── visualize.py         # Xuất ảnh kiểm tra keypoint
    │
    ├── demo/                    # App Streamlit demo inference
    │   ├── app.py               # Giao diện chính (đã finetune)
    │   └── rtm/                 # App tham chiếu RTMPose3D gốc
    │
    ├── tests/                   # Bộ test tự động (dataset, loss, model, pipeline, viz...)
    ├── data/GT/                 # Ảnh + nhãn ground-truth để finetune
    ├── work_dirs/               # Nơi lưu checkpoint & log khi train
    ├── vis/                     # Ảnh trực quan hoá xuất ra
    ├── clone_code/              # Repo tham chiếu (chỉ đọc) — pose3d & theia gốc
    │
    ├── Makefile                 # Các lệnh tắt: test / lint / pre-commit / train...
    ├── run.sh                   # Chạy app (đã set sẵn biến môi trường CUDA)
    ├── ARCHITECTURE.md          # Mô tả kiến trúc hệ thống
    └── GT_JSON_GUIDE.md         # Hướng dẫn định dạng nhãn GT JSON

## Phát triển

Các tác vụ dev gói trong `Makefile` (chạy `make help` để xem đầy đủ):

    make test         # chạy toàn bộ test suite (unit + loss + viz + pipeline + model + flip)
    make lint         # ruff check src/ tests/ tools/
    make pre-commit   # chạy tất cả pre-commit hooks trên mọi file

Cài hook tự chạy mỗi lần commit (tùy chọn):

    uv run pre-commit install

## Ghi chú

- 133 keypoints COCO-WholeBody: body(17) + feet(6) + face(68) + hands(42).
- 3D theo quy ước Z-up, đơn vị mét (camera-relative).
- Lần đầu với checkpoint mặc định tải ~330MB về `~/.cache/rtmpose3d/checkpoints/`.
- PyTorch >= 2.6 mặc định `weights_only=True` làm hỏng load checkpoint cũ;
  `rtmpose3d/inference.py` đã vá `torch.load` về `weights_only=False`.
- Nếu gặp `CUDA error: out of memory` lúc load model: GB10 dùng unified memory,
  kiểm tra `nvidia-smi` xem process khác có đang chiếm VRAM không. Có thể chạy
  `Device = cpu` để test pipeline.
- Pipeline đã verify chạy đúng (1 người, 133 keypoints 2D+3D) trên ảnh mẫu.
