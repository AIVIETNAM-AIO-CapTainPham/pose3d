# Tong Quan Du An

## Muc tieu

Du an POSE3D fine-tune RTMPose3D-L de du doan pose 3D cho bo 24 keypoint
body/anatomical landmark. Nhan duoc lay tu schema MHR70, sau do chon ra 24 diem
phu hop voi bai toan hien tai.

Output chinh:

- 2D keypoints tren anh.
- 3D keypoints trong camera frame, don vi met.
- Skeleton POSE-24 co the visualize GT vs prediction.

## Stack ky thuat

- Python >= 3.11.
- Quan ly moi truong bang `uv`.
- ML/OpenMMLab: `torch`, `mmcv`, `mmengine`, `mmdet`, `mmpose`.
- Demo: `streamlit`, `plotly`, `opencv-python`, `matplotlib`.
- Test/lint: `pytest`, `ruff`, `pre-commit`.

Moi truong da verify gan nhat tren NVIDIA GB10/aarch64:

- `torch==2.12.1+cu130`
- CUDA toolkit build `mmcv`: `/usr/local/cuda-13.0`
- `mmcv==2.2.0`
- `mmpose==1.3.2`, `mmdet==3.3.0`, `mmengine==0.10.7`
- lenh kiem tra: `uv run python -c "import torch, cv2, mmcv, mmpose, mmengine; print('ok')"`

Khong nen tron `torch cu130` voi `CUDA_HOME=$HOME/cuda128`. Neu dung torch cu128
thi moi build `mmcv` voi CUDA 12.8/cuda128.

## Cau truc repo cap cao

```text
POSE3D/
├── src/pose24/              # Package chinh cua du an
├── tools/                   # CLI train/eval/split/visualize
├── demo/app.py              # Streamlit app hien tai
├── tests/                   # Test suite
├── data/GT/                 # Anh + label GT
├── work_dirs/               # Checkpoint, log, visualization theo run
├── vis/                     # Anh visualization xuat tay
├── clone_code/              # Repo tham chieu/vendor read-only
├── README.md
├── ARCHITECTURE.md
├── GT_JSON_GUIDE.md
├── Makefile
└── pyproject.toml
```

## Code chinh vs tham chieu

Code chinh can quan ly va sua doi nam o:

- `src/pose24`
- `tools`
- `demo/app.py`
- `tests`
- config train tai `src/pose24/configs/rtmw3d_l_finetune_pose24.py`

Code tham chieu nam o:

- `clone_code/pose3d`: app/RTMPose3D goc, config goc, package `rtmpose3d`.
- `clone_code/theia`: tham chieu ve loss/pose-camera workflow.

Nen xem `clone_code` la nguon doi chieu, khong phai package runtime chinh.

## Tai lieu co san trong repo

- `README.md`: huong dan moi truong, chay app, make targets.
- `ARCHITECTURE.md`: giai thich sau ve SimCC3D, loss, pipeline train.
- `GT_JSON_GUIDE.md`: giai thich field trong label JSON ground truth.

Thu muc `project_markdown` nay la ban tong hop thuc te sau khi doc code/data,
khong xoa hay thay the cac file tren.

