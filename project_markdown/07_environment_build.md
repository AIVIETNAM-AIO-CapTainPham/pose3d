# Environment Build Notes

## Stack da verify

Ngay verify gan nhat trong repo nay:

```text
torch 2.12.1+cu130
torch CUDA 13.0
GPU NVIDIA GB10
cv2 4.13.0
mmcv 2.2.0
mmpose 1.3.2
mmdet 3.3.0
mmengine 0.10.7
```

Lenh kiem tra:

```bash
uv run python -c "import torch, cv2, mmcv, mmpose, mmengine; print('ok')"
```

## Build mmcv voi torch cu130

Vi torch hien tai la `cu130`, build `mmcv` voi CUDA toolkit 13.0:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/sbsa-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LIBRARY_PATH
export MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"

uv pip install --force-reinstall opencv-python-headless
uv pip install "setuptools<70" wheel
uv pip install "mmcv==2.2.0" --no-binary mmcv --no-build-isolation
```

Lan build da verify:

```text
Built mmcv==2.2.0
Prepared 1 package in 31m 54s
Installed 1 package in 4ms
```

## Loi da gap

### `ModuleNotFoundError: No module named 'mmpose'`

Chua cai OpenMMLab stack:

```bash
uv pip install cython numpy
uv pip install xtcocotools --no-build-isolation
uv pip install "mmengine>=0.7.0" "mmdet>=3.0.0" "mmpose>=1.0.0" tqdm --no-build-isolation
```

### `ModuleNotFoundError: No module named 'mmcv'`

Chua build/cai `mmcv`. Chay block build CUDA 13.0 o tren.

### `AttributeError: module 'cv2' has no attribute '__version__'`

`opencv-python` trong `.venv` bi loi/thieu binary `cv2`. Sua bang:

```bash
uv pip install --force-reinstall opencv-python-headless
```

Kiem tra:

```bash
uv run python -c "import cv2; print(cv2.__version__, cv2.__file__)"
```

## Nguyen tac CUDA

CUDA toolkit dung build `mmcv` phai khop major voi CUDA cua torch:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

- Neu torch la `cu130` / `torch.version.cuda == 13.0`: dung `/usr/local/cuda-13.0`.
- Neu torch la `cu128` / `torch.version.cuda == 12.8`: moi dung `$HOME/cuda128`.

Khong tron `torch cu130` voi `$HOME/cuda128`.
