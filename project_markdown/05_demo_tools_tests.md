# Demo, Tools va Tests

## Demo Streamlit

File hien tai: `demo/app.py`

README cu co nhac `demo/rtm/app.py`, nhung trong repo hien tai app chinh la
`demo/app.py`, va Makefile target `demo` cung tro den file nay.

Chay:

```bash
make demo
```

hoac:

```bash
PYTHONPATH=src uv run streamlit run demo/app.py --server.port 8252 --server.address 0.0.0.0
```

Tinh nang trong app:

- browser sample theo split train/val/test (luon dung `val_pipeline` -- resize
  co dinh, khong random augment -- du browse split nao, de GT va anh hien thi
  luon khop voi input thuc te dua vao model; xem ARCHITECTURE.md SS10.11);
- upload anh ngoai;
- load finetuned checkpoint;
- load original RTMPose3D-L cocktail14 neu co checkpoint;
- ve 2D overlay;
- ve 3D skeleton bang Plotly;
- so sanh GT vs finetuned pred va original RTMPose3D subset body-17.

Hang so quan trong:

- config: `src/pose24/configs/rtmw3d_l_finetune_pose24.py`
- work dir app mac dinh: `work_dirs/pose24`
- original checkpoint: `work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth`
- root index: `[11, 12]`

Khi upload anh ngoai, app tao sample bang bbox user cung cap va fallback
`root_z=2.5`, vi khong co GT depth.

## Tools CLI

### `tools/make_splits.py`

Tao split txt tu `labels/*.json`.

```bash
PYTHONPATH=src uv run python tools/make_splits.py --data-root data/GT
```

Mac dinh train/val/test = 0.85/0.10/0.05.

### `tools/train.py`

Entry train khong can `mim`, import `pose24` de register custom modules, build
`Runner.from_cfg(cfg)` va goi `runner.train()`.

Ho tro:

- `--work-dir`
- `--resume`
- `--amp`
- `--cfg-options`
- `--launcher`

### `tools/eval.py`

Load checkpoint, build model, chay val/test split va in:

- MPJPE
- P-MPJPE

Backbone init pretrained bi tat khi eval vi weights lay tu checkpoint.

### `tools/visualize.py`

Render GT-only hoac GT-vs-Pred:

```bash
PYTHONPATH=src uv run python tools/visualize.py --num 4 --out vis
PYTHONPATH=src uv run python tools/visualize.py --checkpoint work_dirs/pose24_v2/best_MPJPE_epoch_15.pth --num 4 --out vis
```

Output mau hien co:

- `vis/sample_0.png`
- `vis/sample_1.png`
- `vis/sample_2.png`

## Makefile targets

| Target | Y nghia |
|---|---|
| `make splits` | tao split train/val/test |
| `make test` | chay all test suite |
| `make test-unit` | dataset/keypoint tests |
| `make test-loss` | structure loss tests |
| `make test-viz` | visualization tests |
| `make test-pipeline` | full mmpose pipeline tests |
| `make test-model` | build model + loss + predict |
| `make test-flip` | regression cho Flip3DKeypoints |
| `make train` | finetune single GPU AMP |
| `make train-resume` | resume train |
| `make train-multi-gpu` | launcher pytorch |
| `make eval` | eval checkpoint mac dinh |
| `make visualize` | render comparison images |
| `make demo` | chay Streamlit app |
| `make lint` | ruff check |
| `make pre-commit` | pre-commit all files |
| `make clean` | xoa pycache/.pytest_cache |

## Tests

Test files:

- `tests/test_dataset.py`
- `tests/test_flip3d.py`
- `tests/test_loss.py`
- `tests/test_model.py`
- `tests/test_pipeline.py`
- `tests/test_visualization.py`

Pham vi test:

- dataset shape, bbox, image exists, nose above hip trong he y_down;
- keypoint definitions, MHR70 indices unique, flip double-reversible;
- Flip3DKeypoints dong bo 2D/3D khi flip;
- structure loss finite va co gradient;
- model registry/build/output dims;
- full loss step va predict step offline CPU, backbone download tat;
- visualization render duoc file PNG.

Lenh:

```bash
make test
make lint
```

Neu chi can smoke test model:

```bash
PYTHONPATH=src uv run pytest tests/test_model.py -v
```

