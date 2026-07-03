UV      := uv run
PYTEST  := uv run pytest

DATA_ROOT := data/GT
CONFIG    ?= src/pose24/configs/rtmw3d_l_finetune_pose24_v3.py
WORK_DIR  ?= work_dirs/pose24_v3

.PHONY: help splits test test-flip test-unit test-loss test-viz test-pipeline test-model \
        eval train train-resume train-multi-gpu plot-metrics compare-baseline \
        plot-focal-ablation visualize demo demo-tunnel lint pre-commit clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Data ───────────────────────────────────────────────────────────────────
splits:  ## Generate train/val/test split files under data/GT/splits/
	PYTHONPATH=src $(UV) python tools/make_splits.py --data-root $(DATA_ROOT)

# ── Tests ──────────────────────────────────────────────────────────────────
test: test-unit test-loss test-viz test-pipeline test-model test-flip  ## Run all tests

test-flip:  ## Run Flip3DKeypoints regression tests (2D/3D L-R sync)
	PYTHONPATH=src $(PYTEST) tests/test_flip3d.py -v

test-unit:  ## Run unit tests (keypoint definitions + dataset shapes)
	PYTHONPATH=src $(PYTEST) tests/test_dataset.py -v

test-loss:  ## Run structural-loss unit tests (fast, no model build)
	PYTHONPATH=src $(PYTEST) tests/test_loss.py -v

test-viz:  ## Run visualisation unit tests (render to temp PNG)
	PYTHONPATH=src $(PYTEST) tests/test_visualization.py -v

test-pipeline:  ## Run integration tests (full mmpose pipeline)
	PYTHONPATH=src $(PYTEST) tests/test_pipeline.py -v

test-model:  ## Run model integration tests (build + loss + predict, offline)
	PYTHONPATH=src $(PYTEST) tests/test_model.py -v

# ── Training ───────────────────────────────────────────────────────────────
CKPT_BEST ?= work_dirs/pose24/best_MPJPE_epoch_30.pth
eval:  ## Evaluate best checkpoint on val split
	PYTHONPATH=src $(UV) python tools/eval.py $(CKPT_BEST)

train:  ## Finetune RTMPose3D-L (single GPU, AMP)
	PYTHONPATH=src $(UV) python tools/train.py $(CONFIG) \
	  --work-dir $(WORK_DIR) --amp \
	  --cfg-options data_root=$(DATA_ROOT)

train-resume:  ## Resume training from latest checkpoint (AMP)
	PYTHONPATH=src $(UV) python tools/train.py $(CONFIG) \
	  --work-dir $(WORK_DIR) --amp \
	  --resume \
	  --cfg-options data_root=$(DATA_ROOT)

train-multi-gpu:  ## Finetune on all available GPUs
	PYTHONPATH=src $(UV) python tools/train.py $(CONFIG) \
	  --work-dir $(WORK_DIR) \
	  --launcher pytorch \
	  --cfg-options data_root=$(DATA_ROOT)

plot-metrics:  ## Plot val MPJPE/P-MPJPE per epoch from WORK_DIR's logs → PNG
	PYTHONPATH=src $(UV) python tools/plot_metrics.py $(WORK_DIR)

compare-baseline:  ## Per-joint MPJPE: pretrained RTMPose3D-L gốc vs CKPT_BEST finetune
	PYTHONPATH=src $(UV) python tools/compare_baseline.py \
	  --finetune-ckpt $(CKPT_BEST) \
	  --finetune-config $(CONFIG) \
	  --out $(WORK_DIR)/compare_baseline

FOCAL_BEFORE ?= work_dirs/pose24/20260614_022439
FOCAL_AFTER  ?= work_dirs/pose24/20260614_150720
plot-focal-ablation:  ## Ablation: focal-length fallback f=1145 vs f=2074 (historical runs)
	PYTHONPATH=src $(UV) python tools/plot_focal_ablation.py \
	  --before $(FOCAL_BEFORE) \
	  --after $(FOCAL_AFTER)

# ── Visualisation ──────────────────────────────────────────────────────────
CKPT ?=
NUM  ?= 4
visualize:  ## Render GT vs Pred (2D overlay + 3D). Set CKPT=... for predictions.
	PYTHONPATH=src $(UV) python tools/visualize.py \
	  --num $(NUM) --out vis $(if $(CKPT),--checkpoint $(CKPT),)

# ── Demo ───────────────────────────────────────────────────────────────────
PORT ?= 8888
demo:  ## Launch Streamlit demo (local only)
	PYTHONPATH=src $(UV) streamlit run demo/app.py \
	  --server.port $(PORT) --server.address 0.0.0.0

demo-tunnel:  ## Launch Streamlit demo + public Cloudflare quick tunnel
	@trap 'kill 0' EXIT INT TERM; \
	PYTHONPATH=src $(UV) streamlit run demo/app.py \
	  --server.port $(PORT) --server.address 0.0.0.0 \
	  --server.headless true & \
	sleep 5; \
	$(UV) cloudflared tunnel --url http://localhost:$(PORT); \
	wait

# ── Code quality ───────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	$(UV) ruff check src/ tests/ tools/

pre-commit:  ## Run all pre-commit hooks on all files
	$(UV) pre-commit run --all-files

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:  ## Remove build artefacts and pycache
	find src tests tools -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
