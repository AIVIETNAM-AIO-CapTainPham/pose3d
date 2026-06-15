"""Streamlit demo — RTMPose3D POSE-24 finetuned model.

Features
--------
* Dataset sample browser (val/test/train) OR upload an external image.
* 2D overlay + interactive 3D skeleton (GT vs finetuned Pred).
* Third 3D panel: original RTMPose3D (cocktail14, 133-kpt wholebody, body-17
  subset) vs our finetuned Pred — to see the improvement.

Usage
-----
    cd /home/angle/jupyterlab/DEV/POSE3D
    PYTHONPATH=src uv run streamlit run demo/app.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

# ── path setup (must happen before any pose24 import) ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pose24  # noqa: E402, F401  — registers all custom modules

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from pose24.keypoints import (  # noqa: E402
    NUM_KEYPOINTS,
    POSE24_KEYPOINT_NAMES,
    SKELETON_EDGES,
)

# ── constants ──────────────────────────────────────────────────────────────
CONFIG = str(ROOT / "src/pose24/configs/rtmw3d_l_finetune_pose24.py")
WORK_DIR = ROOT / "work_dirs/pose24"
ORIG_CKPT = ROOT / "work_dirs/original_rtmw3d/rtmw3d-l_cocktail14.pth"

GT_COLOR = "#2ecc40"  # green
PRED_COLOR = "#ff4136"  # red
ORIG_COLOR = "#0074d9"  # blue (original RTMPose3D)

ROOT_IDX = [11, 12]  # left_hip, right_hip

# Body-17 (COCO) — the joints the original wholebody model shares with POSE-24.
# POSE-24 indices 0..16 are exactly the 17 COCO body joints, in the same order
# as the first 17 of the wholebody-133 layout.
BODY17 = list(range(17))
BODY17_EDGES = [(i, j) for (i, j) in SKELETON_EDGES if i < 17 and j < 17]


# ── cached resource loaders ────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset(split: str):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import DATASETS

    init_default_scope("mmpose")
    cfg = Config.fromfile(CONFIG)
    return DATASETS.build(cfg[f"{split}_dataloader"]["dataset"])


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading finetuned model…")
def load_model(checkpoint: str):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS

    init_default_scope("mmpose")
    cfg = Config.fromfile(CONFIG)
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    # PyTorch ≥ 2.6 defaults weights_only=True which breaks mmengine checkpoints.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    dev = _device()
    model.to(dev).eval()
    return model, dev


@st.cache_resource(show_spinner="Loading original RTMPose3D (133-kpt)…")
def load_original_model():
    """Build the stock RTMPose3D-L (cocktail14, 133 wholebody kpts).

    Re-uses our ported classes (identical architecture) — only out_channels and
    the codec default z_range differ from the finetuned config.
    """
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS

    if not ORIG_CKPT.exists():
        return None, None

    init_default_scope("mmpose")
    cfg = Config.fromfile(CONFIG)
    mc = copy.deepcopy(cfg.model)
    mc["backbone"]["init_cfg"] = None
    mc["head"]["out_channels"] = 133
    mc["head"]["loss"] = dict(
        type="KLDiscretLossWithWeight",
        use_target_weight=True,
        beta=10.0,
        label_softmax=True,
    )
    # Original codec: default z_range (2.174), 133 kpts.
    mc["head"]["decoder"] = dict(
        type="SimCC3DLabel",
        input_size=(288, 384, 288),
        sigma=(6.0, 6.93, 6.0),
        simcc_split_ratio=2.0,
        normalize=False,
        use_dark=False,
        root_index=(11, 12),
    )
    model = MODELS.build(mc)
    ckpt = torch.load(str(ORIG_CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    dev = _device()
    model.to(dev).eval()
    return model, dev


# ── inference ──────────────────────────────────────────────────────────────


def _run(model, sample):
    from mmengine.dataset import pseudo_collate

    with torch.no_grad():
        results = model.test_step(pseudo_collate([sample]))
    pred = results[0].pred_instances
    pred_3d = np.asarray(pred.keypoints[0])  # (K,3) camera metres
    pred_2d = np.asarray(pred.transformed_keypoints[0])  # (K,2) image pixels
    return pred_2d, pred_3d


def infer_dataset(model, dataset, idx: int):
    return _run(model, dataset[idx])


def build_sample_from_image(image_bgr: np.ndarray, bbox_xyxy, root_z: float = 2.5):
    """Build a packed data sample from a raw image + person bbox (no GT).

    Injects a fallback ``root_z`` so the estimator's back-projection works.
    """
    from mmengine.registry import init_default_scope
    from mmengine.structures import InstanceData
    from mmpose.registry import TRANSFORMS

    init_default_scope("mmpose")
    h, w = image_bgr.shape[:2]
    results = dict(
        img=image_bgr,
        img_shape=(h, w),
        ori_shape=(h, w),
        img_path="<upload>",
        id=0,
        img_id=0,
        bbox=np.asarray(bbox_xyxy, dtype=np.float32).reshape(1, 4),
        bbox_score=np.ones(1, dtype=np.float32),
    )

    for t in (
        dict(type="GetBBoxCenterScale"),
        dict(type="TopdownAffine", input_size=(288, 384)),
        dict(type="PackPoseInputs"),
    ):
        results = TRANSFORMS.build(t)(results)

    sample = results["data_samples"]
    if not hasattr(sample, "gt_instances") or sample.gt_instances is None:
        sample.gt_instances = InstanceData()
    sample.gt_instances.set_field(np.asarray([root_z], dtype=np.float32), "root_z")
    return dict(inputs=results["inputs"], data_samples=sample)


def infer_image(model, image_bgr, bbox_xyxy, root_z: float = 2.5):
    return _run(model, build_sample_from_image(image_bgr, bbox_xyxy, root_z))


# ── per-joint MPJPE (root-relative) ───────────────────────────────────────


def per_joint_mpjpe(gt_3d, pred_3d, visible):
    gt_root = gt_3d[ROOT_IDX].mean(0)
    pred_root = pred_3d[ROOT_IDX].mean(0)
    errs = np.linalg.norm((pred_3d - pred_root) - (gt_3d - gt_root), axis=-1) * 1000
    errs[~visible] = np.nan
    return errs


# ── matplotlib 2D overlay ─────────────────────────────────────────────────


def fig_2d(image_rgb, gt_2d, pred_2d, visible, edges=SKELETON_EDGES):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(image_rgb)

    def _skel(xy, color, label):
        for i, j in edges:
            if visible[i] and visible[j]:
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    color=color,
                    lw=2,
                    alpha=0.85,
                )
        ax.scatter(
            xy[visible, 0],
            xy[visible, 1],
            c=color,
            s=22,
            edgecolors="white",
            lw=0.5,
            label=label,
            zorder=3,
        )

    if gt_2d is not None:
        _skel(gt_2d, GT_COLOR, "GT")
    if pred_2d is not None:
        _skel(pred_2d, PRED_COLOR, "Pred")

    leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    for txt in leg.get_texts():  # legend text → black
        txt.set_color("black")
    ax.set_title("2D keypoints (image space)", fontsize=11, color="black")
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


# ── plotly interactive 3D (generic multi-skeleton) ─────────────────────────


def _root_centre(pts):
    return pts - pts[ROOT_IDX].mean(0)


def _torso_len(rel):
    # left_hip(11) → left_shoulder(5): both in body-17, always present
    return float(np.linalg.norm(rel[11] - rel[5])) + 1e-6


def fig_3d(skeletons, title: str, edges=SKELETON_EDGES):
    """skeletons: list of (pts(K,3) camera-m, color, name, visible|None).

    All skeletons are root-centred and scale-normalised to the FIRST entry's
    torso length, then displayed upright (camera x,y_down,z_fwd → x, z, -y).
    """
    traces = []
    ref_scale = None
    for pts, color, name, visible in skeletons:
        rel = _root_centre(pts)
        scale = _torso_len(rel)
        if ref_scale is None:
            ref_scale = scale
        rel = rel / scale * ref_scale  # normalise to common torso size
        vis = np.ones(len(pts), bool) if visible is None else visible
        dx, dy, dz = rel[:, 0], rel[:, 2], -rel[:, 1]
        for i, j in edges:
            if vis[i] and vis[j]:
                traces.append(
                    go.Scatter3d(
                        x=[dx[i], dx[j]],
                        y=[dy[i], dy[j]],
                        z=[dz[i], dz[j]],
                        mode="lines",
                        line=dict(color=color, width=5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        names = [
            POSE24_KEYPOINT_NAMES[k] if k < NUM_KEYPOINTS else str(k)
            for k in range(len(pts))
        ]
        traces.append(
            go.Scatter3d(
                x=dx[vis],
                y=dy[vis],
                z=dz[vis],
                mode="markers",
                marker=dict(color=color, size=5, line=dict(color="white", width=0.5)),
                name=name,
                text=[n for n, v in zip(names, vis) if v],
                hovertemplate="<b>%{text}</b><br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) m"
                "<extra></extra>",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X →",
            yaxis_title="Z depth",
            zaxis_title="Up ↑",
            aspectmode="data",
            bgcolor="#f8f9fa",
        ),
        legend=dict(
            x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)", font=dict(color="black")
        ),  # legend text → black
        margin=dict(l=0, r=0, t=36, b=0),
        height=480,
        title=dict(text=title, x=0.5, font=dict(color="black")),
        paper_bgcolor="white",
    )
    return fig


# ── per-joint error bar chart ─────────────────────────────────────────────


def fig_error_bar(errs_mm):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    colors = [PRED_COLOR if not np.isnan(v) else "#dddddd" for v in errs_mm]
    ax.bar(
        POSE24_KEYPOINT_NAMES,
        np.where(np.isnan(errs_mm), 0, errs_mm),
        color=colors,
        edgecolor="white",
        lw=0.5,
    )
    mean_val = np.nanmean(errs_mm)
    ax.axhline(
        mean_val, color="navy", ls="--", lw=1.5, label=f"mean = {mean_val:.1f} mm"
    )
    ax.set_ylabel("Error (mm)", color="black")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    leg = ax.legend(fontsize=9)
    for txt in leg.get_texts():
        txt.set_color("black")
    ax.set_title(
        "Per-joint MPJPE — root-relative, visible joints only",
        fontsize=10,
        color="black",
    )
    fig.tight_layout()
    return fig


# ── epoch viz gallery ─────────────────────────────────────────────────────


def render_epoch_gallery():
    vis_dir = WORK_DIR / "vis"
    epoch_dirs = (
        sorted(vis_dir.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
        if vis_dir.exists()
        else []
    )
    if not epoch_dirs:
        return
    st.subheader("Training-hook visualizations (GT vs Pred per epoch)")
    sel = st.select_slider(
        "Epoch", options=[d.name for d in epoch_dirs], value=epoch_dirs[-1].name
    )
    imgs = sorted((vis_dir / sel).glob("sample_*.png"))
    if imgs:
        for c, im in zip(st.columns(min(4, len(imgs))), imgs[:4]):
            c.image(str(im), use_container_width=True, caption=im.stem)


# ── main ──────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="POSE-24 Demo",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🦴 RTMPose3D — POSE-24 Finetune Demo")
    st.caption("Finetuned (24-kpt) vs original RTMPose3D (133-kpt wholebody)")

    # ── sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        source = st.radio("Input source", ["Dataset sample", "Upload image"])

        ckpts = sorted(WORK_DIR.glob("best_MPJPE_epoch_*.pth")) + sorted(
            WORK_DIR.glob("epoch_*.pth")
        )
        if not ckpts:
            st.error("No finetuned checkpoints in work_dirs/pose24/")
            st.stop()
        names = [p.name for p in ckpts]
        default = (
            names.index("best_MPJPE_epoch_30.pth")
            if "best_MPJPE_epoch_30.pth" in names
            else 0
        )
        ckpt_sel = st.selectbox("Finetuned checkpoint", names, index=default)
        checkpoint = str(WORK_DIR / ckpt_sel)

        show_orig = st.toggle(
            "Compare with original RTMPose3D", value=ORIG_CKPT.exists()
        )
        if show_orig and not ORIG_CKPT.exists():
            st.warning("Original checkpoint not found — download first.")
            show_orig = False

    model, device = load_model(checkpoint)
    orig_model = orig_dev = None
    if show_orig:
        orig_model, orig_dev = load_original_model()

    # ── resolve input → (image_rgb, gt_2d, gt_3d, visible, pred fns) ─────────
    if source == "Dataset sample":
        with st.sidebar:
            split = st.selectbox("Split", ["val", "test", "train"])
        dataset = load_dataset(split)
        n = len(dataset)
        with st.sidebar:
            if st.button("🎲 Random sample"):
                st.session_state["idx"] = int(np.random.randint(0, n))
            idx = st.slider(
                "Index", 0, n - 1, value=st.session_state.get("idx", 0), key="idx"
            )
            st.caption(f"{n} samples")

        info = dataset.get_data_info(idx)
        img_bgr = cv2.imread(info["img_path"])
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_2d = np.asarray(info["keypoints"][0])
        gt_3d = np.asarray(info["keypoints_3d"][0])
        vis_raw = info.get("keypoints_visible", np.ones((1, NUM_KEYPOINTS)))[0]
        visible = np.asarray(vis_raw).flatten()[:NUM_KEYPOINTS] > 0.5
        src_caption = info["img_path"]

        pred_2d, pred_3d = infer_dataset(model, dataset, idx)
        orig_2d = orig_3d = None
        if orig_model is not None:
            orig_2d, orig_3d = infer_dataset(orig_model, dataset, idx)

    else:  # Upload image
        up = st.sidebar.file_uploader(
            "Upload a person image", type=["jpg", "jpeg", "png", "webp"]
        )
        root_z = st.sidebar.slider(
            "Assumed distance (m)",
            1.0,
            6.0,
            2.5,
            0.1,
            help="Only affects absolute scale; the 3D "
            "view is normalised so it rarely matters.",
        )
        if up is None:
            st.info("⬅️ Upload a person-centred image to run inference.")
            st.stop()
        file_bytes = np.frombuffer(up.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        bbox = [0, 0, w, h]  # whole image (assume person crop)
        gt_2d = gt_3d = None
        visible = np.ones(NUM_KEYPOINTS, bool)
        src_caption = f"{up.name}  ({w}×{h})  — whole-image bbox"

        pred_2d, pred_3d = infer_image(model, img_bgr, bbox, root_z)
        orig_2d = orig_3d = None
        if orig_model is not None:
            orig_2d, orig_3d = infer_image(orig_model, img_bgr, bbox, root_z)

    # ── header metrics ──────────────────────────────────────────────────────
    errs = per_joint_mpjpe(gt_3d, pred_3d, visible) if gt_3d is not None else None
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "MPJPE (sample)",
        f"{np.nanmean(errs):.1f} mm" if errs is not None else "— (no GT)",
    )
    c2.metric("Visible joints", f"{int(visible.sum())} / {NUM_KEYPOINTS}")
    c3.metric("Checkpoint", ckpt_sel.replace(".pth", ""))
    st.caption(f"`{src_caption}`")
    st.divider()

    # ── row 1: 2D overlay + 3D (GT vs Pred) ─────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.pyplot(fig_2d(image_rgb, gt_2d, pred_2d, visible), use_container_width=True)
        plt.close("all")
    with col_b:
        skels = []
        if gt_3d is not None:
            skels.append((gt_3d, GT_COLOR, "GT", visible))
        skels.append((pred_3d, PRED_COLOR, "Pred (finetuned)", visible))
        st.plotly_chart(
            fig_3d(skels, "Finetuned: GT (green) vs Pred (red)"),
            use_container_width=True,
        )

    # ── row 2: original RTMPose3D vs our Pred (body-17) ─────────────────────
    if orig_3d is not None:
        st.divider()
        st.subheader("Original RTMPose3D (blue) vs Finetuned Pred (red) — body-17")
        body_vis = visible[:17]
        skels = []
        if gt_3d is not None:
            skels.append((gt_3d[:17], GT_COLOR, "GT", body_vis))
        skels.append((pred_3d[:17], PRED_COLOR, "Pred (finetuned)", body_vis))
        skels.append((orig_3d[:17], ORIG_COLOR, "Original RTMPose3D", body_vis))
        st.plotly_chart(
            fig_3d(
                skels, "Body-17: finetuned (red) vs original (blue)", edges=BODY17_EDGES
            ),
            use_container_width=True,
        )
        st.caption(
            "Chỉ so 17 khớp body chung — model gốc (wholebody) không có "
            "7 khớp giải phẫu (olecranon, cubital fossa, acromion, neck)."
        )

    # ── per-joint error + gallery ───────────────────────────────────────────
    if errs is not None:
        st.divider()
        st.pyplot(fig_error_bar(errs), use_container_width=True)
        plt.close("all")

    st.divider()
    render_epoch_gallery()


if __name__ == "__main__":
    main()
