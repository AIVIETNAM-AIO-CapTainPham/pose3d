"""Plot MPJPE / P-MPJPE per epoch from a work_dir's vis_data/scalars.json logs.

Usage:
    PYTHONPATH=src uv run python tools/plot_metrics.py work_dirs/pose24_v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(work_dir: Path) -> dict[int, dict[str, float]]:
    """Read every timestamped run's scalars.json, keep latest record per epoch."""
    by_epoch: dict[int, dict[str, float]] = {}
    run_dirs = sorted(p for p in work_dir.iterdir() if p.is_dir() and p.name[0].isdigit())
    for run_dir in run_dirs:
        scalars_path = run_dir / "vis_data" / "scalars.json"
        if not scalars_path.exists():
            continue
        for line in scalars_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "MPJPE" not in record or "step" not in record:
                continue
            by_epoch[record["step"]] = {
                "MPJPE": record["MPJPE"] * 1000,  # m -> mm
                "P-MPJPE": record["P-MPJPE"] * 1000,
            }
    return by_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics = load_metrics(args.work_dir)
    if not metrics:
        raise SystemExit(f"No MPJPE records found under {args.work_dir}")

    epochs = sorted(metrics)
    mpjpe = [metrics[e]["MPJPE"] for e in epochs]
    p_mpjpe = [metrics[e]["P-MPJPE"] for e in epochs]

    best_epoch = min(epochs, key=lambda e: metrics[e]["MPJPE"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, mpjpe, label="MPJPE", marker=".")
    ax.plot(epochs, p_mpjpe, label="P-MPJPE", marker=".")
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"best epoch {best_epoch}\n{metrics[best_epoch]['MPJPE']:.1f}mm",
        (best_epoch, metrics[best_epoch]["MPJPE"]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("error (mm)")
    ax.set_title(f"{args.work_dir.name} — val MPJPE / P-MPJPE per epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = args.out or args.work_dir / "metrics_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    print(f"Best epoch: {best_epoch}  MPJPE={metrics[best_epoch]['MPJPE']:.2f}mm  "
          f"P-MPJPE={metrics[best_epoch]['P-MPJPE']:.2f}mm")


if __name__ == "__main__":
    main()
