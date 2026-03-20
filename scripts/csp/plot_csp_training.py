from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from scripts.images.field_visualization import EASTERN_HUES, format_for_paper


C_RAW = "0.75"
C_TRAIN = EASTERN_HUES[7]
FONT_LABEL = 7
FONT_TICK = 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-style CSP training convergence.")
    parser.add_argument("--run_dir", type=str, required=True, help="Completed CSP run directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/publication.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=0,
        help="Moving-average window. Use 0 for an automatic choice.",
    )
    return parser.parse_args()


def _default_output_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "publication"


def _auto_smooth_window(n_steps: int, requested: int) -> int:
    if int(requested) > 0:
        return min(int(requested), int(n_steps))
    return max(10, min(200, max(10, int(n_steps) // 100)))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.pad(values.astype(np.float64), (int(window) - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.asarray(smoothed[: values.shape[0]], dtype=np.float32)


def plot_training_curve(
    *,
    run_dir: Path,
    output_dir: Path,
    smooth_window: int = 0,
) -> dict[str, Any]:
    summary_path = run_dir / "metrics" / "training_summary.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")

    with np.load(summary_path, allow_pickle=True) as data:
        loss_history = np.asarray(data["loss_history"], dtype=np.float32)
        training_seconds = float(np.asarray(data["training_seconds"]).item()) if "training_seconds" in data else None

    if loss_history.ndim != 1 or loss_history.size == 0:
        raise ValueError(f"loss_history must be a non-empty 1-D array, got {loss_history.shape}")

    window = _auto_smooth_window(int(loss_history.size), int(smooth_window))
    smooth = _moving_average(loss_history, window)
    steps = np.arange(1, loss_history.shape[0] + 1, dtype=np.int64)

    format_for_paper()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(steps, loss_history, color=C_RAW, linewidth=0.8, alpha=0.9)
    ax.plot(steps, smooth, color=C_TRAIN, linewidth=1.4)
    ax.set_xlabel("Optimizer step", fontsize=FONT_LABEL)
    ax.set_ylabel("ECMMD loss", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_png = output_dir / "fig_csp_training_convergence.png"
    fig_pdf = output_dir / "fig_csp_training_convergence.pdf"
    fig.savefig(fig_png, dpi=180, bbox_inches="tight")
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "figure_png": str(fig_png),
        "figure_pdf": str(fig_pdf),
        "n_steps": int(loss_history.size),
        "smooth_window": int(window),
        "training_seconds": training_seconds,
    }
    (output_dir / "training_curve_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved training figure to {fig_pdf}", flush=True)
    return summary


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(run_dir)
    )
    plot_training_curve(
        run_dir=run_dir,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
