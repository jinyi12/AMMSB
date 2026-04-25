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

from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    publication_figure_width,
    publication_style_tokens,
)


C_RAW = "0.75"
C_TRAIN = EASTERN_HUES[7]
_PUB_STYLE = publication_style_tokens()
FONT_LABEL = _PUB_STYLE["font_label"]
FONT_TICK = _PUB_STYLE["font_tick"]
FIG_WIDTH = publication_figure_width(column_span=1)
FIG_HEIGHT = 2.35
_BRIDGE_MATCHING_MODEL_TYPES = {"conditional_bridge", "conditional_bridge_token_dit"}
_BRIDGE_MATCHING_OBJECTIVES = {
    "paired_conditional_bridge_matching",
    "paired_conditional_multimarginal_bridge_matching",
}
_STATE_PREDICTION_OBJECTIVES = {"paired_prior_bridge_state_prediction"}


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


def _read_optional_scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(()).item())
    return None


def _infer_loss_label(
    *,
    model_type: str | None,
    training_objective: str | None,
    loss_key: str,
) -> str:
    if loss_key == "state_loss_history":
        return "State prediction loss"
    if training_objective in _BRIDGE_MATCHING_OBJECTIVES:
        return "Bridge matching loss"
    if training_objective in _STATE_PREDICTION_OBJECTIVES:
        return "State prediction loss"
    if model_type in _BRIDGE_MATCHING_MODEL_TYPES:
        return "Bridge matching loss"
    return "ECMMD loss"


def _select_loss_history(data: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray]:
    for key in ("loss_history", "state_loss_history"):
        if key in data:
            return key, np.asarray(data[key], dtype=np.float32)
    available = ", ".join(sorted(data.files))
    raise KeyError(
        "training_summary.npz is missing a supported loss history. "
        f"Expected one of: loss_history, state_loss_history. Available keys: {available}"
    )


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
        loss_key, loss_history = _select_loss_history(data)
        training_seconds = float(np.asarray(data["training_seconds"]).item()) if "training_seconds" in data else None
        model_type = _read_optional_scalar_string(data, "model_type")
        training_objective = _read_optional_scalar_string(data, "training_objective")

    if loss_history.ndim != 1 or loss_history.size == 0:
        raise ValueError(f"loss_history must be a non-empty 1-D array, got {loss_history.shape}")

    window = _auto_smooth_window(int(loss_history.size), int(smooth_window))
    smooth = _moving_average(loss_history, window)
    steps = np.arange(1, loss_history.shape[0] + 1, dtype=np.int64)
    loss_label = _infer_loss_label(
        model_type=model_type,
        training_objective=training_objective,
        loss_key=loss_key,
    )

    format_for_paper()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(steps, loss_history, color=C_RAW, linewidth=0.8, alpha=0.9)
    ax.plot(steps, smooth, color=C_TRAIN, linewidth=1.2)
    ax.set_xlabel("Optimizer step", fontsize=FONT_LABEL)
    ax.set_ylabel(loss_label, fontsize=FONT_LABEL)
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
        "model_type": model_type,
        "training_objective": training_objective,
        "loss_key": loss_key,
        "loss_label": loss_label,
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
