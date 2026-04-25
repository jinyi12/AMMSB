"""FAE Publication Figures — SIAM journal submission.

Generates the complete set of publication-quality figures for the FAE
multiscale RFF and optimizer ablation studies.  Follows the visual style
of ``scripts/fae/tran_evaluation/report.py`` (7-inch figures, consistent
fonts, PNG + PDF output).

Three experiment families are covered:

  Optimizer Study (§ Opt)   —  fixed architecture (FiLM, σ = {1,2,4,8}),
                                varying Adam/Muon × L2/NTK-scaled loss.
                                NTK-scaled loss has different magnitude so
                                training curves are shown on separate panels.

  Scale Study (§ Scale)     —  fixed architecture + Muon + L2, comparing
                                single-scale (σ = 1) vs multiscale (σ = {1,2,4,8}).

    Architecture Study (§ Arch) — fixed Muon + σ = {1,2,4,8}, comparing
                                Muon+$\ell_2$ vs Adam+$\ell_2$+Prior.

Usage::

    python scripts/fae/fae_publication_figures.py [--out-dir results/publication]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmsfm.fae.fae_latent_utils import (
    load_fae_checkpoint,
    build_fae_from_checkpoint,
    make_fae_apply_fns,
)
from data.transform_utils import load_transform_info, apply_inverse_transform
from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    publication_figure_width,
    publication_style_tokens,
)


# ============================================================================
# Registry — all experiment runs used in the paper
# ============================================================================

#: (label, run_dir, group, linestyle, color)
#: Convention (docs/publication_figures.md §3.1):
#:   linestyle: solid ("-") = Adam,  dashed ("--") = Muon
#:   hatch on bars: none = Adam,  "//" = Muon
#:   color: one unique EasternHues index per (optimizer × loss) combination
RUNS: List[Tuple[str, str, str, str, str]] = [
    # ── Adam family (solid lines, no hatch) ─────────────────────────────────
    (
        r"Adam + $\ell_2$",
        "results/fae_film_adam_l2_latent128/run_g0ysv6bb",
        "l2",
        "-",
        EASTERN_HUES[4],   # red
    ),
    (
        "Adam + NTK",
        "results/fae_film_adam_ntk_99pct_latent128/run_sphluzvp",
        "ntk",
        "-",
        EASTERN_HUES[0],   # gold
    ),
    (
        r"Adam + $\ell_2$ + Prior",
        "results/fae_film_adam_prior_latent128/run_mgn5f93n",
        "prior",
        "-",
        EASTERN_HUES[1],   # brown
    ),
    (
        "Adam + NTK + Prior",
        "results/fae_film_adam_ntk_prior_latent128/run_uae85cd8",
        "ntk_prior",
        "-",
        EASTERN_HUES[5],   # dark brown
    ),
    # ── Muon family (dashed lines, "//" hatch) ───────────────────────────────
    (
        r"Muon + $\ell_2$",
        "results/fae_film_muon_l2_latent128/run_4cyupstm",
        "l2",
        "--",
        EASTERN_HUES[2],   # deep green
    ),
    (
        "Muon + NTK",
        "results/fae_film_muon_ntk_99pct_latent128/run_ea5yckkq",
        "ntk",
        "--",
        EASTERN_HUES[3],   # teal
    ),
    (
        r"Muon + $\ell_2$ + Prior",
        "results/fae_film_muon_prior_latent128/run_xn2xd51y",
        "prior",
        "--",
        EASTERN_HUES[6],   # terracotta
    ),
    (
        "Muon + NTK + Prior",
        "results/fae_film_muon_ntk_prior_latent128/run_vq1adonq",
        "ntk_prior",
        "--",
        EASTERN_HUES[7],   # steel blue
    ),
]

# Short tags used for table rows / tick labels
SHORT_LABEL: Dict[str, str] = {
    "results/fae_film_adam_l2_latent128/run_g0ysv6bb":            r"Adam+$\ell_2$",
    "results/fae_film_adam_ntk_99pct_latent128/run_sphluzvp":     "Adam+NTK",
    "results/fae_film_adam_prior_latent128/run_mgn5f93n":         r"Adam+$\ell_2$+Prior",
    "results/fae_film_adam_ntk_prior_latent128/run_uae85cd8":     "Adam+NTK+Prior",
    "results/fae_film_muon_l2_latent128/run_4cyupstm":            r"Muon+$\ell_2$",
    "results/fae_film_muon_ntk_99pct_latent128/run_ea5yckkq":     "Muon+NTK",
    "results/fae_film_muon_prior_latent128/run_xn2xd51y":         r"Muon+$\ell_2$+Prior",
    "results/fae_film_muon_ntk_prior_latent128/run_vq1adonq":     "Muon+NTK+Prior",
}


# ============================================================================
# Style constants (shared publication scale)
# ============================================================================
_PUB_STYLE = publication_style_tokens()
FIG_WIDTH = publication_figure_width(column_span=2)
SUBPLOT_HEIGHT = 2     # inches per row
FIELD_ROW_HEIGHT = 1.25

FONT_TITLE  = _PUB_STYLE["font_title"]
FONT_LABEL  = _PUB_STYLE["font_label"]
FONT_LEGEND = _PUB_STYLE["font_legend"]
FONT_TICK   = _PUB_STYLE["font_tick"]

C_OBS    = EASTERN_HUES[7]   # steel blue
C_GEN    = EASTERN_HUES[4]   # red
C_FILL   = EASTERN_HUES[6]   # terracotta
C_GRID   = "#cccccc"
CMAP_FIELD = "cividis"

# EMA decay for training-curve smoothing
EMA_ALPHA = 0.15

# Default H-schedule for the updated Tran inclusion dataset (9 marginals, indices 0–8).
# Maps normalised time fractions t/T = i/8 to filter-width H values.
LEGACY_DEFAULT_H_SCHEDULE = [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0]
DEFAULT_H_SCHEDULE = [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

# Groups included in cross-model comparison figures.
# All four loss-type groups (l2, ntk, prior, ntk_prior) are included.
COMPARISON_GROUPS = {"l2", "ntk", "prior", "ntk_prior"}


# ============================================================================
# I/O helpers
# ============================================================================

def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save figure as PNG (dpi=150) and PDF (vector)."""
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _set_tick_fontsize(ax: plt.Axes, size: float = FONT_TICK) -> None:
    ax.tick_params(axis="both", labelsize=size)


def _apply_sci_yticks(ax: plt.Axes) -> None:
    """Apply ×10^n scientific notation to linear y-axes (matches compare_latent_geometry_models.py)."""
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)


def add_column_cbar_horizontal(fig, axes_col, mappable,
                               pad=0.008, height=0.012,
                               fontsize=FONT_TICK):
    """
    Add a horizontal colorbar below a column of axes.
    axes_col: list of Axes (all rows in one column)
    """
    fig.canvas.draw()

    bbs = [ax.get_position() for ax in axes_col]
    x0 = min(bb.x0 for bb in bbs)
    x1 = max(bb.x1 for bb in bbs)
    y0 = min(bb.y0 for bb in bbs)

    cax = fig.add_axes([
        x0,            # align with column left
        y0 - pad - height,  # just below column
        x1 - x0,       # same width as column
        height         # thin horizontal bar
    ])

    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=fontsize)
    return cbar

def _safe_key(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip()).strip("_").lower()
    return key if key else "run"


def _normalize_mathtext_label(label: str) -> str:
    """Normalize over-escaped mathtext sequences from serialized labels.

    Some pipelines persist labels like ``$\\\\ell_2$`` in npz metadata.
    Matplotlib mathtext expects ``$\\ell_2$``, so collapse repeated
    backslashes before rendering legends/titles.
    """
    s = str(label)
    while "\\\\" in s:
        s = s.replace("\\\\", "\\")
    return s


def _select_default_h_schedule(
    *,
    n_marginals: Optional[int] = None,
    t_frac: Optional[float] = None,
) -> list[float]:
    if n_marginals == len(DEFAULT_H_SCHEDULE):
        return DEFAULT_H_SCHEDULE
    if n_marginals == len(LEGACY_DEFAULT_H_SCHEDULE):
        return LEGACY_DEFAULT_H_SCHEDULE
    if t_frac is not None:
        candidates = [DEFAULT_H_SCHEDULE, LEGACY_DEFAULT_H_SCHEDULE]
        return min(
            candidates,
            key=lambda schedule: float(
                np.min(np.abs(np.linspace(0.0, 1.0, len(schedule)) - float(t_frac)))
            ),
        )
    return DEFAULT_H_SCHEDULE


def _time_fraction_to_H(t_frac: float, *, n_marginals: Optional[int] = None) -> float:
    """Map a normalised time fraction t/T to the corresponding H value."""
    schedule = _select_default_h_schedule(n_marginals=n_marginals, t_frac=t_frac)
    n = len(schedule)
    idx = round(t_frac * (n - 1))
    idx = max(0, min(idx, n - 1))
    return schedule[idx]


def _marginal_value_to_H(t_val: float, *, tidx: Optional[int] = None, n_marginals: Optional[int] = None) -> float:
    """Map a stored marginal key value to the physical filter width H."""
    schedule = _select_default_h_schedule(
        n_marginals=n_marginals,
        t_frac=float(t_val) if 0.0 <= float(t_val) <= 1.0 else None,
    )
    if tidx is not None and n_marginals == len(schedule):
        idx = max(0, min(int(tidx), len(schedule) - 1))
        return schedule[idx]
    if 0.0 <= float(t_val) <= 1.0:
        return _time_fraction_to_H(float(t_val), n_marginals=n_marginals)
    if any(abs(float(t_val) - float(h)) < 1e-8 for h in schedule):
        return float(t_val)
    return float(t_val)


def _H_label(t_frac: float) -> str:
    """Return a LaTeX H-band label for a normalised time fraction."""
    H = _marginal_value_to_H(t_frac)
    if H == int(H):
        return "$H = %g$" % H
    return "$H = %.2f$" % H


def _H_val_label(H_val: float) -> str:
    """Return a LaTeX H-band label for a filter-width H value (not a fraction)."""
    if H_val == int(H_val):
        return "$H = %g$" % H_val
    return "$H = %.2f$" % H_val


# ============================================================================
# Data loaders
# ============================================================================

def _load_training_loss(run_dir: str) -> Optional[NDArray]:
    p = os.path.join(run_dir, "logs", "training_loss.npy")
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)


def _load_eval(run_dir: str) -> Optional[dict]:
    # Prefer full re-evaluation data over training-time metrics
    for fname in ("eval_results_full.json", "eval_results.json"):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def _ema_smooth(y: NDArray, alpha: float = EMA_ALPHA) -> NDArray:
    """Exponential moving average smoothing."""
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _resolve_checkpoint_path(run_dir: str) -> Optional[Path]:
    """Return best available checkpoint path for a run directory."""
    candidates = [
        Path(run_dir) / "checkpoints" / "best_state.pkl",
        Path(run_dir) / "checkpoints" / "state.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _build_model_io_from_run(run_dir: str, *, decode_mode: str = "standard"):
    """Build (encode_fn, decode_fn, metadata) from a run checkpoint."""
    ckpt_path = _resolve_checkpoint_path(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    ckpt = load_fae_checkpoint(ckpt_path)
    autoencoder, params, batch_stats, meta = build_fae_from_checkpoint(ckpt)
    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        params,
        batch_stats,
        decode_mode=decode_mode,
    )
    return encode_fn, decode_fn, meta


def _get_dataset_for_run(run_dir: str) -> dict:
    """Load dataset npz associated with a run via args.json."""
    args_path = Path(run_dir) / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in {run_dir}")
    with args_path.open() as f:
        args = json.load(f)
    data_path = args.get("data_path", "")
    if not data_path:
        raise ValueError(f"Run {run_dir} has no data_path in args.json")
    if not os.path.isabs(data_path):
        data_path = str((Path(run_dir).parents[2] / data_path).resolve())
    data = np.load(data_path, allow_pickle=True)
    return {
        "path": data_path,
        "npz": data,
        "args": args,
    }


def _list_marginal_keys(npz_data) -> List[str]:
    return sorted(
        [k for k in npz_data.keys() if str(k).startswith("raw_marginal_")],
        key=lambda k: float(str(k).replace("raw_marginal_", "")),
    )


def _radial_psd(field_2d: NDArray[np.float32]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute radially averaged 2D PSD for one field image."""
    f = np.asarray(field_2d, dtype=np.float64)
    fft = np.fft.fft2(f)
    psd2d = (np.abs(fft) ** 2)
    psd2d = np.fft.fftshift(psd2d)

    h, w = psd2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = min(cy, cx)

    radial = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)
    for r in range(max_r):
        m = (rr == r)
        c = np.count_nonzero(m)
        if c > 0:
            radial[r] = float(psd2d[m].mean())
            counts[r] = c

    # normalized radial frequency in [0, 0.5]
    freqs = np.arange(max_r, dtype=np.float64) / float(max(h, w))
    return freqs, radial


def _safe_log_psd_distance(psd_ref: NDArray, psd_cmp: NDArray) -> float:
    eps = 1e-12
    a = np.log(np.asarray(psd_ref, dtype=np.float64) + eps)
    b = np.log(np.asarray(psd_cmp, dtype=np.float64) + eps)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ============================================================================
# Figure 1: Training convergence — Optimizer ablation (σ = {1,2,4,8})
#   Two-panel: (a) L2-loss methods, (b) NTK-scaled methods.
#   NTK-scaled loss has different magnitude so curves are not comparable
#   on the same y-axis.
# ============================================================================

def fig1_training_convergence(out_dir: Path, max_steps: int = 50_000) -> None:
    """Training loss curves — one panel per loss family (four panels).

    Panel 1: L2 loss        — Adam+L2,       Muon+L2.
    Panel 2: NTK loss       — Adam+NTK,      Muon+NTK.
    Panel 3: L2 + Prior     — Adam+Prior,    Muon+Prior.
    Panel 4: NTK + Prior    — Adam+NTK+Prior, Muon+NTK+Prior.

    Each panel has its own y-axis (different loss magnitudes).
    No descriptive subplot titles per publication conventions (§3.2).
    Smoothed curve (solid/dashed = Adam/Muon) over translucent raw fill.
    """
    l2_runs       = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "l2"]
    ntk_runs      = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "ntk"]
    prior_runs    = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "prior"]
    ntk_prior_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "ntk_prior"]

    fig, axes = plt.subplots(
        1, 4, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT), sharey=False,
    )

    def _plot_panel(ax: plt.Axes, run_list: list) -> None:
        for label, d, ls, color in run_list:
            loss = _load_training_loss(d)
            if loss is None:
                continue
            n = len(loss)
            steps = np.linspace(0, max_steps, n)
            smoothed = _ema_smooth(loss)
            ax.plot(steps / 1e3, smoothed, linestyle=ls, color=color,
                    linewidth=1.6, label=label, zorder=3)
            ax.fill_between(steps / 1e3, loss * 0.97, loss * 1.03,
                            color=color, alpha=0.08, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("Training step ($\\times 10^3$)", fontsize=FONT_LABEL)
        ax.set_ylabel("Training loss", fontsize=FONT_LABEL)
        ax.grid(which="both", alpha=0.25, color=C_GRID)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
        _set_tick_fontsize(ax)

    _plot_panel(axes[0], l2_runs)
    _plot_panel(axes[1], ntk_runs)
    _plot_panel(axes[2], prior_runs)
    _plot_panel(axes[3], ntk_prior_runs)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig1_training_convergence")


# ============================================================================
# Figure 2: Test performance bar chart — Optimizer + Scale studies
# ============================================================================

def fig2_performance_bars(out_dir: Path) -> None:
    """Bar chart of test relative MSE across core model combinations.

    Two bar groups displayed side by side:
      (a) Optimizer study: fixed σ = {1,2,4,8}, vary optimizer/loss.
      (b) Scale study: fixed Muon + L2, vary σ.
    
    """
    # ── Gather data ──────────────────────────────────────────────────────────
    opt_data = []   # (label, rel_mse, color, run_dir, linestyle)
    for label, d, grp, ls, color in RUNS:
        if grp not in COMPARISON_GROUPS:
            continue
        ev = _load_eval(d)
        if ev is None:
            continue
        opt_data.append((label, float(ev["test_rel_mse"]), color, d, ls))

    # Deduplicate by run directory so each method appears once.
    seen: set[str] = set()
    unique: list = []
    for item in opt_data:
        key = item[3]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    opt_data = unique

    labels = [x[0] for x in opt_data]
    vals   = [x[1] for x in opt_data]
    colors = [x[2] for x in opt_data]
    # Hatch convention: none = Adam (solid line), "//" = Muon (dashed line)
    hatches = ["//" if x[4] == "--" else "" for x in opt_data]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH*0.7, SUBPLOT_HEIGHT))
    x = np.arange(n)
    bars = []
    for i in range(n):
        b = ax.bar(x[i], vals[i], color=colors[i], edgecolor="k", linewidth=0.7,
                   hatch=hatches[i], width=0.55, zorder=3)
        bars.append(b[0])

    # Annotate bars with values.
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.0008,
            "%.4f" % v,
            ha="center", va="bottom",
            fontsize=FONT_TICK - 0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL, rotation=25, ha="right")
    ax.set_ylabel(r"$\|\widehat{u}-u\|_2^2\,/\,\|u\|_2^2$", fontsize=FONT_LABEL)
    ax.set_ylim(0, max(vals) * 1.20)
    ax.grid(axis="y", alpha=0.3, color=C_GRID)
    _apply_sci_yticks(ax)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig2_performance_bars")


# ============================================================================
# Figure 3: Performance vs observation time
# ============================================================================

def fig3_observation_time(out_dir: Path) -> None:
    """Relative MSE as grouped bars over methods, one subplot per H.

    ``training_time_results`` covers held-in observation times and
    ``held_out_results`` covers held-out scales. Both are keyed by
    normalised observation time and are mapped here to physical filter width H.
    """
    # Select key runs for this figure: the eight core deterministic combinations.
    sel_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                if grp in COMPARISON_GROUPS]
    seen_dirs: set = set()
    dedup: list = []
    for x in sel_runs:
        if x[1] not in seen_dirs:
            seen_dirs.add(x[1])
            dedup.append(x)
    sel_runs = dedup

    records = []
    all_H_entries: list[tuple[float, bool]] = []
    for label, d, ls, color in sel_runs:
        ev = _load_eval(d)
        if ev is None:
            continue
        merged: dict[float, tuple[float, bool]] = {}
        tt = ev.get("training_time_results", {})
        for key, payload in tt.items():
            t = float(key)
            H_val = _marginal_value_to_H(t)
            merged[H_val] = (float(payload["rel_mse"]), False)
        ho = ev.get("held_out_results", {})
        for key, payload in ho.items():
            t = float(key)
            H_val = _marginal_value_to_H(t)
            merged[H_val] = (float(payload["rel_mse"]), True)
        if not merged:
            continue
        H_items = sorted(merged.items(), key=lambda item: item[0])
        H_vals = [float(h) for h, _ in H_items]
        rels = [float(val[0]) for _, val in H_items]
        is_held_out = [bool(val[1]) for _, val in H_items]
        records.append((label, d, ls, color, H_vals, rels, is_held_out))
        all_H_entries.extend((h, ho_flag) for h, ho_flag in zip(H_vals, is_held_out))

    if not records:
        print("  [fig3] no training_time_results found, skipping.")
        return

    H_order = [float(h) for h in DEFAULT_H_SCHEDULE[1:]]
    H_is_held_out = {float(h): False for h in H_order}
    for h, ho_flag in all_H_entries:
        if float(h) in H_is_held_out:
            H_is_held_out[float(h)] = H_is_held_out[float(h)] or bool(ho_flag)
    n_panels = len(H_order)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH * 0.8, 1.5 * n_rows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    x = np.arange(len(records), dtype=np.float64)
    legend_handles = []
    for label, d, ls, color, *_rest in records:
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="k",
                linewidth=0.7,
                hatch="//" if "Muon" in label else "",
                label=SHORT_LABEL.get(d, label),
            )
        )

    y_max = 0.0
    flat_axes = list(axes.flat)
    for ax, H_val in zip(flat_axes, H_order):
        vals: list[float] = []
        colors: list[str] = []
        hatches: list[str] = []
        for label, d, ls, color, H_vals, rels, _is_held_out in records:
            rel = float("nan")
            for h_i, r_i in zip(H_vals, rels):
                if abs(h_i - H_val) < 1e-8:
                    rel = float(r_i)
                    break
            vals.append(rel)
            colors.append(color)
            hatches.append("//" if "Muon" in label else "")

        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            y_max = max(y_max, max(finite_vals))

        for i, val in enumerate(vals):
            if not np.isfinite(val):
                continue
            ax.bar(
                x[i], val, width=0.5, color=colors[i], edgecolor="k",
                linewidth=0.7, hatch=hatches[i], zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ho_tag = " (HO)" if H_is_held_out.get(float(H_val), False) else ""
        ax.set_title(_H_val_label(H_val) + ho_tag, fontsize=FONT_LABEL)
        ax.grid(axis="y", alpha=0.25, color=C_GRID)
        _set_tick_fontsize(ax)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    for ax in flat_axes[:n_panels]:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, max(1e-4, y_max * 1.25))
    for row_axes in axes:
        row_axes[0].set_ylabel(r"$\|\widehat{u}-u\|_2^2\,/\,\|u\|_2^2$", fontsize=FONT_LABEL)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 4),
        fontsize=FONT_LEGEND,
        framealpha=0.9,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save_fig(fig, out_dir, "fig3_observation_time")


# ============================================================================
# Figure 4: Mean held-out relative MSE — interpolation summary
# ============================================================================

def fig4_held_out(out_dir: Path) -> None:
    """Mean relative MSE across held-out filter scales (interpolation score).

    ``held_out_results`` stores performance at H-bands not seen during
    training (H ≈ 1.25 and H ≈ 2.50).  This figure collapses those into a
    single mean held-out relative MSE per method — directly comparable to
    ``test_rel_mse`` in fig2 but restricted to unseen scales.

    No error bars: each ``rel_mse`` entry is already a mean over the test
    set; the variance across only 2 H-bands is not informative.
    """
    ho_data = []   # (label, mean_ho_rel_mse, color, run_dir, linestyle)
    for label, d, grp, ls, color in RUNS:
        if grp not in COMPARISON_GROUPS:
            continue
        ev = _load_eval(d)
        if ev is None or not ev.get("held_out_results"):
            continue
        ho = ev["held_out_results"]
        keys = sorted(ho.keys(), key=float)
        band_vals = [float(ho[k]["rel_mse"]) for k in keys
                     if np.isfinite(float(ho[k]["rel_mse"]))]
        if not band_vals:
            continue
        mean_ho = float(np.mean(band_vals))
        ho_data.append((label, mean_ho, color, d, ls))

    # Deduplicate by run directory
    seen: set = set()
    unique: list = []
    for item in ho_data:
        if item[3] not in seen:
            seen.add(item[3])
            unique.append(item)
    ho_data = unique

    if not ho_data:
        print("  [fig4] no held_out_results found, skipping.")
        return

    labels = [x[0] for x in ho_data]
    vals   = [x[1] for x in ho_data]
    colors = [x[2] for x in ho_data]
    hatches = ["//" if x[4] == "--" else "" for x in ho_data]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.7, SUBPLOT_HEIGHT * 0.8))
    x = np.arange(n)
    bars = []
    for i in range(n):
        b = ax.bar(x[i], vals[i], color=colors[i], edgecolor="k", linewidth=0.7,
                   hatch=hatches[i], width=0.55, zorder=3)
        bars.append(b[0])

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.04,
            "%.2e" % v,
            ha="center", va="bottom",
            fontsize=FONT_TICK - 0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL, rotation=25, ha="right")
    ax.set_ylabel(r"$\|\widehat{u}-u\|_2^2\,/\,\|u\|_2^2$", fontsize=FONT_LABEL)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3, color=C_GRID)
    _apply_sci_yticks(ax)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig4_held_out")


# ============================================================================
# Figure 5: Reconstruction fields — per-sample, t1–t7 (excluding t0)
# ============================================================================

def fig5_reconstruction_fields(out_dir: Path) -> None:
    """Professional reconstruction figures split by sample and split type.

    Produces separate figures for each sample (3 samples) and for each split:
      - ``fig5_sample{idx}_train_fields``
      - ``fig5_sample{idx}_heldout_fields``

    Each figure has one column per time point in that split and two rows:
    original (top) and reconstruction (bottom).  Time index ``t0`` is excluded.
    """
    # Proposed method used for qualitative reconstruction panels.
    run_dir = "results/fae_film_adam_ntk_prior_latent128/run_uae85cd8"

    try:
        encode_fn, decode_fn, meta = _build_model_io_from_run(run_dir, decode_mode="standard")
        ds = _get_dataset_for_run(run_dir)
    except Exception as exc:
        print(f"  [fig5] failed to load checkpoint/dataset: {exc}")
        return

    data = ds["npz"]
    grid_coords = np.asarray(data["grid_coords"], dtype=np.float32)
    resolution = int(data["resolution"])
    marginal_keys = _list_marginal_keys(data)
    transform_info = load_transform_info(data)

    held_out_indices = [int(i) for i in np.asarray(data.get("held_out_indices", []), dtype=np.int32)]
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        held_out_indices = sorted(set(held_out_indices) | {0})
    ho_set = set(held_out_indices)

    # Build H-band column labels from marginal time values.
    marginal_t_vals = [float(str(k).replace("raw_marginal_", "")) for k in marginal_keys]

    sample_ids = [0, 1, 2]

    def _reconstruct_single(field_flat: NDArray[np.float32]) -> NDArray[np.float32]:
        u = field_flat[None, :, None].astype(np.float32)
        x = np.broadcast_to(grid_coords[None, :, :], (1, grid_coords.shape[0], grid_coords.shape[1])).astype(np.float32)
        z = encode_fn(u, x)
        u_hat = decode_fn(z, x)
        return np.asarray(u_hat[0, :, 0], dtype=np.float32)

    # Collect reconstructions per sample, split by train/held-out and excluding t0.
    sample_records: Dict[int, Dict[str, List[Tuple[int, float, NDArray[np.float32], NDArray[np.float32]]]]] = {
        s: {"train": [], "held_out": []} for s in sample_ids
    }

    for tidx, key in enumerate(marginal_keys):
        if tidx == 0:
            continue
        t_val = marginal_t_vals[tidx]
        split = "held_out" if tidx in ho_set else "train"
        fields = np.asarray(data[key], dtype=np.float32)

        for sid in sample_ids:
            if sid >= fields.shape[0]:
                continue
            orig = fields[sid]
            recon = _reconstruct_single(orig)
            # Inverse-transform to physical scale for display.
            orig_phys = apply_inverse_transform(orig[None, :], transform_info)[0]
            recon_phys = apply_inverse_transform(recon[None, :], transform_info)[0]
            sample_records[sid][split].append((tidx, t_val, orig_phys, recon_phys))

    def _col_label(tidx: int, t_val: float, is_ho: bool) -> str:
        ho_tag = " (HO)" if is_ho else ""
        H_val = _marginal_value_to_H(t_val, tidx=tidx, n_marginals=len(marginal_keys))
        return _H_val_label(H_val) + ho_tag

    def _plot_sample_split(sample_id: int, split: str, entries: List[Tuple[int, float, NDArray, NDArray]]) -> None:
        if not entries:
            return
        entries = sorted(entries, key=lambda x: x[0])
        n_cols = len(entries)
        # Scale figure width proportionally so held-out panels (few columns)
        # are not stretched to full FIG_WIDTH.
        col_width = min(FIG_WIDTH / max(n_cols, 1), FIG_WIDTH / 5)
        fig_w = col_width * n_cols + 0.6  # 0.6 for ylabel padding
        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=(fig_w, FIELD_ROW_HEIGHT * 1.35),
            squeeze=False,
        )

        for col, (tidx, t_val, orig, recon) in enumerate(entries):
            o2 = np.asarray(orig, dtype=np.float32).reshape(resolution, resolution)
            r2 = np.asarray(recon, dtype=np.float32).reshape(resolution, resolution)
            vmin = float(min(o2.min(), r2.min()))
            vmax = float(max(o2.max(), r2.max()))

            axes[0, col].imshow(o2, origin="lower", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
            axes[0, col].set_title(
                _col_label(tidx, t_val, tidx in ho_set),
                fontsize=FONT_TITLE,
            )
            axes[0, col].axis("off")

            axes[1, col].imshow(r2, origin="lower", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel(
            "GT",
            fontsize=FONT_LABEL,
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )
        axes[1, 0].set_ylabel(
            "Recon",
            fontsize=FONT_LABEL,
            rotation=0,
            ha="right",
            va="center",
            labelpad=16,
        )

        fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.06,
                            hspace=0.03, wspace=0.03)
        _save_fig(fig, out_dir, f"fig5_sample{sample_id}_{split}_fields")

    for sid in sample_ids:
        _plot_sample_split(sid, "train", sample_records[sid]["train"])
        _plot_sample_split(sid, "held_out", sample_records[sid]["held_out"])




# ============================================================================
# Figure 6: PSD spectral analysis
# ============================================================================

def _psd_time_label(tk: str) -> str:
    """Convert PSD time key like ``t1_train`` to an H-band label.

    Uses ``$H_i$`` format (ordinal index) because the actual filter-width
    value is not stored in the PSD npz.  Consistent with the ``$H=value$``
    convention used in field panels where the value is available.
    """
    parts = str(tk).split("_", 1)
    tidx_str = parts[0]  # e.g. "t1"
    split_raw = parts[1] if len(parts) > 1 else ""
    ho_tag = " (HO)" if "held_out" in split_raw else ""
    return f"$H_{{{tidx_str[1:]}}}${ho_tag}"


def fig6_psd_spectral(out_dir: Path) -> None:
    """Time-resolved PSD analysis (no averaging across times).

    Generates:
      1) ``fig6_psd_spectral_per_time``: per-time PSD overlays.
      2) ``fig6b_psd_mismatch_per_time``: per-time log-PSD mismatch trends.
    """
    psd_path = out_dir / "psd_data.npz"
    if not psd_path.exists():
        print("  [fig6] psd_data.npz not found — run scripts/fae/compute_psd.py first. Skipping.")
        return

    data = np.load(psd_path, allow_pickle=True)

    freqs = data["freqs"]
    labels_arr = data["labels"]
    time_keys = list(data["time_keys"]) if "time_keys" in data.files else []
    if not time_keys:
        print("  [fig6] time_keys not found in psd_data.npz; rerun compute_psd.py. Skipping.")
        return

    labels_text = [_normalize_mathtext_label(str(lbl)) for lbl in labels_arr]
    colors_psd = _resolve_run_colors(labels_text)

    # ------------------------------------------------------------------
    # Figure 6a: per-time PSD overlays
    # ------------------------------------------------------------------
    n = len(time_keys)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for idx, tk in enumerate(time_keys):
        ax = axes[idx // n_cols, idx % n_cols]
        gt_key = f"psd_gt_{tk}"
        if gt_key in data:
            ax.loglog(freqs, data[gt_key], "k-", linewidth=1.8, label="GT", alpha=0.8, zorder=4)

        for i, lbl_str in enumerate(labels_text):
            key = f"psd_{_safe_key(lbl_str)}_{tk}"
            if key not in data:
                continue
            color = colors_psd[i]
            ax.loglog(freqs, data[key], color=color, linewidth=1.1, alpha=0.9, label=lbl_str, zorder=3)

        ax.set_title(_psd_time_label(str(tk)), fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.22, color=C_GRID)
        ax.set_xlabel("$k$", fontsize=FONT_LABEL)
        ax.set_ylabel("PSD", fontsize=FONT_LABEL)
        _set_tick_fontsize(ax)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    handles = [Line2D([0], [0], color="k", lw=1.8, label="GT")]
    for i, lbl_str in enumerate(labels_text):
        handles.append(Line2D([0], [0], color=colors_psd[i], lw=1.3, label=lbl_str))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=FONT_LEGEND, framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, out_dir, "fig6_psd_spectral_per_time")

    # ------------------------------------------------------------------
    # Figure 6b: per-time mismatch curves
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    def _time_order_key(tk: str) -> tuple[int, str]:
        t = int(str(tk).split("_")[0].replace("t", ""))
        return (t, str(tk))

    ordered = sorted([str(tk) for tk in time_keys], key=_time_order_key)
    x = np.arange(len(ordered), dtype=np.float64)

    for i, lbl_str in enumerate(labels_text):
        color = colors_psd[i]
        y = []
        for tk in ordered:
            gt_key = f"psd_gt_{tk}"
            pred_key = f"psd_{_safe_key(lbl_str)}_{tk}"
            if (gt_key not in data) or (pred_key not in data):
                y.append(np.nan)
                continue
            y.append(_safe_log_psd_distance(data[gt_key], data[pred_key]))
        ax2.plot(x, y, marker="o", linewidth=1.2, color=color, label=lbl_str)

    ax2.set_xticks(x)
    ax2.set_xticklabels([_psd_time_label(tk) for tk in ordered], rotation=30, ha="right", fontsize=FONT_TICK)
    ax2.set_ylabel("log-PSD mismatch", fontsize=FONT_LABEL)
    ax2.set_xlabel("Scale band", fontsize=FONT_LABEL)
    ax2.grid(alpha=0.25, color=C_GRID)
    ax2.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax2)
    fig2.tight_layout()
    _save_fig(fig2, out_dir, "fig6b_psd_mismatch_per_time")


# Figures 13-15 removed — superseded by latent-geometry metrics in
# scripts/fae/tran_evaluation/compare_latent_geometry_models.py


# ============================================================================
# Figure 7: Muon sample comparison — Muon+L2 vs Muon+NTK vs Muon+NTK+Prior
# ============================================================================

#: Runs used in the Muon comparison. Order determines row order.
_NTK_COMPARE_RUNS: List[Tuple[str, str, str]] = [
    (r"Muon + $\ell_2$", "results/fae_film_muon_l2_latent128/run_4cyupstm", EASTERN_HUES[2]),
    ("Muon + NTK", "results/fae_film_muon_ntk_99pct_latent128/run_ea5yckkq", EASTERN_HUES[3]),
    ("Muon + NTK + Prior", "results/fae_film_muon_ntk_prior_latent128/run_vq1adonq", EASTERN_HUES[7]),
]


def fig7_ntk_sample_comparison(out_dir: Path) -> None:
    """Side-by-side sample reconstructions comparing Muon ablations.

    Generates two figures per sample:

      ``fig7_sampleN_all_times``
        Rows: Ground truth, Muon+L2, Muon+NTK, Muon+NTK+Prior.
        Columns: H=1.0 to H=6.0 (excluding H=0 microscale).
        Shared colour bar per column for fair visual comparison.

      ``fig7_sampleN_t7_zoom``
        Zoomed single-column comparison at the final (coarsest, H=6.0)
        marginal with per-pixel error maps to highlight the effect of
        Muon NTK / prior ablations on high-frequency detail.

    All panels use the cividis colourmap, consistent fonts, and the
    shared FIG_WIDTH / FONT_* constants.
    """

    # Ensure publication rcParams are applied for this figure family.
    format_for_paper()

    # ── Load dataset (shared across all three runs) ─────────────────────────
    ref_run = _NTK_COMPARE_RUNS[0][1]
    try:
        ds = _get_dataset_for_run(ref_run)
    except Exception as exc:
        print(f"  [fig7] failed to load dataset: {exc}")
        return

    data = ds["npz"]
    grid_coords = np.asarray(data["grid_coords"], dtype=np.float32)
    resolution = int(data["resolution"])
    marginal_keys = _list_marginal_keys(data)
    transform_info = load_transform_info(data)

    held_out_indices = [int(i) for i in np.asarray(data.get("held_out_indices", []), dtype=np.int32)]
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        held_out_indices = sorted(set(held_out_indices) | {0})
    ho_set = set(held_out_indices)
    marginal_t_vals = [float(str(k).replace("raw_marginal_", "")) for k in marginal_keys]

    # ── Build (encode, decode) for each run ─────────────────────────────────
    model_fns: List[Tuple[str, str, object, object]] = []  # (label, color, enc, dec)
    for label, run_dir, color in _NTK_COMPARE_RUNS:
        try:
            enc, dec, _ = _build_model_io_from_run(run_dir, decode_mode="standard")
            model_fns.append((label, color, enc, dec))
        except Exception as exc:
            print(f"  [fig7] skipping {label}: {exc}")

    if not model_fns:
        print("  [fig7] no models loaded, skipping.")
        return

    def _reconstruct(enc_fn, dec_fn, field_flat):
        u = field_flat[None, :, None].astype(np.float32)
        x = np.broadcast_to(
            grid_coords[None, :, :],
            (1, grid_coords.shape[0], grid_coords.shape[1]),
        ).astype(np.float32)
        z = enc_fn(u, x)
        u_hat = dec_fn(z, x)
        return np.asarray(u_hat[0, :, 0], dtype=np.float32)

    def _col_label(tidx: int) -> str:
        t_val = marginal_t_vals[tidx] if tidx < len(marginal_t_vals) else 0.0
        ho_tag = " (HO)" if tidx in ho_set else ""
        H_val = _marginal_value_to_H(t_val, tidx=tidx, n_marginals=len(marginal_keys))
        return _H_val_label(H_val) + ho_tag

    def _inv(flat: NDArray) -> NDArray:
        return apply_inverse_transform(flat[None, :], transform_info)[0]

    sample_ids = [0, 1, 2]

    for sid in sample_ids:
        # entries: list of (tidx, orig_2d_phys, {label: recon_2d_phys})
        entries: List[Tuple[int, NDArray, Dict[str, NDArray]]] = []
        for tidx, key in enumerate(marginal_keys):
            if tidx == 0:
                continue
            fields = np.asarray(data[key], dtype=np.float32)
            if sid >= fields.shape[0]:
                continue
            orig_flat = fields[sid]
            recons: Dict[str, NDArray] = {}
            for label, color, enc_fn, dec_fn in model_fns:
                recon_flat = _reconstruct(enc_fn, dec_fn, orig_flat)
                recons[label] = _inv(recon_flat).reshape(resolution, resolution)
            entries.append(
                (tidx, _inv(orig_flat).reshape(resolution, resolution), recons)
            )
        entries.sort(key=lambda e: e[0])

        if not entries:
            continue

        # ── Figure 7a: all training times, all methods ─────────────────────
        n_cols = len(entries)
        n_rows = 1 + len(model_fns)  # GT + each method
        im_handles = [None] * n_cols # to store one image handle per column for shared colorbar later
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(FIG_WIDTH*1.5, FIELD_ROW_HEIGHT * n_rows),
            squeeze=False,
        )

        row_labels = ["Ground truth"] + [m[0] for m in model_fns]

        for col, (tidx, orig_2d, recons) in enumerate(entries):
            col_stack = [orig_2d] + [recons[m[0]] for m in model_fns]
            norm = Normalize(
                vmin=float(min(f.min() for f in col_stack)),
                vmax=float(max(f.max() for f in col_stack)),
            )

            # Row 0: ground truth
            im_col = axes[0, col].imshow(
                orig_2d,
                origin="lower",
                cmap=CMAP_FIELD,
                norm=norm,
            )
            im_handles[col] = im_col  # store for shared colorbar
            axes[0, col].set_title(_col_label(tidx), fontsize=FONT_TITLE)
            axes[0, col].axis("off")

            # Rows 1+: reconstructions
            for row_i, (label, color, _, _) in enumerate(model_fns, start=1):
                axes[row_i, col].imshow(
                    recons[label],
                    origin="lower",
                    cmap=CMAP_FIELD,
                    norm=norm,
                )
                axes[row_i, col].axis("off")

        # Row labels on the left (horizontal for publication readability)
        for row_i, rl in enumerate(row_labels):
            axes[row_i, 0].set_ylabel(
                rl,
                fontsize=FONT_LABEL,
                rotation=0,
                ha="right",
                va="center",
                labelpad=26,
            )
            axes[row_i, 0].yaxis.set_visible(True)
            axes[row_i, 0].tick_params(left=False, labelleft=False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.12,
                            hspace=0.06, wspace=0.04)
        for col in range(n_cols):
            add_column_cbar_horizontal(fig, list(axes[:, col]), im_handles[col])
        _save_fig(fig, out_dir, f"fig7_sample{sid}_all_times")

        # ── Figure 7b: t7 zoom with error maps ─────────────────────────────
        t7_entry = [e for e in entries if e[0] == 7]
        if not t7_entry:
            t7_entry = [entries[-1]]  # fallback to last time
        tidx_t7, orig_t7, recons_t7 = t7_entry[0]

        n_methods = len(model_fns)
        # Layout: 2 rows × n_methods columns
        # Row 0: method reconstructions
        # Row 1: signed difference maps (recon - GT), centered at zero
        n_cols_zoom = n_methods
        fig_z, axes_z = plt.subplots(
            2, n_cols_zoom,
            figsize=(FIG_WIDTH / 1.5, FIELD_ROW_HEIGHT * 2),
            squeeze=False,
        )

        vmin_r = float(min(orig_t7.min(),
                           min(recons_t7[m[0]].min() for m in model_fns)))
        vmax_r = float(max(orig_t7.max(),
                           max(recons_t7[m[0]].max() for m in model_fns)))

        # Compute signed difference maps for shared zero-centered color scale.
        diff_maps = {}
        for label, color, _, _ in model_fns:
            diff_maps[label] = recons_t7[label] - orig_t7
        diff_absmax = float(max(np.abs(d).max() for d in diff_maps.values()))

        # --- when plotting reconstructions (inside the loop) store one image handle ---
        im_recon = None
        for col_i, (label, color, _, _) in enumerate(model_fns):
            recon = recons_t7[label]
            diff = diff_maps[label]

            # store the first recon image handle for colorbar use
            if im_recon is None:
                im_recon = axes_z[0, col_i].imshow(recon, origin="lower", cmap=CMAP_FIELD,
                                                vmin=vmin_r, vmax=vmax_r)
            else:
                axes_z[0, col_i].imshow(recon, origin="lower", cmap=CMAP_FIELD,
                                        vmin=vmin_r, vmax=vmax_r)
            axes_z[0, col_i].set_title(label, fontsize=FONT_TITLE)
            axes_z[0, col_i].axis("off")

            im = axes_z[1, col_i].imshow(
                diff,
                origin="lower",
                cmap="RdBu_r",
                vmin=-diff_absmax,
                vmax=diff_absmax,
            )
            axes_z[1, col_i].axis("off")

        # --- after plotting, add two colorbars: top (recon) + bottom (diff) ---
        fig_z.subplots_adjust(left=0.10, right=0.93, top=0.95, bottom=0.07,
                            hspace=0.18, wspace=0.06)

        # colorbar for top row (reconstructions)
        if im_recon is not None:
            cax_top = fig_z.add_axes([0.945, 0.58, 0.015, 0.33])   # x, y, width, height
            cbar_top = fig_z.colorbar(im_recon, cax=cax_top)
            cbar_top.ax.tick_params(labelsize=FONT_TICK)
            cbar_top.set_label(r"$\hat{u}$", fontsize=FONT_LABEL)

        # colorbar for diff 
        cax = fig_z.add_axes([0.945, 0.10, 0.015, 0.33])
        cbar = fig_z.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label(r"$\hat{u} - u$", fontsize=FONT_LABEL)
        _save_fig(fig_z, out_dir, f"fig7_sample{sid}_t7_zoom")


# ============================================================================
# Figure 8: MSBM generated fields — multiple realizations at each knot
# ============================================================================

#: Default MSBM run directory for generated field panels.
_MSBM_RUN_DIR: Optional[str] = None  # set by --msbm-run-dir

# Number of generated realizations to display.
_N_GEN_SHOW = 4


def fig8_generated_fields(out_dir: Path) -> None:
    """MSBM backward SDE generated fields at each knot time.

    Rows: GT (row 0), then ``_N_GEN_SHOW`` generated realizations.
    Columns: one per displayed dataset marginal, labelled ``$H=value$`` with ``(HO)`` tags.
    All fields displayed in inverse-transformed physical scale.
    """
    if _MSBM_RUN_DIR is None:
        print("  [fig8] --msbm-run-dir not provided, skipping.")
        return

    from scripts.fae.tran_evaluation.generate import (
        generate_backward_realizations,
    )
    from scripts.fae.tran_evaluation.core import (
        build_default_H_schedule,
        load_ground_truth,
        load_time_index_mapping,
    )

    run_dir = Path(_MSBM_RUN_DIR)
    if not run_dir.is_dir():
        print(f"  [fig8] MSBM run dir not found: {run_dir}")
        return

    # Discover dataset path from args.txt
    args_txt = run_dir / "args.txt"
    if not args_txt.exists():
        print(f"  [fig8] args.txt not found in {run_dir}")
        return
    train_cfg: Dict[str, str] = {}
    for line in args_txt.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        train_cfg[k.strip()] = v.strip()

    ds_path_str = train_cfg.get("data_path", "")
    ds_path = Path(ds_path_str)
    if not ds_path.exists():
        ds_path = REPO_ROOT / ds_path_str
    if not ds_path.exists():
        print(f"  [fig8] dataset not found: {ds_path_str}")
        return

    # Load GT and time indices
    gt = load_ground_truth(ds_path)
    resolution = gt["resolution"]
    time_indices = load_time_index_mapping(run_dir)
    ho_set = {int(idx) for idx in gt.get("held_out_indices", [])}

    # Build H_schedule for labels
    H_meso = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    full_H_schedule = build_default_H_schedule(H_meso, 6.0)

    n_show = max(_N_GEN_SHOW, 1)

    # Generate backward realizations
    try:
        gen = generate_backward_realizations(
            run_dir=run_dir,
            dataset_npz_path=ds_path,
            n_realizations=n_show,
            sample_idx=0,
            seed=42,
            use_ema=True,
            drift_clip_norm=None,
            device=None,
            decode_mode="standard",
        )
    except Exception as exc:
        print(f"  [fig8] generation failed: {exc}")
        return

    trajectory_fields = gen.get("trajectory_fields_phys_all")
    if trajectory_fields is None:
        trajectory_fields = gen.get("trajectory_fields_phys")
    if trajectory_fields is None:
        print("  [fig8] trajectory_fields_phys not available.")
        return
    trajectory_time_indices = gen.get("trajectory_all_time_indices")
    if trajectory_time_indices is None:
        trajectory_time_indices = time_indices

    T_knots = trajectory_fields.shape[0]
    n_real = trajectory_fields.shape[1]
    n_show = min(n_show, n_real)

    # ── Figure: GT + n_show generated realizations ─────────────────────
    n_rows = 1 + n_show
    n_cols = T_knots
    fig_width = FIG_WIDTH * 1.5 if n_cols > 5 else FIG_WIDTH
    # Store one mappable per column so each column can get a shared horizontal colorbar.
    im_handles = [None] * n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, FIELD_ROW_HEIGHT * n_rows),
        squeeze=False,
    )

    for k in range(T_knots):
        ds_idx = int(trajectory_time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else float(ds_idx)
        ho_tag = " (HO)" if ds_idx in ho_set else ""
        col_label = _H_val_label(H_val) + ho_tag

        gt_field = gt["fields_by_index"][ds_idx][0]  # sample 0
        gen_fields_k = trajectory_fields[k, :n_show]  # (n_show, res²)
        col_stack = [gt_field.reshape(resolution, resolution)] + [
            gen_fields_k[r].reshape(resolution, resolution) for r in range(n_show)
        ]
        norm = Normalize(
            vmin=float(min(f.min() for f in col_stack)),
            vmax=float(max(f.max() for f in col_stack)),
        )

        # Row 0: GT
        im_col = axes[0, k].imshow(
            gt_field.reshape(resolution, resolution),
            origin="lower", cmap=CMAP_FIELD, norm=norm,
        )
        im_handles[k] = im_col
        axes[0, k].set_title(col_label, fontsize=FONT_TITLE)
        axes[0, k].axis("off")

        # Rows 1..n_show: generated realizations
        for r in range(n_show):
            axes[r + 1, k].imshow(
                gen_fields_k[r].reshape(resolution, resolution),
                origin="lower", cmap=CMAP_FIELD, norm=norm,
            )
            axes[r + 1, k].axis("off")

    # Row labels
    axes[0, 0].set_ylabel(
        "GT", fontsize=FONT_LABEL, rotation=0, ha="right", va="center", labelpad=16,
    )
    for r in range(n_show):
        axes[r + 1, 0].set_ylabel(
            f"Gen {r + 1}", fontsize=FONT_LABEL, rotation=0,
            ha="right", va="center", labelpad=16,
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.03,
                        hspace=0.06, wspace=0.04)
    for col in range(n_cols):
        add_column_cbar_horizontal(fig, list(axes[:, col]), im_handles[col])
    _save_fig(fig, out_dir, "fig8_generated_fields")


# ============================================================================
# Helper: resolve colours from RUNS registry for metrics labels
# ============================================================================

def _resolve_run_colors(labels: List[str]) -> List[str]:
    """Map metrics labels to RUNS-registry colours with fallback."""
    _run_color_map: Dict[str, str] = {}
    for _, d, _, _, c in RUNS:
        short = SHORT_LABEL.get(d)
        if short:
            _run_color_map[short] = c
    _fallback = list(EASTERN_HUES)

    def _match(lbl: str, idx: int) -> str:
        if lbl in _run_color_map:
            return _run_color_map[lbl]
        for k, v in _run_color_map.items():
            if k in lbl or lbl in k:
                return v
        return _fallback[idx % len(_fallback)]

    return [_match(lbl, i) for i, lbl in enumerate(labels)]


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FAE publication figures for SIAM submission."
    )
    parser.add_argument(
        "--out-dir",
        default="results/publication",
        help="Output directory for all figures and tables (default: results/publication).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50_000,
        help="Total training steps (used for x-axis of training curves).",
    )
    parser.add_argument(
        "--msbm-run-dir",
        type=str,
        default=None,
        help="MSBM training run directory for fig11 (generated field panels). "
             "If omitted, fig11 is skipped.",
    )
    args = parser.parse_args()

    # Set MSBM run dir for fig11 (generated fields).
    global _MSBM_RUN_DIR
    _MSBM_RUN_DIR = args.msbm_run_dir

    # Global publication-style matplotlib defaults (fonts/cmap/mathtext).
    format_for_paper()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {out_dir.resolve()}")

    steps = [
        ("fig1:  Training convergence (4-panel: L2, NTK, Prior, NTK+Prior)",
         lambda: fig1_training_convergence(out_dir, args.max_steps)),
        ("fig2:  Performance bar chart (all 8 latent-128 methods)",
         lambda: fig2_performance_bars(out_dir)),
        ("fig3:  Observation-time curves",
         lambda: fig3_observation_time(out_dir)),
        ("fig4:  Held-out interpolation",
         lambda: fig4_held_out(out_dir)),
        ("fig5: Reconstruction fields (physical scale)",
         lambda: fig5_reconstruction_fields(out_dir)),
        ("fig6: PSD spectral analysis (requires psd_data.npz)",
         lambda: fig6_psd_spectral(out_dir)),
        ("fig7: NTK sample comparison (Adam+L2 vs Adam+NTK vs Muon+NTK+Prior)",
         lambda: fig7_ntk_sample_comparison(out_dir)),
        ("fig8: MSBM generated fields (multiple realizations)",
         lambda: fig8_generated_fields(out_dir)),
    ]

    for desc, fn in steps:
        print(f"  Generating {desc}...")
        try:
            fn()
        except Exception as exc:
            print(f"    WARNING: {desc} failed — {exc}")

    print(f"\nDone.  All figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
