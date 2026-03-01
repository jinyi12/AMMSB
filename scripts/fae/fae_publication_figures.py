"""FAE Publication Figures — SIAM journal submission.

Generates the complete set of publication-quality figures for the FAE
multiscale RFF and optimizer ablation studies.  Follows the visual style
of ``scripts/fae/tran_evaluation/report.py`` (7-inch figures, consistent
fonts, PNG + PDF output).

Four experiment families are covered:

  Optimizer Study (§ Opt)   —  fixed architecture (FiLM, σ = {1,2,4,8}),
                                varying Adam/Muon × L2/NTK-scaled loss.
                                NTK-scaled loss has different magnitude so
                                training curves are shown on separate panels.

  Scale Study (§ Scale)     —  fixed architecture + Muon + L2, comparing
                                single-scale (σ = 1) vs multiscale (σ = {1,2,4,8}).

    Architecture Study (§ Arch) — fixed Muon + σ = {1,2,4,8}, comparing
                                                                Muon+$\ell_2$ vs Adam+$\ell_2$+Prior.

    Denoiser Study (§ Den)    —  Denoiser decoder, single and
                                multiscale.  Separate from deterministic methods
                                because the denoiser uses a different evaluation
                                protocol (iterative denoising) and training loss.

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
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.fae_latent_utils import (
    load_fae_checkpoint,
    build_attention_fae_from_checkpoint,
    make_fae_apply_fns,
)
from scripts.images.field_visualization import format_for_paper


# ============================================================================
# Registry — all experiment runs used in the paper
# ============================================================================

#: (label, run_dir, group, linestyle, color)
RUNS: List[Tuple[str, str, str, str, str]] = [
    # ── Optimizer study (L2 loss): σ = {1, 2, 4, 8} ────────────────────────
    (
        r"Adam + $\ell_2$",
        "results/fae_film_adam_l2_99pct/run_bnqm4evk",
        "opt_l2",
        "-",
        "#d62728",   # red
    ),
    (
        r"Muon + $\ell_2$",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "opt_l2",
        "-",
        "#2ca02c",   # green (best)
    ),
    # ── Optimizer study (NTK-scaled loss): σ = {1, 2, 4, 8} ────────────────
    (
        "Adam + NTK",
        "results/fae_film_adam_ntk_99pct/run_2hnr5shv",
        "opt_ntk",
        "--",
        "#ff7f0e",   # orange
    ),
    (
        "Muon + NTK",
        "results/fae_film_muon_ntk_99pct/run_tug7ucuw",
        "opt_ntk",
        "-.",
        "#9467bd",   # purple
    ),
    # ── Scale study: vary σ, fixed Muon + L2 ────────────────────────────────
    (
        r"Muon + $\ell_2$, $\sigma\!=\!1$",
        "results/fae_deterministic_film/run_90ndogk3",
        "scale",
        "--",
        "#1f77b4",   # blue
    ),
    (
        r"Muon + $\ell_2$",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "scale",
        "-",
        "#2ca02c",   # green
    ),
    # ── Architecture study: Muon+L2 vs Adam+L2+Prior ───────────────────────
    (
        r"Muon + $\ell_2$",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "arch",
        "-",
        "#2ca02c",   # green
    ),
    (
        r"Adam + $\ell_2$ + Prior",
        "results/fae_film_prior_multiscale/run_66nrnp5e",
        "arch",
        "--",
        "#8c564b",   # brown
    ),
    # ── Denoiser study: single-scale and default multiscale ────────────────
    (
        r"Denoiser, $\sigma\!=\!1$",
        "results/fae_denoiser_film_heek/run_ezndnxw0",
        "den",
        "--",
        "#e377c2",   # pink
    ),
    (
        "Denoiser",
        "results/fae_denoiser_film_heek_multiscale/run_9vl5sblh",
        "den",
        "-",
        "#17becf",   # cyan
    ),
    # ── NTK + prior study: film & denoiser with prior_loss_weight=1 ─────
    (
        "Adam + NTK + Prior",
        "results/adam_ntk_prior/run_zaql9zhd",
        "ntk_prior",
        "--",
        "#bcbd22",   # olive
    ),
    (
        "Muon + NTK + Prior",
        "results/muon_ntk_prior/run_r6flmspu",
        "ntk_prior",
        "-",
        "#17becf",   # teal
    ),
    (
        "Denoiser + Adam + NTK + Prior",
        "results/fae_denoiser_adam_ntk_prior/run_kz7gp1ny",
        "den_prior",
        "--",
        "#e377c2",   # pink
    ),
    (
        "Denoiser + Muon + NTK + Prior",
        "results/denoiser_muon_ntk_prior/run_l41wdiei",
        "den_prior",
        "-",
        "#7f7f7f",   # gray
    ),
]

# Short tags used for table rows / tick labels
SHORT_LABEL: Dict[str, str] = {
    "results/fae_film_adam_l2_99pct/run_bnqm4evk":                r"Adam+$\ell_2$",
    "results/fae_film_adam_ntk_99pct/run_2hnr5shv":               "Adam+NTK",
    "results/fae_film_muon_ntk_99pct/run_tug7ucuw":               "Muon+NTK",
    "results/fae_deterministic_film_multiscale/run_ujlkslav":      r"Muon+$\ell_2$*",
    "results/fae_deterministic_film/run_90ndogk3":                 r"Muon+$\ell_2$, $\sigma$=1",
    "results/fae_film_prior_multiscale/run_66nrnp5e":              r"Adam+$\ell_2$+Prior",
    "results/fae_denoiser_film_heek/run_ezndnxw0":                r"Denoiser, $\sigma$=1",
    "results/fae_denoiser_film_heek_multiscale/run_9vl5sblh":     "Denoiser",
    "results/adam_ntk_prior/run_zaql9zhd":                         "Adam+NTK+Prior",
    "results/muon_ntk_prior/run_r6flmspu":                         "Muon+NTK+Prior",
    "results/fae_denoiser_adam_ntk_prior/run_kz7gp1ny":            "Den+Adam+NTK+Prior",
    "results/denoiser_muon_ntk_prior/run_l41wdiei":                "Den+Muon+NTK+Prior",
}


# ============================================================================
# Style constants (mirrors report.py)
# ============================================================================
FIG_WIDTH = 7.0          # inches; all multi-panel figs
SUBPLOT_HEIGHT = 2.5     # inches per row
FIELD_ROW_HEIGHT = 1.75

FONT_TITLE  = 8
FONT_LABEL  = 7
FONT_LEGEND = 6.5
FONT_TICK   = 7

C_OBS    = "#2166ac"
C_GEN    = "#b2182b"
C_FILL   = "#d6604d"
C_GRID   = "#cccccc"

# EMA decay for training-curve smoothing
EMA_ALPHA = 0.15

# Groups included in cross-model comparison figures (exclude denoisers because
# they use a different iterative evaluation protocol).
COMPARISON_GROUPS = {"opt_l2", "opt_ntk", "scale", "arch", "ntk_prior"}


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


def _safe_key(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip()).strip("_").lower()
    return key if key else "run"


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


def _build_model_io_from_run(run_dir: str, *, decode_mode: str = "auto"):
    """Build (encode_fn, decode_fn, metadata) from a run checkpoint."""
    ckpt_path = _resolve_checkpoint_path(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    ckpt = load_fae_checkpoint(ckpt_path)
    autoencoder, params, batch_stats, meta = build_attention_fae_from_checkpoint(ckpt)
    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        params,
        batch_stats,
        decode_mode=decode_mode,
        denoiser_num_steps=32,
        denoiser_noise_scale=1.0,
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
    """Training loss curves for the optimizer ablation study (two panels).

    Panel (a): L2-loss methods (Adam + L2, Muon + L2).
    Panel (b): NTK-scaled methods (Adam + NTK, Muon + NTK).

    Each panel has its own y-axis scale because NTK-scaled loss has
    fundamentally different magnitude than L2 loss.  Smoothed curve
    (solid) over raw values (translucent fill) for each method.
    """
    l2_runs  = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "opt_l2"]
    ntk_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "opt_ntk"]

    fig, (ax_l2, ax_ntk) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT), sharey=False,
    )

    def _plot_panel(ax: plt.Axes, run_list: list, title: str) -> None:
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
        ax.set_ylabel("Training loss (log scale)", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.25, color=C_GRID)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
        _set_tick_fontsize(ax)

    _plot_panel(
        ax_l2, l2_runs,
        r"(a) $\ell_2$ loss",
    )
    _plot_panel(
        ax_ntk, ntk_runs,
        r"(b) NTK-scaled loss",
    )

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

    The colour scheme matches Figure 1.  Bars are annotated with the
    numeric value for easy reading.

    Denoiser runs are excluded (different evaluation protocol).
    """
    # ── Gather data ──────────────────────────────────────────────────────────
    opt_data = []   # (label, rel_mse, color, run_dir)
    for label, d, grp, ls, color in RUNS:
        if grp not in COMPARISON_GROUPS:
            continue
        ev = _load_eval(d)
        if ev is None:
            continue
        opt_data.append((label, float(ev["test_rel_mse"]), color, d))

    # Deduplicate by run directory so each method appears once.
    seen: set[str] = set()
    unique: list = []
    for item in opt_data:
        key = item[3]
        if key not in seen:
            seen.add(key)
            unique.append(item[:3])  # (label, rel_mse, color)
    opt_data = unique

    labels = [x[0] for x in opt_data]
    vals   = [x[1] for x in opt_data]
    colors = [x[2] for x in opt_data]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))
    x = np.arange(n)
    bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.7,
                  width=0.55, zorder=3)

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
    ax.set_ylabel("Test relative MSE", fontsize=FONT_LABEL)
    ax.set_title(
        "FAE reconstruction quality — full model comparison",
        fontsize=FONT_TITLE,
    )
    ax.set_ylim(0, max(vals) * 1.20)
    ax.grid(axis="y", alpha=0.3, color=C_GRID)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig2_performance_bars")


# ============================================================================
# Figure 3: Performance vs observation time
# ============================================================================

def fig3_observation_time(out_dir: Path) -> None:
    """Relative MSE as a function of observation time fraction t/T.

    ``training_time_results`` in eval_results.json stores held-in MSE at
    five observation fractions: 0.14, 0.43, 0.57, 0.86, 1.0.

    Each curve shows how well the model reconstructs the field as more
    temporal observations become available.  All methods should improve
    monotonically; steeper curves indicate stronger use of early observations.
    """
    # Select key runs for this figure: all core non-denoiser combinations
    sel_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                if grp in COMPARISON_GROUPS]
    # Deduplicate by run dir (Muon+L2 appears in opt and arch)
    seen_dirs: set = set()
    dedup: list = []
    for x in sel_runs:
        if x[1] not in seen_dirs:
            seen_dirs.add(x[1])
            dedup.append(x)
    sel_runs = dedup

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    for label, d, ls, color in sel_runs:
        ev = _load_eval(d)
        if ev is None or not ev.get("training_time_results"):
            continue
        tt = ev["training_time_results"]
        times = sorted(float(k) for k in tt)
        rels  = [float(tt[str(t)]["rel_mse"]) if str(t) in tt
                 else float(tt["%.16g" % t]["rel_mse"])
                 for t in times]
        # fallback for float-key lookup
        if not rels:
            times_keys = sorted(tt.keys(), key=float)
            times = [float(k) for k in times_keys]
            rels  = [float(tt[k]["rel_mse"]) for k in times_keys]
        ax.plot(times, rels, linestyle=ls, color=color,
                marker="o", markersize=3.5, linewidth=1.4,
                label=label, zorder=3)

    ax.set_xlabel("Observation time fraction $t/T$", fontsize=FONT_LABEL)
    ax.set_ylabel("Relative MSE at time $t$", fontsize=FONT_LABEL)
    ax.set_yscale("log")
    ax.set_title(
        r"Reconstruction quality vs.\ observation time",
        fontsize=FONT_TITLE,
    )
    ax.grid(which="both", alpha=0.25, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig3_observation_time")


# ============================================================================
# Figure 4: Held-out generalisation
# ============================================================================

def fig4_held_out(out_dir: Path) -> None:
    """Relative MSE at held-out vs. training time fractions.

    ``held_out_results`` in eval_results.json reports performance at two
    time fractions not seen during training: t/T ≈ 0.286 and t/T ≈ 0.714.
    These measure interpolation ability.

    Pairs of bars (held-out t ≈ 0.286, t ≈ 0.714) for each method, grouped
    by held-out time.  Lower values = better interpolation.
    """
    # Gather results
    records: List[Tuple[str, float, float, str]] = []  # (label, r1, r2, color)
    for label, d, grp, ls, color in RUNS:
        if grp not in COMPARISON_GROUPS:
            continue
        ev = _load_eval(d)
        if ev is None or not ev.get("held_out_results"):
            continue
        ho = ev["held_out_results"]
        # Robustly extract the two held-out keys
        keys = sorted(ho.keys(), key=float)
        r1 = float(ho[keys[0]]["rel_mse"]) if len(keys) > 0 else float("nan")
        r2 = float(ho[keys[1]]["rel_mse"]) if len(keys) > 1 else float("nan")
        records.append((label, r1, r2, color))

    # Deduplicate
    seen: set = set()
    unique: list = []
    for rec in records:
        if rec[0] not in seen:
            seen.add(rec[0])
            unique.append(rec)
    records = unique

    n = len(records)
    labels_plot = [r[0] for r in records]
    r1s    = [r[1] for r in records]
    r2s    = [r[2] for r in records]
    colors = [r[3] for r in records]

    x = np.arange(n)
    w = 0.30
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    bars1 = ax.bar(x - w / 2, r1s, width=w, color=colors, edgecolor="k",
                   linewidth=0.7, alpha=0.70, label=r"$t/T \approx 0.29$",
                   zorder=3)
    bars2 = ax.bar(x + w / 2, r2s, width=w, color=colors, edgecolor="k",
                   linewidth=0.7, alpha=1.00, hatch="///",
                   label=r"$t/T \approx 0.71$", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=FONT_LABEL, rotation=25, ha="right")
    ax.set_ylabel("Relative MSE (held-out)", fontsize=FONT_LABEL)
    ax.set_title("Held-out interpolation accuracy — full model comparison",
                 fontsize=FONT_TITLE)
    ax.set_ylim(0, max(max(r1s), max(r2s)) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", alpha=0.3, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig4_held_out")


# ============================================================================
# Figure 5: Latent-space quality — collapse diagnostics
# ============================================================================

def fig5_latent_diagnostics(out_dir: Path) -> None:
    """Latent-space collapse diagnostics for runs with complete eval.

    Three horizontal-bar subplots following the report.py visual language:
      (a) ``decode_sensitivity``  — ratio of zero-latent MSE to full MSE;
          value >> 1 means the decoder genuinely uses the latent code.
      (b) ``zero_latent_mse / test_mse``  — how much worse a zero-code
          decode is; large ratio = healthy utilisation.
      (c) ``latent_var_mean``  — mean variance of latent codes across the
          test batch; near-zero indicates posterior collapse.

    Uses per-run colours from the RUNS registry for consistency with all
    other publication figures.  Value annotations are placed inline.
    """
    entries: list = []
    for label, d, grp, ls, color in RUNS:
        if grp not in COMPARISON_GROUPS:
            continue
        ev = _load_eval(d)
        if ev is None or not ev.get("collapse_diagnostics"):
            continue
        cd = ev["collapse_diagnostics"]
        test_mse = float(ev.get("test_mse", 1e-12))
        entries.append((label, cd, color, test_mse))

    # Deduplicate by label
    seen: set = set()
    unique: list = []
    for e in entries:
        if e[0] not in seen:
            seen.add(e[0])
            unique.append(e)
    entries = unique

    if not entries:
        return

    labels  = [e[0] for e in entries]
    colors  = [e[2] for e in entries]
    sens    = [float(e[1]["decode_sensitivity"]) for e in entries]
    zlr     = [float(e[1]["zero_latent_mse"]) / max(1e-12, e[3])
               for e in entries]
    latvar  = [float(e[1]["latent_var_mean"]) for e in entries]

    fig, axes = plt.subplots(1, 3,
                             figsize=(FIG_WIDTH, SUBPLOT_HEIGHT + 0.3),
                             sharey=True)

    def _hbar(ax: plt.Axes, vals: List[float], title: str,
              xlabel: str, fmt: str = "%.2f",
              ref_line: Optional[float] = None) -> None:
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, edgecolor="k", linewidth=0.6,
                       height=0.55, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        if ref_line is not None:
            ax.axvline(ref_line, color=C_GEN, ls="--", lw=0.9, alpha=0.6,
                       zorder=4)
        ax.grid(axis="x", alpha=0.25, color=C_GRID)
        _set_tick_fontsize(ax)
        # Inline value annotations
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                    fmt % v, va="center", ha="left", fontsize=FONT_TICK - 0.5)

    _hbar(axes[0], sens,   "(a) Decode sensitivity",
          r"$\mathrm{MSE}_{z=0}\,/\,\mathrm{MSE}_z$", "%.2f", ref_line=1.0)
    _hbar(axes[1], zlr,    "(b) Zero-latent ratio",
          r"$\mathrm{MSE}_{z=0}\,/\,\mathrm{MSE}_{\mathrm{test}}$", "%.1f")
    _hbar(axes[2], latvar, "(c) Latent variance",
          r"mean $\mathrm{Var}[z]$", "%.4f")

    axes[0].invert_yaxis()

    fig.suptitle("Latent-space utilisation diagnostics", fontsize=FONT_TITLE + 1,
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, out_dir, "fig5_latent_diagnostics")


# ============================================================================
# Figure 6: Reconstruction quality vs σ-scale (multi-panel bar)
# ============================================================================

def fig6_scale_comparison(out_dir: Path) -> None:
    """Per-time-fraction comparison for single-scale vs multiscale decoder.

    Two grouped bars at each observation time fraction — σ = 1 vs
    σ ∈ {1,2,4,8} — both using Muon + L2.  Shows that multiscale RFF
    improves fine-grained reconstruction especially at late times.
    """
    runs_scale = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                  if grp == "scale"]
    # Collect per-time results
    time_labels: Optional[List[str]] = None
    bars_data: list = []   # (label, [rel_mse per time], color)

    for label, d, ls, color in runs_scale:
        ev = _load_eval(d)
        if ev is None or not ev.get("training_time_results"):
            continue
        tt = ev["training_time_results"]
        keys = sorted(tt.keys(), key=float)
        if time_labels is None:
            time_labels = ["%.2f" % float(k) for k in keys]
        vals = [float(tt[k]["rel_mse"]) for k in keys]
        bars_data.append((label, vals, color))

    if not bars_data or time_labels is None:
        return

    n_time = len(time_labels)
    n_bars = len(bars_data)
    x = np.arange(n_time)
    total_width = 0.7
    w = total_width / n_bars
    offsets = np.linspace(-(total_width - w) / 2, (total_width - w) / 2, n_bars)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    for i, (label, vals, color) in enumerate(bars_data):
        ax.bar(x + offsets[i], vals, width=w, color=color,
               edgecolor="k", linewidth=0.6, label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["$t/T = %s$" % tl for tl in time_labels],
        fontsize=FONT_LABEL,
    )
    ax.set_ylabel("Relative MSE", fontsize=FONT_LABEL)
    ax.set_title(
        r"Single-scale ($\sigma\!=\!1$) vs multiscale ($\sigma\!\in\!\{1,2,4,8\}$) — Muon + $\ell_2$",
        fontsize=FONT_TITLE,
    )
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig6_scale_comparison_per_time")


# ============================================================================
# Figure 7: Architecture comparison — Det-FiLM vs FiLM+Latent Prior
# ============================================================================

def fig7_architecture_comparison(out_dir: Path) -> None:
    """Per-time-fraction bar comparison for Muon+L2 vs Adam+L2+Prior.

    Same format as Figure 6, but contrasts the two decoder architectures.

    Note: FiLM+Prior uses a DETERMINISTIC decoder trained with a denoiser
    ELBO objective that includes a diffusion prior on the latent space.
    This is distinct from a denoising decoder (Figure 7b).  The test MSE
    for FiLM+Prior reflects reconstruction quality after sampling latents
    through the prior, not straight L2 training.
    """
    runs_arch = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                 if grp == "arch"]
    # Deduplicate
    seen: set = set()
    dedup: list = []
    for x in runs_arch:
        if x[0] not in seen:
            seen.add(x[0])
            dedup.append(x)
    runs_arch = dedup

    time_labels: Optional[List[str]] = None
    bars_data: list = []

    for label, d, ls, color in runs_arch:
        ev = _load_eval(d)
        if ev is None or not ev.get("training_time_results"):
            continue
        tt = ev["training_time_results"]
        keys = sorted(tt.keys(), key=float)
        if time_labels is None:
            time_labels = ["%.2f" % float(k) for k in keys]
        vals = [float(tt[k]["rel_mse"]) for k in keys]
        bars_data.append((label, vals, color))

    if not bars_data or time_labels is None:
        return

    n_time = len(time_labels)
    n_bars = len(bars_data)
    x = np.arange(n_time)
    total_width = 0.65
    w = total_width / n_bars
    offsets = np.linspace(-(total_width - w) / 2, (total_width - w) / 2, n_bars)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    for i, (label, vals, color) in enumerate(bars_data):
        ax.bar(x + offsets[i], vals, width=w, color=color,
               edgecolor="k", linewidth=0.6, label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["$t/T = %s$" % tl for tl in time_labels],
        fontsize=FONT_LABEL,
    )
    ax.set_ylabel("Relative MSE (eval with $\\ell_2$)", fontsize=FONT_LABEL)
    ax.set_title(
        r"Architecture comparison — Muon + $\ell_2$ vs Adam + $\ell_2$ + Prior",
        fontsize=FONT_TITLE,
    )
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax)

    fig.text(
        0.5, -0.06,
        "Note: Adam+$\\ell_2$+Prior uses deterministic decoder + latent diffusion prior "
        "(ELBO objective); Muon+$\\ell_2$ uses plain L2 loss.",
        ha="center", fontsize=FONT_TICK - 0.5, style="italic",
    )

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig7_architecture_comparison_per_time")


# ============================================================================
# Figure 8: Summary table — publication-ready numbers
# ============================================================================

def fig8_summary_table(out_dir: Path) -> None:
    """LaTeX-formatted summary statistics printed to file and rendered as figure.

    Produces two outputs:
      • ``table_summary.tex``  — LaTeX table body (booktabs-style) ready for
        copy-paste into a SIAM paper.
      • ``fig8_summary_table.{png,pdf}``  — Matplotlib render of the same
        table for slide decks / arxiv submissions.
    """
    # Collect all runs with eval results
    rows: list = []
    processed: set = set()
    for label, d, grp, ls, color in RUNS:
        if d in processed:
            continue
        ev = _load_eval(d)
        if ev is None:
            continue
        processed.add(d)
        cd = ev.get("collapse_diagnostics") or {}
        ho = ev.get("held_out_results") or {}
        ho_keys = sorted(ho.keys(), key=float)
        ho_r1 = ho[ho_keys[0]]["rel_mse"] if len(ho_keys) > 0 else float("nan")
        ho_r2 = ho[ho_keys[1]]["rel_mse"] if len(ho_keys) > 1 else float("nan")
        rows.append({
            "label":    label,
            "group":    grp,
            "test_mse": ev["test_mse"],
            "test_rel": ev["test_rel_mse"],
            "ho_r1":    ho_r1,
            "ho_r2":    ho_r2,
            "sens":     cd.get("decode_sensitivity", float("nan")),
            "latvar":   cd.get("latent_var_mean", float("nan")),
        })

    # ── LaTeX table ──────────────────────────────────────────────────────────
    tex_path = out_dir / "table_summary.tex"
    with open(tex_path, "w") as fp:
        fp.write("% Auto-generated by fae_publication_figures.py\n")
        fp.write("\\begin{table}[ht]\n")
        fp.write("  \\centering\n")
        fp.write("  \\caption{FAE reconstruction performance summary. "
                 "$^*$\\,Muon+$\\ell_2$ with $\\sigma\\in\\{1,2,4,8\\}$; "
                 "\\textit{(d)} denoiser-ELBO objective.}\n")
        fp.write("  \\label{tab:fae_results}\n")
        fp.write("  \\small\n")
        fp.write("  \\begin{tabular}{lcccccc}\n")
        fp.write("    \\toprule\n")
        fp.write(
            "    Method & Test MSE & Test Rel.~MSE"
            " & HO ($t\\approx0.29$) & HO ($t\\approx0.71$)"
            " & Decode sens. & Lat. var. \\\\\n"
        )
        fp.write("    \\midrule\n")
        group_sep = {"opt": False, "scale": False, "arch": False}
        prev_grp = None
        for r in rows:
            g = r["group"]
            if prev_grp is not None and g != prev_grp:
                fp.write("    \\midrule\n")
            prev_grp = g
            la = r["label"].replace("\\textit", "").replace("\\", "\\") \
                            .replace("$", "$").replace("{", "{").replace("}", "}")
            sens_s = "%.3f" % r["sens"] if not np.isnan(r["sens"]) else "---"
            latv_s = "%.4f" % r["latvar"] if not np.isnan(r["latvar"]) else "---"
            fp.write(
                "    %s & %.5f & %.5f & %.5f & %.5f & %s & %s \\\\\n" % (
                    r["label"],
                    r["test_mse"], r["test_rel"],
                    r["ho_r1"],    r["ho_r2"],
                    sens_s, latv_s,
                )
            )
        fp.write("    \\bottomrule\n")
        fp.write("  \\end{tabular}\n")
        fp.write("\\end{table}\n")
    print("  LaTeX table written to:", tex_path)

    # ── Matplotlib render ─────────────────────────────────────────────────
    col_hdr = [
        "Method",
        "Test MSE",
        "Rel. MSE",
        r"HO $t{\approx}0.29$",
        r"HO $t{\approx}0.71$",
        "Dec. sens.",
        "Lat. var.",
    ]
    cell_data = []
    row_colors = []
    G_COLORS = {"opt_l2": "#f7f9ff", "opt_ntk": "#f0f0ff",
                 "scale": "#f9fff7", "arch": "#fff9f7", "den": "#fff7f9"}
    for r in rows:
        sens_s = "%.3f" % r["sens"] if not np.isnan(r["sens"]) else "—"
        latv_s = "%.4f" % r["latvar"] if not np.isnan(r["latvar"]) else "—"
        cell_data.append([
            r["label"],
            "%.5f" % r["test_mse"],
            "%.5f" % r["test_rel"],
            "%.5f" % r["ho_r1"],
            "%.5f" % r["ho_r2"],
            sens_s,
            latv_s,
        ])
        row_colors.append([G_COLORS.get(r["group"], "white")] * len(col_hdr))

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, 0.28 * (len(rows) + 2)),
    )
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_hdr,
        cellColours=row_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_TICK - 0.5)
    tbl.auto_set_column_width(list(range(len(col_hdr))))
    # Bold header row
    for j in range(len(col_hdr)):
        tbl[0, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig8_summary_table")


# ============================================================================
# Figure 9: Training-loss comparison panel (two-panel: σ=1 vs σ={1,2,4,8})
# ============================================================================

def fig9_twoscale_training(out_dir: Path, max_steps: int = 50_000) -> None:
    """Two-panel training loss: single-scale (σ=1) vs multiscale (σ={1,2,4,8}).

    Left panel: σ = 1, comparing Adam+L2, Muon+NTK, Det-FiLM (Muon+L2).
    Right panel: σ = {1,2,4,8}, comparing all four optimizer variants.

    Allows direct visual comparison of how multiscale affects training dynamics.
    """
    single_scale_runs = [
        (r"Adam + $\ell_2$",   "results/fae_film_adam_l2/run_10qdkz0u",              "-",  "#d62728"),
        ("Muon + NTK",          "results/fae_film_muon_ntk/run_zjvaw52j",             "-.", "#9467bd"),
        (r"Muon + $\ell_2$",   "results/fae_deterministic_film/run_90ndogk3",         "-",  "#2ca02c"),
    ]
    multi_scale_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                        if grp in {"opt_l2", "opt_ntk"}]

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(FIG_WIDTH, SUBPLOT_HEIGHT),
                                    sharey=False)

    def _plot_curves(ax: plt.Axes, run_list: list, title: str) -> None:
        for label, d, ls, color in run_list:
            loss = _load_training_loss(d)
            if loss is None:
                continue
            n = len(loss)
            steps = np.linspace(0, max_steps, n)
            smoothed = _ema_smooth(loss)
            ax.plot(steps / 1e3, smoothed, linestyle=ls, color=color,
                    linewidth=1.5, label=label, zorder=3)
            ax.fill_between(steps / 1e3, loss * 0.96, loss * 1.04,
                            color=color, alpha=0.07, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("Step ($\\times 10^3$)", fontsize=FONT_LABEL)
        ax.set_ylabel("Training loss", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.25, color=C_GRID)
        ax.legend(fontsize=FONT_LEGEND - 0.5, framealpha=0.9, loc="upper right")
        _set_tick_fontsize(ax)

    _plot_curves(ax1, single_scale_runs,
                 r"(a) Single-scale $\sigma = 1$")
    _plot_curves(ax2, multi_scale_runs,
                 r"(b) Multiscale $\sigma \in \{1,2,4,8\}$")

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig9_twoscale_training")


# ============================================================================
# Figure 10: Reconstruction fields — per-sample, t1–t7 (excluding t0)
# ============================================================================

def fig10_reconstruction_fields(out_dir: Path) -> None:
    """Professional reconstruction figures split by sample and split type.

    Produces separate figures for each sample (3 samples) and for each split:
      - ``fig10_sample{idx}_train_fields``
      - ``fig10_sample{idx}_heldout_fields``

    Each figure has one column per time point in that split and two rows:
    original (top) and reconstruction (bottom).  Time index ``t0`` is excluded.
    """
    # Proposed method used for qualitative reconstruction panels.
    run_dir = "results/adam_ntk_prior/run_zaql9zhd"

    try:
        encode_fn, decode_fn, meta = _build_model_io_from_run(run_dir, decode_mode="auto")
        ds = _get_dataset_for_run(run_dir)
    except Exception as exc:
        print(f"  [fig10] failed to load checkpoint/dataset: {exc}")
        return

    data = ds["npz"]
    grid_coords = np.asarray(data["grid_coords"], dtype=np.float32)
    resolution = int(data["resolution"])
    marginal_keys = _list_marginal_keys(data)

    held_out_indices = [int(i) for i in np.asarray(data.get("held_out_indices", []), dtype=np.int32)]
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        held_out_indices = sorted(set(held_out_indices) | {0})
    ho_set = set(held_out_indices)

    # sample IDs requested by revision note
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
            continue  # explicit exclusion requested
        t_val = float(str(key).replace("raw_marginal_", ""))
        split = "held_out" if tidx in ho_set else "train"
        fields = np.asarray(data[key], dtype=np.float32)

        for sid in sample_ids:
            if sid >= fields.shape[0]:
                continue
            orig = fields[sid]
            recon = _reconstruct_single(orig)
            sample_records[sid][split].append((tidx, t_val, orig, recon))

    def _plot_sample_split(sample_id: int, split: str, entries: List[Tuple[int, float, NDArray, NDArray]]) -> None:
        if not entries:
            return
        entries = sorted(entries, key=lambda x: x[0])
        n_cols = len(entries)
        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=(FIG_WIDTH, FIELD_ROW_HEIGHT * 1.35),
            squeeze=False,
        )

        for col, (tidx, t_val, orig, recon) in enumerate(entries):
            o2 = np.asarray(orig, dtype=np.float32).reshape(resolution, resolution)
            r2 = np.asarray(recon, dtype=np.float32).reshape(resolution, resolution)
            vmin = float(min(o2.min(), r2.min()))
            vmax = float(max(o2.max(), r2.max()))

            axes[0, col].imshow(o2, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"$t_{{{tidx}}}$", fontsize=FONT_TITLE)
            axes[0, col].axis("off")

            axes[1, col].imshow(r2, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[1, col].axis("off")

        # Compact publication styling: row labels only, no verbose suptitle.
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
        _save_fig(fig, out_dir, f"fig10_sample{sample_id}_{split}_fields")

    for sid in sample_ids:
        _plot_sample_split(sid, "train", sample_records[sid]["train"])
        _plot_sample_split(sid, "held_out", sample_records[sid]["held_out"])


# ============================================================================
# Figure 11: Denoiser comparison — separate from deterministic methods
# ============================================================================

def fig11_denoiser_comparison(out_dir: Path) -> None:
    """Bar chart comparing denoiser runs at single vs multiscale.

    Includes a caveat that denoiser test MSE is evaluated using iterative
    denoising (default 32 steps) and may be inflated relative to deterministic
    methods.  Also shows Adam+$\ell_2$+Prior (latent diffusion) for context, since
    it uses the denoiser ELBO objective with a deterministic decoder.

    Denoiser runs use ``decoder_type=denoiser_standard``; Adam+$\ell_2$+Prior uses
    ``decoder_type=film`` with ``loss_type=denoiser``.
    """
    den_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                if grp in {"den", "arch"}]
    # Deduplicate by label
    seen: set = set()
    dedup: list = []
    for x in den_runs:
        if x[0] not in seen:
            seen.add(x[0])
            dedup.append(x)
    den_runs = dedup

    data: list = []  # (label, rel_mse, color)
    for label, d, ls, color in den_runs:
        ev = _load_eval(d)
        if ev is None:
            continue
        data.append((label, float(ev["test_rel_mse"]), color))

    if not data:
        print("  [fig11] no denoiser eval results, skipping.")
        return

    labels = [x[0] for x in data]
    vals   = [x[1] for x in data]
    colors = [x[2] for x in data]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))
    x = np.arange(n)
    bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.7,
                  width=0.50, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + max(vals) * 0.02,
            "%.4f" % v,
            ha="center", va="bottom",
            fontsize=FONT_TICK - 0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL - 0.5, rotation=20, ha="right")
    ax.set_ylabel("Test relative MSE", fontsize=FONT_LABEL)
    ax.set_title(
        "Denoiser and latent-prior architectures — test reconstruction quality",
        fontsize=FONT_TITLE,
    )
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3, color=C_GRID)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    _set_tick_fontsize(ax)

    fig.text(
        0.5, -0.06,
        "Caveat: Denoiser uses iterative denoising at eval (32 steps by default). "
        "Increasing --denoiser-eval-sample-steps may improve results.",
        ha="center", fontsize=FONT_TICK - 0.5, style="italic",
    )

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig11_denoiser_comparison")


# ============================================================================
# Figure 12: PSD spectral analysis (placeholder — needs raw reconstruction data)
# ============================================================================

def fig12_psd_spectral(out_dir: Path) -> None:
    """Time-resolved PSD analysis (no averaging across times).

    Generates:
      1) ``fig12_psd_spectral_per_time``: per-time PSD overlays.
      2) ``fig12b_psd_mismatch_per_time``: per-time log-PSD mismatch trends.

    This follows the Tran-evaluation philosophy where spectral diagnostics are
    analyzed per scale/time, not collapsed into a single global average.
    """
    psd_path = out_dir / "psd_data.npz"
    if not psd_path.exists():
        print("  [fig12] psd_data.npz not found — run scripts/fae/compute_psd.py first. Skipping.")
        return

    data = np.load(psd_path, allow_pickle=True)

    freqs = data["freqs"]
    labels_arr = data["labels"]
    time_keys = list(data["time_keys"]) if "time_keys" in data.files else []
    if not time_keys:
        print("  [fig12] time_keys not found in psd_data.npz; rerun compute_psd.py. Skipping.")
        return

    labels_text = [str(lbl) for lbl in labels_arr]
    colors_psd = _resolve_run_colors(labels_text)

    # ------------------------------------------------------------------
    # Figure 12a: per-time PSD overlays
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

        ax.set_title(tk, fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.22, color=C_GRID)
        ax.set_xlabel("k", fontsize=FONT_LABEL)
        ax.set_ylabel("PSD", fontsize=FONT_LABEL)
        _set_tick_fontsize(ax)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    handles = [Line2D([0], [0], color="k", lw=1.8, label="GT")]
    for i, lbl_str in enumerate(labels_text):
        handles.append(Line2D([0], [0], color=colors_psd[i], lw=1.3, label=lbl_str))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=FONT_LEGEND, framealpha=0.9)
    fig.suptitle("Time-resolved PSD comparison", fontsize=FONT_TITLE + 1, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, out_dir, "fig12_psd_spectral_per_time")
    # Backward-compatible name used elsewhere in the pipeline.
    fig_alias, ax_alias = plt.subplots(1, 1, figsize=(1, 1))
    plt.close(fig_alias)
    # Re-render quickly from saved object by plotting same data to legacy name.
    # Keep both names in output to avoid stale-file ambiguity.
    fig_legacy, axes_legacy = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes_legacy = np.atleast_2d(axes_legacy)
    for idx, tk in enumerate(time_keys):
        ax = axes_legacy[idx // n_cols, idx % n_cols]
        gt_key = f"psd_gt_{tk}"
        if gt_key in data:
            ax.loglog(freqs, data[gt_key], "k-", linewidth=1.8, label="GT", alpha=0.8, zorder=4)
        for i, lbl_str in enumerate(labels_text):
            key = f"psd_{_safe_key(lbl_str)}_{tk}"
            if key not in data:
                continue
            color = colors_psd[i]
            ax.loglog(freqs, data[key], color=color, linewidth=1.1, alpha=0.9, label=lbl_str, zorder=3)
        ax.set_title(tk, fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.22, color=C_GRID)
        ax.set_xlabel("k", fontsize=FONT_LABEL)
        ax.set_ylabel("PSD", fontsize=FONT_LABEL)
        _set_tick_fontsize(ax)
    for idx in range(n, n_rows * n_cols):
        axes_legacy[idx // n_cols, idx % n_cols].set_visible(False)
    fig_legacy.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=FONT_LEGEND, framealpha=0.9)
    fig_legacy.suptitle("Time-resolved PSD comparison", fontsize=FONT_TITLE + 1, y=1.02)
    fig_legacy.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig_legacy, out_dir, "fig12_psd_spectral")

    # ------------------------------------------------------------------
    # Figure 12b: per-time mismatch curves
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
    ax2.set_xticklabels(ordered, rotation=30, ha="right", fontsize=FONT_TICK)
    ax2.set_ylabel("log-PSD mismatch", fontsize=FONT_LABEL)
    ax2.set_xlabel("time/split", fontsize=FONT_LABEL)
    ax2.set_title("Per-time spectral mismatch (lower is better)", fontsize=FONT_TITLE)
    ax2.grid(alpha=0.25, color=C_GRID)
    ax2.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax2)
    fig2.tight_layout()
    _save_fig(fig2, out_dir, "fig12b_psd_mismatch_per_time")


# ============================================================================
# Figure 13: Latent-regularization evidence
# ============================================================================

def fig13_latent_regularization(out_dir: Path) -> None:
    """Plot latent-regularization evidence from psd_latent_metrics.json.

    Uses checkpoint-based diagnostics to compare:
      - latent variance magnitude,
      - effective rank,
      - variance spread (q90/q10),
      - time-wise reconstruction imbalance (MSE CV).

    Lower spread/CV and balanced effective rank support stronger latent
    regularization and less over-optimization of specific fields.

    Colour scheme: uses per-run colours from the RUNS registry (resolved
    via label matching) for visual consistency with all other figures.
    Falls back to the report.py ``C_OBS`` / ``C_GEN`` alternation.
    """
    metrics_path = out_dir / "psd_latent_metrics.json"
    if not metrics_path.exists():
        print("  [fig13] psd_latent_metrics.json not found — run compute_psd.py first. Skipping.")
        return

    with metrics_path.open() as f:
        metrics = json.load(f)

    if not metrics:
        print("  [fig13] empty metrics json, skipping.")
        return

    labels = list(metrics.keys())
    x = np.arange(len(labels), dtype=np.float64)

    # Resolve colours from RUNS registry for visual consistency.
    colors = _resolve_run_colors(labels)

    lat_var = [float(metrics[k].get("latent", {}).get("latent_var_mean", np.nan)) for k in labels]
    eff_rank = [float(metrics[k].get("latent", {}).get("effective_rank", np.nan)) for k in labels]
    spread = [float(metrics[k].get("latent", {}).get("latent_var_spread_q90_q10", np.nan)) for k in labels]
    time_cv = [float(metrics[k].get("time_recon_balance", {}).get("mse_cv", np.nan)) for k in labels]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * 2.0))
    axes = np.asarray(axes)

    entries = [
        (axes[0, 0], lat_var,  "(a) Latent variance mean",
         r"mean $\mathrm{Var}[z]$", "%.4f", False),
        (axes[0, 1], eff_rank, "(b) Effective rank",
         r"$\exp(H(\mathbf{p}))$", "%.1f", False),
        (axes[1, 0], spread,   "(c) Variance spread",
         r"$q_{90}/q_{10}$", "%.1f", True),
        (axes[1, 1], time_cv,  "(d) Reconstruction imbalance",
         r"CV of MSE across times", "%.3f", True),
    ]

    for ax, vals, title, ylabel, fmt, lower_better in entries:
        bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.6, zorder=3,
                      width=0.6)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=FONT_TICK)
        ax.grid(axis="y", alpha=0.25, color=C_GRID)
        _set_tick_fontsize(ax)

        # Value annotations on top of each bar
        arr = np.asarray(vals, dtype=np.float64)
        y_max = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1.0
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        v + y_max * 0.02,
                        fmt % v,
                        ha="center", va="bottom",
                        fontsize=FONT_TICK - 0.5)

        # Highlight best method with a thicker dark edge
        finite = np.isfinite(arr)
        if np.any(finite):
            idx_best = int(np.nanargmin(arr) if lower_better else np.nanargmax(arr))
            bars[idx_best].set_linewidth(2.0)
            bars[idx_best].set_edgecolor("k")

        # Direction arrow annotation
        direction_text = r"$\leftarrow$ lower = better" if lower_better else r"$\rightarrow$ higher = better"
        ax.annotate(direction_text, xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=FONT_LEGEND,
                    fontstyle="italic", color="#555555")

    fig.suptitle("Latent-regularization evidence (checkpoint diagnostics)",
                 fontsize=FONT_TITLE + 1, y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig13_latent_regularization")


# ============================================================================
# Figure 14: Per-marginal latent diagnostics (effective rank + isotropy)
# ============================================================================

def fig14_per_marginal_latent(out_dir: Path) -> None:
    """Per-marginal latent diagnostics across time indices.

    Two-panel line plot:
      (a) Effective rank per marginal — shows whether the diffusion prior
          maintains high-dimensional utilisation uniformly across scales,
          or whether certain time indices suffer rank collapse.
      (b) Isotropy ratio per marginal — smallest/largest eigenvalue ratio.
          Values near 1 indicate near-isotropic latents; near 0 indicates
          dominant directions.

    Each method is a separate curve.  Time indices on the x-axis are
    labelled with their train/held-out status.  Consistent per-run
    colours from the RUNS registry.
    """
    metrics_path = out_dir / "psd_latent_metrics.json"
    if not metrics_path.exists():
        print("  [fig14] psd_latent_metrics.json not found — run compute_psd.py first. Skipping.")
        return

    with metrics_path.open() as f:
        metrics = json.load(f)

    if not metrics:
        print("  [fig14] empty metrics json, skipping.")
        return

    # Collect all time keys across runs
    all_time_keys: set = set()
    for label in metrics:
        pml = metrics[label].get("per_marginal_latent", {})
        all_time_keys.update(pml.keys())

    if not all_time_keys:
        print("  [fig14] no per_marginal_latent data found — rerun compute_psd.py. Skipping.")
        return

    ordered_tkeys = sorted(all_time_keys, key=lambda s: (int(s.split("_")[0][1:]), s))

    # Resolve per-run colours
    labels = list(metrics.keys())
    colors = _resolve_run_colors(labels)

    fig, (ax_rank, ax_iso) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT + 0.2), sharey=False,
    )

    x = np.arange(len(ordered_tkeys), dtype=np.float64)

    for i, label in enumerate(labels):
        pml = metrics[label].get("per_marginal_latent", {})
        eff_ranks = [float(pml.get(tk, {}).get("effective_rank", np.nan))
                     for tk in ordered_tkeys]
        isotropy = [float(pml.get(tk, {}).get("isotropy_ratio", np.nan))
                    for tk in ordered_tkeys]

        ax_rank.plot(x, eff_ranks, marker="o", markersize=4, linewidth=1.4,
                     color=colors[i], label=label, zorder=3)
        ax_iso.plot(x, isotropy, marker="s", markersize=4, linewidth=1.4,
                    color=colors[i], label=label, zorder=3)

    # Format x-axis: show time index and train/held-out
    tick_labels = []
    for tk in ordered_tkeys:
        tidx = tk.split("_")[0]
        split = "HO" if "held_out" in tk else "Tr"
        tick_labels.append(f"{tidx}\n({split})")

    for ax, title, ylabel in [
        (ax_rank, "(a) Effective rank per marginal",
         r"$\exp(H(\mathbf{p}))$"),
        (ax_iso, "(b) Isotropy ratio per marginal",
         r"$\lambda_{\min}/\lambda_{\max}$"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=FONT_TICK - 0.5)
        ax.set_xlabel("Time index (marginal)", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.grid(which="both", alpha=0.25, color=C_GRID)
        _set_tick_fontsize(ax)

    # Legend outside on the right to avoid crowding with x tick labels.
    handles, leg_labels = ax_rank.get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="center left",
        ncol=1,
        fontsize=FONT_LEGEND,
        framealpha=0.9,
        bbox_to_anchor=(0.99, 0.5),
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    _save_fig(fig, out_dir, "fig14_per_marginal_latent")


# ============================================================================
# Figure 15: Inter-marginal distance regularity
# ============================================================================

def fig15_inter_marginal_distance(out_dir: Path) -> None:
    """Inter-marginal Bures-Wasserstein distance regularity.

    Two-panel figure:
      (a) W2 distance between consecutive time-index marginals in latent
          space.  For the MSBM, evenly spaced marginals are easier to
          transport between — the score network sees a consistent step size.
          Large jumps indicate scale-specific latent clustering.
      (b) Decomposition: mean-shift (L2 distance between centroids) vs
          covariance (Bures) component.  Reveals whether distance variation
          comes from centroid drift or shape change.

    A well-regularised latent space (e.g. diffusion prior) should show
    more uniform W2 steps and lower Bures covariance contribution.
    """
    metrics_path = out_dir / "psd_latent_metrics.json"
    if not metrics_path.exists():
        print("  [fig15] psd_latent_metrics.json not found — run compute_psd.py first. Skipping.")
        return

    with metrics_path.open() as f:
        metrics = json.load(f)

    if not metrics:
        print("  [fig15] empty metrics json, skipping.")
        return

    # Collect all inter-marginal pair keys across runs
    all_pair_keys: set = set()
    for label in metrics:
        imd = metrics[label].get("inter_marginal_w2", {})
        all_pair_keys.update(imd.keys())

    if not all_pair_keys:
        print("  [fig15] no inter_marginal_w2 data found — rerun compute_psd.py. Skipping.")
        return

    # Order by first time index in the pair
    def _pair_sort_key(pk: str) -> tuple:
        # e.g. "t1_train_to_t3_train"
        parts = pk.split("_to_")
        t0 = int(parts[0].split("_")[0][1:])
        t1 = int(parts[1].split("_")[0][1:]) if len(parts) > 1 else t0
        return (t0, t1)

    ordered_pairs = sorted(all_pair_keys, key=_pair_sort_key)

    labels = list(metrics.keys())
    colors = _resolve_run_colors(labels)

    fig, (ax_w2, ax_decomp) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT + 0.3), sharey=False,
    )

    x = np.arange(len(ordered_pairs), dtype=np.float64)
    bar_width = 0.7 / max(len(labels), 1)

    # Panel (a): W2 line plot per run
    for i, label in enumerate(labels):
        imd = metrics[label].get("inter_marginal_w2", {})
        w2_vals = [float(imd.get(pk, {}).get("w2", np.nan))
                   for pk in ordered_pairs]
        ax_w2.plot(x, w2_vals, marker="o", markersize=4, linewidth=1.4,
                   color=colors[i], label=label, zorder=3)

        # Annotate CV of W2 distances (regularity measure)
        w2_arr = np.asarray(w2_vals, dtype=np.float64)
        finite = w2_arr[np.isfinite(w2_arr)]
        if len(finite) > 1:
            cv = float(np.std(finite) / (np.mean(finite) + 1e-12))
            ax_w2.annotate(
                f"CV={cv:.2f}",
                xy=(x[-1], w2_vals[-1] if np.isfinite(w2_vals[-1]) else 0),
                xytext=(5, 5 + i * 12), textcoords="offset points",
                fontsize=FONT_TICK - 0.5, color=colors[i],
            )

    # Panel (b): Stacked bars — mean shift vs Bures covariance component
    n_runs = len(labels)
    offsets = np.linspace(-(0.7 - bar_width) / 2, (0.7 - bar_width) / 2, n_runs) if n_runs > 1 else [0.0]

    for i, label in enumerate(labels):
        imd = metrics[label].get("inter_marginal_w2", {})
        mean_l2 = [float(imd.get(pk, {}).get("mean_l2", np.nan))
                   for pk in ordered_pairs]
        bures = [float(np.sqrt(max(0, imd.get(pk, {}).get("bures_covariance_term", 0))))
                 for pk in ordered_pairs]

        ax_decomp.bar(x + offsets[i], mean_l2, width=bar_width,
                      color=colors[i], edgecolor="k", linewidth=0.5,
                      alpha=0.8, zorder=3, label=f"{label} (mean)" if i == 0 else None)
        ax_decomp.bar(x + offsets[i], bures, width=bar_width,
                      bottom=mean_l2, color=colors[i], edgecolor="k",
                      linewidth=0.5, alpha=0.4, hatch="///", zorder=3,
                      label=f"{label} (Bures)" if i == 0 else None)

    # Pair labels: "t1→t3", "t3→t4", etc.
    pair_labels = []
    for pk in ordered_pairs:
        parts = pk.split("_to_")
        t0 = parts[0].split("_")[0]
        t1 = parts[1].split("_")[0] if len(parts) > 1 else "?"
        pair_labels.append(f"{t0}$\\to${t1}")

    for ax, title, ylabel in [
        (ax_w2, "(a) Bures-Wasserstein $W_2$ between consecutive marginals",
         r"$W_2(z_{t_i}, z_{t_{i+1}})$"),
        (ax_decomp, "(b) Decomposition: centroid shift + covariance mismatch",
         "Distance component"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=FONT_TICK - 0.5, rotation=30, ha="right")
        ax.set_xlabel("Consecutive marginal pair", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.grid(axis="y", alpha=0.25, color=C_GRID)
        _set_tick_fontsize(ax)

    # Legend for panel (a): place outside panel to avoid overlap.
    ax_w2.legend(
        fontsize=FONT_LEGEND,
        framealpha=0.9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    # Custom legend for decomposition panel
    from matplotlib.patches import Patch as _Patch
    decomp_handles = [
        _Patch(facecolor="#888888", edgecolor="k", alpha=0.8, label="Centroid shift"),
        _Patch(facecolor="#888888", edgecolor="k", alpha=0.4, hatch="///", label="Bures (cov)"),
    ]
    ax_decomp.legend(handles=decomp_handles, fontsize=FONT_LEGEND,
                     framealpha=0.9, loc="best")

    fig.suptitle("Inter-marginal latent distance regularity",
                 fontsize=FONT_TITLE + 1, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    _save_fig(fig, out_dir, "fig15_inter_marginal_distance")


# ============================================================================
# Figure 16: NTK sample comparison — Adam+L2 vs Adam+NTK vs Adam+NTK+Prior
# ============================================================================

#: Runs used in the NTK sample comparison.  Order determines row order.
_NTK_COMPARE_RUNS: List[Tuple[str, str, str]] = [
    (r"Adam + $\ell_2$",     "results/fae_film_adam_l2_99pct/run_bnqm4evk",  "#d62728"),
    ("Adam + NTK",            "results/fae_film_adam_ntk_99pct/run_2hnr5shv", "#ff7f0e"),
    ("Adam + NTK + Prior",    "results/adam_ntk_prior/run_zaql9zhd",          "#bcbd22"),
]


def fig16_ntk_sample_comparison(out_dir: Path) -> None:
    """Side-by-side sample reconstructions comparing NTK training effects.

    Generates two figures per sample:

      ``fig16_sampleN_all_times``
        Rows: Ground truth, Adam+L2, Adam+NTK, Adam+NTK+Prior.
        Columns: t1–t7 (excluding t0).
        Shared colour bar per column for fair visual comparison.

      ``fig16_sampleN_t7_zoom``
        Zoomed single-column comparison at **t7** (the final, hardest
        marginal) with per-pixel error maps to highlight the effect of
        NTK loss scaling on high-frequency detail.

    All panels use the viridis colourmap, consistent fonts, and the
    shared FIG_WIDTH / FONT_* constants.
    """

    # Ensure publication rcParams are applied for this figure family.
    format_for_paper()

    # ── Load dataset (shared across all three runs) ─────────────────────────
    ref_run = _NTK_COMPARE_RUNS[0][1]
    try:
        ds = _get_dataset_for_run(ref_run)
    except Exception as exc:
        print(f"  [fig16] failed to load dataset: {exc}")
        return

    data = ds["npz"]
    grid_coords = np.asarray(data["grid_coords"], dtype=np.float32)
    resolution = int(data["resolution"])
    marginal_keys = _list_marginal_keys(data)

    # ── Build (encode, decode) for each run ─────────────────────────────────
    model_fns: List[Tuple[str, str, object, object]] = []  # (label, color, enc, dec)
    for label, run_dir, color in _NTK_COMPARE_RUNS:
        try:
            enc, dec, _ = _build_model_io_from_run(run_dir, decode_mode="auto")
            model_fns.append((label, color, enc, dec))
        except Exception as exc:
            print(f"  [fig16] skipping {label}: {exc}")

    if not model_fns:
        print("  [fig16] no models loaded, skipping.")
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

    sample_ids = [0, 1, 2]

    for sid in sample_ids:
        # ── Collect per-time reconstructions ────────────────────────────────
        # entries: list of (tidx, orig_2d, {label: recon_2d})
        entries: List[Tuple[int, NDArray, Dict[str, NDArray]]] = []
        for tidx, key in enumerate(marginal_keys):
            if tidx == 0:
                continue  # skip t0
            fields = np.asarray(data[key], dtype=np.float32)
            if sid >= fields.shape[0]:
                continue
            orig_flat = fields[sid]
            recons: Dict[str, NDArray] = {}
            for label, color, enc_fn, dec_fn in model_fns:
                recons[label] = _reconstruct(enc_fn, dec_fn, orig_flat).reshape(
                    resolution, resolution
                )
            entries.append(
                (tidx, orig_flat.reshape(resolution, resolution), recons)
            )
        entries.sort(key=lambda e: e[0])

        if not entries:
            continue

        # ── Figure 16a: all training times, all methods ─────────────────────
        n_cols = len(entries)
        n_rows = 1 + len(model_fns)  # GT + each method
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(FIG_WIDTH, FIELD_ROW_HEIGHT * n_rows),
            squeeze=False,
        )

        row_labels = ["Ground truth"] + [m[0] for m in model_fns]

        for col, (tidx, orig_2d, recons) in enumerate(entries):
            # Determine shared vmin/vmax across GT and all reconstructions
            all_fields = [orig_2d] + [recons[m[0]] for m in model_fns]
            vmin = float(min(f.min() for f in all_fields))
            vmax = float(max(f.max() for f in all_fields))

            # Row 0: ground truth
            axes[0, col].imshow(orig_2d, origin="lower", cmap="viridis",
                                vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"$t_{{{tidx}}}$", fontsize=FONT_TITLE)
            axes[0, col].axis("off")

            # Rows 1+: reconstructions
            for row_i, (label, color, _, _) in enumerate(model_fns, start=1):
                axes[row_i, col].imshow(recons[label], origin="lower",
                                        cmap="viridis", vmin=vmin, vmax=vmax)
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

        fig.subplots_adjust(left=0.10, right=0.995, top=0.96, bottom=0.05,
                            hspace=0.06, wspace=0.04)
        _save_fig(fig, out_dir, f"fig16_sample{sid}_all_times")

        # ── Figure 16b: t7 zoom with error maps ────────────────────────────
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
            figsize=(FIG_WIDTH, FIELD_ROW_HEIGHT * 2.35),
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

        for col_i, (label, color, _, _) in enumerate(model_fns):
            recon = recons_t7[label]
            diff = diff_maps[label]
            rmse = float(np.sqrt(np.mean((orig_t7 - recon) ** 2)))

            # Row 1: reconstruction
            axes_z[0, col_i].imshow(recon, origin="lower", cmap="viridis",
                                    vmin=vmin_r, vmax=vmax_r)
            axes_z[0, col_i].set_title(f"{label}\nRMSE={rmse:.4f}",
                                       fontsize=FONT_TITLE)
            axes_z[0, col_i].axis("off")

            # Row 2: signed difference (aligned under each method)
            im = axes_z[1, col_i].imshow(
                diff,
                origin="lower",
                cmap="RdBu_r",
                vmin=-diff_absmax,
                vmax=diff_absmax,
            )
            axes_z[1, col_i].axis("off")

        # Row labels (publication-friendly; avoids per-panel title collisions)
        axes_z[0, 0].set_ylabel(
            "Reconstruction",
            fontsize=FONT_LABEL,
            rotation=0,
            ha="right",
            va="center",
            labelpad=24,
        )
        axes_z[1, 0].set_ylabel(
            r"$\hat{u}-u$",
            fontsize=FONT_LABEL,
            rotation=0,
            ha="right",
            va="center",
            labelpad=24,
        )

        # Shared colorbar for signed differences; dedicated axis prevents overlap.
        fig_z.subplots_adjust(left=0.10, right=0.93, top=0.95, bottom=0.07,
                      hspace=0.18, wspace=0.06)
        cax = fig_z.add_axes([0.945, 0.12, 0.015, 0.33])
        cbar = fig_z.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_TICK)
        cbar.set_label(r"$\hat{u} - u$", fontsize=FONT_LABEL)
        _save_fig(fig_z, out_dir, f"fig16_sample{sid}_t7_zoom")


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
    _fallback = [C_OBS, C_GEN, "#ff7f0e", "#9467bd", "#2ca02c",
                 "#8c564b", "#e377c2", "#17becf"]

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
    args = parser.parse_args()

    # Global publication-style matplotlib defaults (fonts/cmap/mathtext).
    format_for_paper()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {out_dir.resolve()}")

    steps = [
        ("fig1:  Training convergence (2-panel: L2 vs NTK)",
         lambda: fig1_training_convergence(out_dir, args.max_steps)),
        ("fig2:  Performance bar chart (deterministic methods)",
         lambda: fig2_performance_bars(out_dir)),
        ("fig3:  Observation-time curves",
         lambda: fig3_observation_time(out_dir)),
        ("fig4:  Held-out interpolation",
         lambda: fig4_held_out(out_dir)),
        ("fig5:  Latent collapse diagnostics",
         lambda: fig5_latent_diagnostics(out_dir)),
        ("fig6:  Scale comparison per time",
         lambda: fig6_scale_comparison(out_dir)),
        ("fig7:  Architecture comparison (Muon+$\\ell_2$ vs Adam+$\\ell_2$+Prior)",
         lambda: fig7_architecture_comparison(out_dir)),
        ("fig8:  Summary table (PNG + .tex)",
         lambda: fig8_summary_table(out_dir)),
        ("fig9:  Two-panel training (single vs multi scale)",
         lambda: fig9_twoscale_training(out_dir, args.max_steps)),
        ("fig10: Reconstruction fields (t1-t7, Adam+NTK+Prior)",
         lambda: fig10_reconstruction_fields(out_dir)),
        ("fig11: Denoiser comparison (separate evaluation protocol)",
         lambda: fig11_denoiser_comparison(out_dir)),
        ("fig12: PSD spectral analysis (requires psd_data.npz)",
         lambda: fig12_psd_spectral(out_dir)),
        ("fig13: Latent-regularization evidence (requires psd_latent_metrics.json)",
         lambda: fig13_latent_regularization(out_dir)),
        ("fig14: Per-marginal latent diagnostics (requires psd_latent_metrics.json)",
         lambda: fig14_per_marginal_latent(out_dir)),
        ("fig15: Inter-marginal distance regularity (requires psd_latent_metrics.json)",
         lambda: fig15_inter_marginal_distance(out_dir)),
        ("fig16: NTK sample comparison (Adam+L2 vs Adam+NTK vs Adam+NTK+Prior)",
         lambda: fig16_ntk_sample_comparison(out_dir)),
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
