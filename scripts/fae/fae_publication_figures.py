"""FAE Publication Figures — SIAM journal submission.

Generates the complete set of publication-quality figures for the FAE
multiscale RFF and optimizer ablation studies.  Follows the visual style
of ``scripts/fae/tran_evaluation/report.py`` (7-inch figures, consistent
fonts, PNG + PDF output).

Three experiment families are covered:

  Optimizer Study (§ Opt)   —  fixed architecture (FiLM, σ = {1,2,4,8}),
                                varying Adam/Muon × L2/NTK-scaled loss.

  Scale Study (§ Scale)     —  fixed architecture + Muon + L2, comparing
                                single-scale (σ = 1) vs multiscale (σ = {1,2,4,8}).

  Architecture Study (§ Arc) — fixed Muon + σ = {1,2,4,8}, comparing
                                Det-FiLM (L2) vs FiLM+Prior (denoiser ELBO).

Usage::

    python scripts/fae/fae_publication_figures.py [--out-dir results/publication]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray


# ============================================================================
# Registry — all experiment runs used in the paper
# ============================================================================

#: (label, run_dir, group, linestyle, color)
RUNS: List[Tuple[str, str, str, str, str]] = [
    # ── Optimizer study: σ = {1, 2, 4, 8} ──────────────────────────────────
    (
        r"Adam + $\ell_2$",
        "results/fae_film_adam_l2_99pct/run_bnqm4evk",
        "opt",
        "-",
        "#d62728",   # red
    ),
    (
        "Adam + NTK",
        "results/fae_film_adam_ntk_99pct/run_2hnr5shv",
        "opt",
        "--",
        "#ff7f0e",   # orange
    ),
    (
        "Muon + NTK",
        "results/fae_film_muon_ntk_99pct/run_tug7ucuw",
        "opt",
        "-.",
        "#9467bd",   # purple
    ),
    (
        r"Muon + $\ell_2$  (ours)",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "opt",
        "-",
        "#2ca02c",   # green (best)
    ),
    # ── Scale study: vary σ, fixed Muon + L2 ────────────────────────────────
    (
        r"Muon, $\sigma\!=\!1$",
        "results/fae_deterministic_film/run_90ndogk3",
        "scale",
        "--",
        "#1f77b4",   # blue
    ),
    (
        r"Muon, $\sigma\!\in\!\{1,2,4,8\}$",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "scale",
        "-",
        "#2ca02c",   # green
    ),
    # ── Architecture study: FiLM vs FiLM+Prior, σ = {1,2,4,8} ──────────────
    (
        r"Det-FiLM ($\ell_2$ reg)",
        "results/fae_deterministic_film_multiscale/run_ujlkslav",
        "arch",
        "-",
        "#2ca02c",   # green
    ),
    (
        "FiLM + Prior (denoiser)",
        "results/fae_film_prior_multiscale/run_66nrnp5e",
        "arch",
        "--",
        "#8c564b",   # brown
    ),
]

# Short tags used for table rows / tick labels
SHORT_LABEL: Dict[str, str] = {
    "results/fae_film_adam_l2_99pct/run_bnqm4evk":                r"Adam+$\ell_2$",
    "results/fae_film_adam_ntk_99pct/run_2hnr5shv":               "Adam+NTK",
    "results/fae_film_muon_ntk_99pct/run_tug7ucuw":               "Muon+NTK",
    "results/fae_deterministic_film_multiscale/run_ujlkslav":      r"Muon+$\ell_2$*",
    "results/fae_deterministic_film/run_90ndogk3":                 r"Muon, $\sigma$=1",
    "results/fae_film_prior_multiscale/run_66nrnp5e":              "FiLM+Prior",
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


# ============================================================================
# Data loaders
# ============================================================================

def _load_training_loss(run_dir: str) -> Optional[NDArray]:
    p = os.path.join(run_dir, "logs", "training_loss.npy")
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)


def _load_eval(run_dir: str) -> Optional[dict]:
    p = os.path.join(run_dir, "eval_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _ema_smooth(y: NDArray, alpha: float = EMA_ALPHA) -> NDArray:
    """Exponential moving average smoothing."""
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


# ============================================================================
# Figure 1: Training convergence — Optimizer ablation (σ = {1,2,4,8})
# ============================================================================

def fig1_training_convergence(out_dir: Path, max_steps: int = 50_000) -> None:
    """Training loss curves for the optimizer ablation study.

    Single panel, log-scale y-axis.  Smoothed curve (solid) over raw
    values (translucent fill) for each method.  Demonstrates that Muon
    converges faster and to lower loss than Adam variants.
    """
    opt_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS if grp == "opt"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    for label, d, ls, color in opt_runs:
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
    ax.set_title(
        r"Training convergence — optimizer ablation, $\sigma \in \{1,2,4,8\}$",
        fontsize=FONT_TITLE,
    )
    ax.grid(which="both", alpha=0.25, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper right")
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig1_training_convergence")


# ============================================================================
# Figure 2: Test performance bar chart — Optimizer + Scale studies
# ============================================================================

def fig2_performance_bars(out_dir: Path) -> None:
    """Grouped bar chart of test relative MSE across optimizer and scale study.

    Two bar groups displayed side by side:
      (a) Optimizer study: fixed σ = {1,2,4,8}, vary optimizer/loss.
      (b) Scale study: fixed Muon + L2, vary σ.

    The colour scheme matches Figure 1.  Bars are annotated with the
    numeric value for easy reading.
    """
    # ── Gather data ──────────────────────────────────────────────────────────
    opt_data = []   # (label, rel_mse, color)
    for label, d, grp, ls, color in RUNS:
        if grp not in {"opt", "scale"}:
            continue
        # deduplicate: det_film_multiscale appears in both opt and scale
        ev = _load_eval(d)
        if ev is None:
            continue
        opt_data.append((label, float(ev["test_rel_mse"]), color, grp))

    # Remove exact duplicates (same (d, grp) pair) – keep first.
    seen: set = set()
    unique: list = []
    for item in opt_data:
        key = (item[0], item[3])
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
        "FAE reconstruction quality — optimizer and scale comparison",
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
    # Select key runs for this figure: optimizer study + architecture study
    sel_runs = [(label, d, ls, c) for label, d, grp, ls, c in RUNS
                if grp in {"opt", "arch"}]
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
        if grp not in {"opt"}:  # show only optimizer study for clarity
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
    ax.set_title("Held-out interpolation accuracy — optimizer comparison",
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

    Three horizontal-bar subplots:
      (a) ``decode_sensitivity``  — ratio of full MSE to zero-latent MSE;
          value >> 1 means the decoder genuinely uses the latent code.
      (b) ``zero_latent_mse / full_mse``  — how much worse a zero-code
          decode is; large ratio = healthy utilisation.
      (c) ``latent_var_mean``  — mean variance of latent codes across the
          test batch; near-zero indicates posterior collapse.

    Only runs that have non-null collapse_diagnostics are included.
    """
    entries: list = []
    for label, d, grp, ls, color in RUNS:
        if grp not in {"opt", "scale"}:
            continue
        ev = _load_eval(d)
        if ev is None or not ev.get("collapse_diagnostics"):
            continue
        cd = ev["collapse_diagnostics"]
        entries.append((label, cd, color))

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
    zlr     = [float(e[1]["zero_latent_mse"]) / max(1e-12, float(
                   _load_eval([x[1] for x in RUNS if x[0] == e[0]][0])["test_mse"]
               )) for e in entries]
    latvar  = [float(e[1]["latent_var_mean"]) for e in entries]

    fig, axes = plt.subplots(1, 3,
                             figsize=(FIG_WIDTH, SUBPLOT_HEIGHT),
                             sharey=True)

    def _hbar(ax: plt.Axes, vals: List[float], title: str,
              xlabel: str, ref_line: Optional[float] = None) -> None:
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, edgecolor="k", linewidth=0.6,
                height=0.60, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        if ref_line is not None:
            ax.axvline(ref_line, color="grey", ls="--", lw=0.9, alpha=0.7,
                       zorder=4)
        ax.grid(axis="x", alpha=0.3, color=C_GRID)
        _set_tick_fontsize(ax)

    _hbar(axes[0], sens,   "Decode sensitivity",
          r"$\mathrm{MSE}_0 / \mathrm{MSE}_z$", ref_line=1.0)
    _hbar(axes[1], zlr,    "Zero-latent ratio",
          r"$\mathrm{MSE}_{z=0} / \mathrm{MSE}_{\mathrm{test}}$")
    _hbar(axes[2], latvar, "Latent variance",
          r"$\mathrm{Var}[z]$")

    axes[0].invert_yaxis()

    fig.suptitle("Latent-space utilisation diagnostics", fontsize=FONT_TITLE + 1,
                 y=1.01)
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
# Figure 7: Architecture comparison — Det-FiLM vs FiLM+Prior
# ============================================================================

def fig7_architecture_comparison(out_dir: Path) -> None:
    """Per-time-fraction bar comparison for Det-FiLM vs FiLM+Prior.

    Same format as Figure 6, but contrasts the two decoder architectures.
    Note: FiLM+Prior is optimised with a denoiser ELBO; its test MSE
    reflects reconstruction after prior-based sampling, not L2 training.
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
        r"Architecture comparison — Det-FiLM vs FiLM+Prior, $\sigma\!\in\!\{1,2,4,8\}$",
        fontsize=FONT_TITLE,
    )
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25, color=C_GRID)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
    _set_tick_fontsize(ax)

    fig.text(
        0.5, -0.06,
        r"\textit{Note: FiLM+Prior trained with denoiser ELBO; Det-FiLM trained with $\ell_2$.}",
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
    G_COLORS = {"opt": "#f7f9ff", "scale": "#f9fff7", "arch": "#fff9f7"}
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
                        if grp == "opt"]

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
# Figure 10: Reconstruction fields from best model
# ============================================================================

def fig10_reconstruction_fields(out_dir: Path) -> None:
    """Reconstruction image grid from the best-performing model.

    Loads existing PNG snapshots written by the trainer at evaluation
    intervals and composites them into a single figure:
    row 1 — held-out time points (sparse observation),
    row 2 — training time points (dense observation).

    Images are resized if necessary to the first-found resolution.
    """
    best_dir = Path("results/fae_deterministic_film_multiscale/run_ujlkslav/figures")
    if not best_dir.exists():
        print("  [fig10] figures/ directory not found, skipping.")
        return

    # Read the existing PNGs produced by the trainer
    from PIL import Image as PILImage  # type: ignore

    held_figs   = sorted(best_dir.glob("recon_t*_held_out.png"))
    train_figs  = sorted(best_dir.glob("recon_t*_train.png"))

    # Limit to 4 per row maximum
    held_figs  = held_figs[:4]
    train_figs = train_figs[:4]

    n_held  = len(held_figs)
    n_train = len(train_figs)
    n_rows  = (1 if n_held  > 0 else 0) + \
               (1 if n_train > 0 else 0)
    n_cols  = max(n_held, n_train)

    if n_rows == 0 or n_cols == 0:
        print("  [fig10] no reconstruction PNGs found, skipping.")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH, FIELD_ROW_HEIGHT * n_rows + 0.5),
    )
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    def _show(ax: plt.Axes, path: Path, title: str) -> None:
        img = np.asarray(PILImage.open(path))
        ax.imshow(img, interpolation="lanczos")
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.axis("off")

    row = 0
    if n_held > 0:
        for col, p in enumerate(held_figs):
            idx = int(p.stem.split("_t")[1].split("_")[0])
            _show(axes[row, col], p, "Held-out $t_{%d}$" % idx)
        for col in range(n_held, n_cols):
            axes[row, col].axis("off")
        row += 1
    if n_train > 0:
        for col, p in enumerate(train_figs):
            idx = int(p.stem.split("_t")[1].split("_")[0])
            _show(axes[row, col], p, "Training $t_{%d}$" % idx)
        for col in range(n_train, n_cols):
            axes[row, col].axis("off")

    fig.suptitle(
        r"Field reconstructions — Muon + $\ell_2$, $\sigma\!\in\!\{1,2,4,8\}$ (best model)",
        fontsize=FONT_TITLE + 1,
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, out_dir, "fig10_reconstruction_fields")


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {out_dir.resolve()}")

    steps = [
        ("fig1:  Training convergence (optimizer ablation)",
         lambda: fig1_training_convergence(out_dir, args.max_steps)),
        ("fig2:  Performance bar chart",
         lambda: fig2_performance_bars(out_dir)),
        ("fig3:  Observation-time curves",
         lambda: fig3_observation_time(out_dir)),
        ("fig4:  Held-out interpolation",
         lambda: fig4_held_out(out_dir)),
        ("fig5:  Latent collapse diagnostics",
         lambda: fig5_latent_diagnostics(out_dir)),
        ("fig6:  Scale comparison per time",
         lambda: fig6_scale_comparison(out_dir)),
        ("fig7:  Architecture comparison per time",
         lambda: fig7_architecture_comparison(out_dir)),
        ("fig8:  Summary table (PNG + .tex)",
         lambda: fig8_summary_table(out_dir)),
        ("fig9:  Two-panel training (single vs multi scale)",
         lambda: fig9_twoscale_training(out_dir, args.max_steps)),
        ("fig10: Reconstruction field grid",
         lambda: fig10_reconstruction_fields(out_dir)),
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
