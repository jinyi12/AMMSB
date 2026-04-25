from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scripts.fae.tran_evaluation.latent_geometry_model_summary import (
    MODEL_METRICS,
    _safe_float,
)
from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    publication_figure_width,
    publication_style_tokens,
)


_PUB_STYLE = publication_style_tokens()
_FIG_W_BARS = publication_figure_width(column_span=1)
_FIG_W_DELTA = publication_figure_width(column_span=2, fraction=0.58)
_FIG_H = 2.45
_FONT_LABEL = _PUB_STYLE["font_label"]
_FONT_LEGEND = _PUB_STYLE["font_legend"]
_FONT_TICK = _PUB_STYLE["font_tick"]
_FONT_TITLE = _PUB_STYLE["font_title"]

_BASELINE_COLOR = EASTERN_HUES[4]
_TREATMENT_COLOR = EASTERN_HUES[7]


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _apply_sci_yticks(ax: plt.Axes, yscale: str | None) -> None:
    if yscale == "log":
        return
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)


def _resolve_metric_yscale(spec: dict[str, Any], vals: np.ndarray) -> str | None:
    explicit = spec.get("yscale")
    if explicit:
        return str(explicit)
    finite_positive = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite_positive.size >= 2:
        dynamic_range = float(np.max(finite_positive) / np.min(finite_positive))
        if dynamic_range >= 1e3:
            return "log"
    return None


def _annotate_bars(ax: plt.Axes, vals: np.ndarray, yscale: str | None) -> None:
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        return
    if yscale == "log":
        positive_vals = finite_vals[finite_vals > 0.0]
        if positive_vals.size == 0:
            return
        for idx, value in enumerate(vals):
            if not np.isfinite(value) or value <= 0.0:
                continue
            ax.text(
                idx,
                value * 1.12,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=_FONT_LEGEND,
            )
        return

    span = float(np.max(finite_vals) - np.min(finite_vals))
    offset = 0.03 * span if span > 0.0 else max(abs(float(finite_vals[0])) * 0.04, 0.02)
    for idx, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        ax.text(
            idx,
            value + offset,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=_FONT_LEGEND,
        )


def remove_legacy_pairwise_outputs(out_dir: Path) -> None:
    legacy_stems = [
        "latent_geom_model_metric_matrix",
        "latent_geom_l2_ntk_prior_chain",
        "latent_geom_l2_ntk_diffusion_prior_chain",
        "latent_geom_l2_ntk_sigreg_chain",
        "latent_geom_ntk_effect",
        "latent_geom_prior_effect",
        "latent_geom_diffusion_prior_effect",
        "latent_geom_sigreg_effect",
        "latent_geom_model_flags",
        "latent_geom_model_summary",
    ]
    for stem in legacy_stems:
        for ext in ("png", "pdf", "json", "csv", "md"):
            path = out_dir / f"{stem}.{ext}"
            if path.exists():
                path.unlink()

    for prefix in (
        "latent_geom_ntk_effect_",
        "latent_geom_prior_effect_",
        "latent_geom_diffusion_prior_effect_",
        "latent_geom_sigreg_effect_",
        "latent_geom_l2_ntk_",
        "latent_geom_model_metric_",
    ):
        for ext in ("png", "pdf"):
            for path in out_dir.glob(f"{prefix}*.{ext}"):
                path.unlink()


def plot_pair_metric_bars(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    if len(summaries) != 2:
        return

    format_for_paper()
    ordered = sorted(summaries, key=lambda row: str(row.get("run_role", "")))
    labels = [
        str(row.get("run_label", row.get("run_role", ""))).strip() or str(row.get("run_role", "run"))
        for row in ordered
    ]
    colors = [_BASELINE_COLOR, _TREATMENT_COLOR]
    x = np.arange(2, dtype=np.float64)

    for spec in MODEL_METRICS:
        key = spec["key"]
        std_key = key.replace("_mean_over_time", "_std_over_time")
        vals = np.asarray([_safe_float(row.get(key)) for row in ordered], dtype=np.float64)
        errs = np.asarray([_safe_float(row.get(std_key)) for row in ordered], dtype=np.float64)
        errs = np.where(np.isfinite(errs), errs, 0.0)

        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_BARS, _FIG_H))
        ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="black",
            yerr=errs,
            capsize=2.0,
            error_kw={"linewidth": 0.8, "alpha": 0.8},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=_FONT_TICK)
        ax.set_ylabel(spec["display"], fontsize=_FONT_LABEL)
        ax.set_title(spec["title"], fontsize=_FONT_TITLE)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)
        yscale = _resolve_metric_yscale(spec, vals)
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        elif yscale == "log":
            finite_positive = vals[np.isfinite(vals) & (vals > 0.0)]
            if finite_positive.size:
                ax.set_ylim(float(np.min(finite_positive) / 1.8), float(np.max(finite_positive) * 2.2))
        _apply_sci_yticks(ax, yscale)
        _annotate_bars(ax, vals, yscale)

        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=_BASELINE_COLOR, edgecolor="black", label=labels[0]),
                Patch(facecolor=_TREATMENT_COLOR, edgecolor="black", label=labels[1]),
            ],
            loc="upper left",
            frameon=False,
            fontsize=_FONT_LEGEND,
        )
        fig.tight_layout(pad=0.7)
        _save_fig(fig, out_dir, f"latent_geom_pair_metric_{spec['label']}")


def plot_pair_time_deltas(pairwise: dict[str, Any], *, out_dir: Path) -> None:
    per_time = list(pairwise.get("per_time", []))
    if not per_time:
        return

    format_for_paper()
    x = np.asarray([int(row.get("dataset_time_index", idx)) for idx, row in enumerate(per_time)], dtype=np.int64)
    baseline_label = str(dict(pairwise.get("baseline", {})).get("run_label", "Baseline"))
    treatment_label = str(dict(pairwise.get("treatment", {})).get("run_label", "Treatment"))

    for spec in MODEL_METRICS:
        y_vals: list[float] = []
        for row in per_time:
            metrics = list(row.get("metrics", []))
            match = next((metric for metric in metrics if str(metric.get("metric_key", "")) == spec["key"]), None)
            y_vals.append(_safe_float(None if match is None else match.get("signed_relative_delta")))
        y = np.asarray(y_vals, dtype=np.float64)

        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_DELTA, _FIG_H))
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        ax.plot(x, y, color=_TREATMENT_COLOR, linewidth=1.3)
        ax.scatter(x, y, s=22, color=_TREATMENT_COLOR, edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_xlabel(r"Modeled time index $t$", fontsize=_FONT_LABEL)
        ax.set_ylabel("Signed relative improvement (%)", fontsize=_FONT_LABEL)
        ax.set_title(
            f"{spec['title']}: {treatment_label} vs {baseline_label}",
            fontsize=_FONT_TITLE,
        )
        ax.grid(alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        finite_y = y[np.isfinite(y)]
        if finite_y.size:
            y_lim = 1.12 * max(float(np.max(np.abs(finite_y))), 0.05)
            ax.set_ylim(-y_lim, y_lim)
        fig.tight_layout(pad=0.7)
        _save_fig(fig, out_dir, f"latent_geom_pair_time_delta_{spec['label']}")
