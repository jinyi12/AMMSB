from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scripts.fae.tran_evaluation.latent_geometry_model_summary import (
    LOSS_ORDER,
    MODEL_METRICS,
    TRACK_ORDER,
    _safe_float,
    _summary_sort_key,
)
from scripts.images.field_visualization import EASTERN_HUES


_FIG_W_BARS = 5.5
_FIG_W_CHAIN = 4.9
_FIG_H = 2.5
_FONT_LABEL = 7
_FONT_LEGEND = 6.5
_FONT_TICK = 7


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _apply_sci_yticks(ax: plt.Axes, yscale: str | None) -> None:
    """Apply ×10^n scientific notation to linear-scale y-axes with large values."""
    if yscale == "log":
        return
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)


def remove_legacy_multi_metric_figures(out_dir: Path) -> None:
    """Remove pre-refactor grid figures to avoid confusing output directories."""
    legacy_stems = [
        "latent_geom_model_metric_matrix",
        "latent_geom_l2_ntk_prior_chain",
        "latent_geom_ntk_effect",
        "latent_geom_prior_effect",
        "latent_geom_model_flags",
    ]
    for stem in legacy_stems:
        for ext in ("png", "pdf"):
            path = out_dir / f"{stem}.{ext}"
            if path.exists():
                path.unlink()

    for prefix in ("latent_geom_ntk_effect_", "latent_geom_prior_effect_"):
        for ext in ("png", "pdf"):
            for path in out_dir.glob(f"{prefix}*.{ext}"):
                path.unlink()


def plot_model_metric_bars(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    if not summaries:
        return

    ordered = sorted(summaries, key=_summary_sort_key)
    x = np.arange(len(ordered), dtype=np.float64)
    optimizers_present = {
        str(row.get("optimizer", "")).lower()
        for row in ordered
        if str(row.get("optimizer", "")).strip()
    }

    def _config_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("track", "")),
            str(row.get("decoder_type", "")),
            str(row.get("scale", "")),
            str(row.get("loss_type", "")),
            int(row.get("prior_flag", 0)),
        )

    def _config_sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        track, decoder, scale, loss, prior = key
        return (
            TRACK_ORDER.get(str(track), 99),
            str(decoder),
            str(scale),
            LOSS_ORDER.get(str(loss), 99),
            int(prior),
        )

    configs = sorted({_config_key(row) for row in ordered}, key=_config_sort_key)
    config_colors = {cfg: EASTERN_HUES[i % len(EASTERN_HUES)] for i, cfg in enumerate(configs)}

    def _tick_label(row: dict[str, Any]) -> str:
        loss = str(row.get("loss_type", "")).lower()
        if loss == "ntk_scaled":
            loss_tag = "NTK-Scale"
        elif loss == "ntk_prior_balanced":
            loss_tag = "NTK-Bal"
        else:
            loss_tag = loss.upper()
        prior = "+P" if int(row.get("prior_flag", 0)) == 1 else ""
        decoder = str(row.get("decoder_type", "film"))
        decoder_tag = "" if decoder == "film" else "Den"
        suffix = f" {decoder_tag}".rstrip()
        return f"{loss_tag}{prior}{suffix}".strip()

    tick_labels = [_tick_label(row) for row in ordered]
    colors = [config_colors[_config_key(row)] for row in ordered]
    hatches = []
    for row in ordered:
        opt = str(row.get("optimizer", "")).lower()
        if opt == "muon":
            hatches.append("//")
        elif opt == "adam":
            hatches.append("")
        else:
            hatches.append("..")

    for spec in MODEL_METRICS:
        key = spec["key"]
        vals = np.asarray([_safe_float(row.get(key)) for row in ordered], dtype=np.float64)
        std_key = key.replace("_mean_over_time", "_std_over_time")
        errs = np.asarray([_safe_float(row.get(std_key)) for row in ordered], dtype=np.float64)
        errs = np.where(np.isfinite(errs), errs, 0.0)

        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_BARS, _FIG_H))
        bars = ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="black",
            alpha=0.9,
            yerr=errs,
            capsize=2.0,
            error_kw={"linewidth": 0.8, "alpha": 0.8},
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=_FONT_TICK)
        ax.set_ylabel(spec["display"], fontsize=_FONT_LABEL)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        _apply_sci_yticks(ax, yscale)

        if "adam" in optimizers_present or "muon" in optimizers_present:
            from matplotlib.patches import Patch

            legend_items = []
            if "adam" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="", label="ADAM"))
            if "muon" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="MUON"))
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left", frameon=False, fontsize=_FONT_LEGEND)

        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_model_metric_{spec['label']}")


def plot_ntk_prior_chain(
    summaries: list[dict[str, Any]],
    effects: dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    rows = list(effects.get("ntk_prior_chain", []))
    if not rows:
        return

    cell_index = {str(row.get("matrix_cell_id", "")): row for row in summaries}
    x = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    x_labels = ["L2", "NTK", "NTK+Prior"]
    stage_colors = [EASTERN_HUES[0], EASTERN_HUES[1], EASTERN_HUES[2]]

    for spec in MODEL_METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_CHAIN, _FIG_H))
        key = spec["key"]
        std_key = key.replace("_mean_over_time", "_std_over_time")

        seen_labels: set[str] = set()
        for row in rows:
            std = cell_index.get(str(row.get("standard", "")))
            ntk = cell_index.get(str(row.get("ntk", "")))
            ntk_prior = cell_index.get(str(row.get("ntk_prior", "")))
            if std is None or ntk is None or ntk_prior is None:
                continue
            y0 = _safe_float(std.get(key))
            y1 = _safe_float(ntk.get(key))
            y2 = _safe_float(ntk_prior.get(key))
            if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
                continue
            e0 = _safe_float(std.get(std_key))
            e1 = _safe_float(ntk.get(std_key))
            e2 = _safe_float(ntk_prior.get(std_key))
            e0 = 0.0 if not np.isfinite(e0) else float(e0)
            e1 = 0.0 if not np.isfinite(e1) else float(e1)
            e2 = 0.0 if not np.isfinite(e2) else float(e2)

            yscale = str(spec.get("yscale") or "")
            if yscale == "log" and (y0 <= 0.0 or y1 <= 0.0 or y2 <= 0.0):
                continue

            optimizer_key = str(row.get("optimizer", "")).lower()
            optimizer = optimizer_key.upper()
            label = optimizer if optimizer and optimizer not in seen_labels else None
            if label is not None:
                seen_labels.add(label)

            linestyle = "-" if optimizer_key == "adam" else "--"
            x_off = x + (-0.04 if optimizer_key == "adam" else 0.04)
            ax.errorbar(
                x_off,
                [y0, y1, y2],
                yerr=[e0, e1, e2],
                fmt="none",
                ecolor="0.35",
                elinewidth=0.9,
                capsize=2.0,
            )
            ax.plot(
                x_off,
                [y0, y1, y2],
                linestyle=linestyle,
                linewidth=1.2,
                color="0.35",
                label=label,
            )
            ax.scatter(
                x_off,
                [y0, y1, y2],
                s=28,
                c=stage_colors,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=_FONT_TICK)
        ax.set_ylabel(spec["display"], fontsize=_FONT_LABEL)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        _apply_sci_yticks(ax, yscale)

        if seen_labels:
            from matplotlib.lines import Line2D

            legend_items = []
            if "ADAM" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="-", label="ADAM"))
            if "MUON" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="MUON"))
            if legend_items:
                ax.legend(
                    handles=legend_items,
                    loc="upper center",
                    ncol=len(legend_items),
                    frameon=False,
                    fontsize=_FONT_LEGEND,
                )

        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_l2_ntk_prior_chain_{spec['label']}")
