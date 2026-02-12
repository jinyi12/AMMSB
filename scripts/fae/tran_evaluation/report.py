"""Phase 7 – Visualisation and reporting.

Generates the complete set of diagnostic figures and summary tables
prescribed by the Tran-aligned evaluation plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from scripts.images.field_visualization import format_for_paper  # noqa: F401


# ============================================================================
# Colour palette
# ============================================================================
C_OBS = "#2166ac"
C_GEN = "#b2182b"
C_FILL = "#d6604d"

# ============================================================================
# Colormap
# ============================================================================
CMAP_FIELD = "viridis"
CMAP_DIFF = "RdBu_r"

# ============================================================================
# Font sizes (tuned for multi-panel figures at 7-inch width)
# ============================================================================
FONT_TITLE = 8
FONT_LABEL = 7
FONT_LEGEND = 6.5
FONT_TICK = 7

# ============================================================================
# Standard layout constants
# ============================================================================
FIG_WIDTH = 7.0                # inches, all multi-panel figures
SUBPLOT_HEIGHT = 2.5           # inches per row for grid plots (PDF, corr, PSD)
SUBPLOT_HEIGHT_SQ = 2.3        # inches per row for square subplots (QQ)
FIELD_ROW_HEIGHT = 1.75        # inches per row for field-image grids
N_COLS = 3                     # standard columns for grid plots


# ============================================================================
# Helpers
# ============================================================================

def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save *fig* as both PNG (dpi=150) and PDF with tight bounding box."""
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"{name}.{ext}",
            dpi=150,
            bbox_inches="tight",
        )


def _band_label(band: int, H_schedule: Optional[List[float]] = None) -> str:
    """Professional label for detail band d_l = u_{H_l} - u_{H_{l+1}}."""
    if H_schedule is not None and band + 1 < len(H_schedule):
        H_lo = H_schedule[band]
        H_hi = H_schedule[band + 1]
        return f"$u_{{{H_lo:.3g}}} - u_{{{H_hi:.3g}}}$"
    return f"Band {band}"


def _set_tick_fontsize(ax, size: float = FONT_TICK) -> None:
    """Set tick label font size on both axes."""
    ax.tick_params(axis="both", labelsize=size)


def _sample_finite(
    values: NDArray,
    rng: np.random.Generator,
    *,
    max_samples: int,
) -> NDArray[np.float64]:
    """Return up to max_samples finite values sampled from *values* (flattened).

    For large arrays, samples with replacement to avoid O(N) indexing/masks.
    Intended for plotting only (not for metrics).
    """
    flat = np.asarray(values).ravel()
    n = int(flat.size)
    if n == 0:
        return np.asarray([], dtype=np.float64)

    if n <= max_samples:
        sampled = flat
        sampled = sampled[np.isfinite(sampled)]
        return np.asarray(sampled, dtype=np.float64)

    # Sample with replacement for speed; oversample to survive NaN/Inf filtering.
    target = int(max_samples)
    oversample = int(min(max(2 * target, 10_000), 5_000_000))

    out: list[NDArray] = []
    remaining = target
    for _ in range(5):
        idx = rng.integers(0, n, size=oversample, endpoint=False)
        chunk = flat[idx]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            oversample = min(2 * oversample, 10_000_000)
            continue
        if chunk.size > remaining:
            chunk = chunk[:remaining]
        out.append(np.asarray(chunk, dtype=np.float64))
        remaining -= int(chunk.size)
        if remaining <= 0:
            break
        oversample = min(2 * oversample, 10_000_000)

    if not out:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(out, axis=0)


def _fd_nbins(values: NDArray[np.float64], *, min_bins: int, max_bins: int) -> int:
    """Freedman–Diaconis-style bin count with sane clamps."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return min_bins

    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return min_bins

    q25, q75 = np.percentile(v, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        nb = int(np.sqrt(v.size))
        return int(np.clip(nb, min_bins, max_bins))

    bin_width = 2.0 * iqr * (v.size ** (-1.0 / 3.0))
    if bin_width <= 1e-12:
        return min_bins

    nb = int(np.ceil((vmax - vmin) / bin_width))
    return int(np.clip(nb, min_bins, max_bins))


def _gaussian_smooth_1d(y: NDArray[np.float64], sigma_bins: float) -> NDArray[np.float64]:
    if sigma_bins <= 0:
        return np.asarray(y, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        return np.asarray(
            gaussian_filter1d(np.asarray(y, dtype=np.float64), sigma=sigma_bins, mode="nearest"),
            dtype=np.float64,
        )
    except Exception:
        # Lightweight fallback without extra imports.
        sigma_bins = float(sigma_bins)
        radius = int(max(1, np.ceil(3.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= float(np.sum(kernel))
        return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _density_pair_curve(
    obs_values: NDArray,
    gen_values: NDArray,
    rng: np.random.Generator,
    *,
    max_samples: int = 500_000,
    min_bins: int = 50,
    max_bins: int = 220,
    smooth_sigma_bins: float = 1.2,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fast PDF-like density curves via histogram + light smoothing."""
    obs = _sample_finite(obs_values, rng, max_samples=max_samples)
    gen = _sample_finite(gen_values, rng, max_samples=max_samples)

    if obs.size == 0 or gen.size == 0:
        x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        y0 = np.zeros_like(x)
        return x, y0, y0

    all_vals = np.concatenate([obs, gen], axis=0)
    x_lo = float(np.min(all_vals))
    x_hi = float(np.max(all_vals))
    if x_hi <= x_lo:
        x_lo -= 1.0
        x_hi += 1.0

    margin = 0.05 * (x_hi - x_lo)
    x_lo -= margin
    x_hi += margin

    n_bins = _fd_nbins(all_vals, min_bins=min_bins, max_bins=max_bins)

    y_obs, edges = np.histogram(obs, bins=n_bins, range=(x_lo, x_hi), density=True)
    y_gen, _ = np.histogram(gen, bins=edges, density=True)

    x = 0.5 * (edges[:-1] + edges[1:])
    y_obs_s = _gaussian_smooth_1d(y_obs.astype(np.float64, copy=False), smooth_sigma_bins)
    y_gen_s = _gaussian_smooth_1d(y_gen.astype(np.float64, copy=False), smooth_sigma_bins)

    return x.astype(np.float64, copy=False), y_obs_s, y_gen_s


# ============================================================================
# Figure 1 – Conditioning consistency
# ============================================================================

def plot_conditioning(
    condition: NDArray,
    filtered_macro: NDArray,
    resolution: int,
    out_dir: Path,
) -> None:
    """Conditioning-field consistency check.

    **What it shows.**  Top row: the macro-scale conditioning field *c_i*
    alongside several generated fields re-filtered to the same macro scale.
    Bottom row: self-difference (zero sanity check) followed by pixel-wise
    difference maps with a shared colour scale.

    **Diagnostic value.**  Verifies that macro-scale conditioning is
    preserved.  Generated fields, when re-filtered to the macro scale,
    should match the conditioning field almost exactly.  The leading
    zero-field column in row 2 provides a visual reference for the
    colour-scale midpoint.

    **Pass / fail.**  The mean relative error should be < 5 %.  Systematic
    bias or large outliers indicate a conditioning-enforcement bug.
    """
    K = filtered_macro.shape[0]
    n_show = min(4, K)
    n_cols = n_show + 1  # condition + n_show re-filtered

    fig, axes = plt.subplots(2, n_cols, figsize=(FIG_WIDTH, 2 * FIELD_ROW_HEIGHT))

    c_flat = condition.ravel().astype(np.float64)
    c_2d = c_flat.reshape(resolution, resolution)

    # Global field limits across condition AND all re-filtered realisations.
    field_vmin = float(c_2d.min())
    field_vmax = float(c_2d.max())
    for j in range(n_show):
        fj = filtered_macro[j].ravel()
        field_vmin = min(field_vmin, float(fj.min()))
        field_vmax = max(field_vmax, float(fj.max()))

    # Top row: coarse field + re-filtered realisations.
    ax = axes[0, 0]
    ax.imshow(c_2d, origin="lower", cmap=CMAP_FIELD,
              vmin=field_vmin, vmax=field_vmax)
    ax.set_title("Coarse $c_i$", fontsize=FONT_TITLE)
    ax.axis("off")

    for j in range(n_show):
        ax = axes[0, j + 1]
        img = filtered_macro[j].reshape(resolution, resolution)
        ax.imshow(img, origin="lower", cmap=CMAP_FIELD,
                  vmin=field_vmin, vmax=field_vmax)
        ax.set_title(f"$\\hat{{c}}^{{({j+1})}}$", fontsize=FONT_TITLE)
        ax.axis("off")

    # Bottom row: self-diff (zero reference) + difference maps.
    # Note: differences must be visualised on a separate (near-zero) scale.
    diffs = np.stack(
        [
            (filtered_macro[j].ravel().astype(np.float64) - c_flat)
            .reshape(resolution, resolution)
            for j in range(n_show)
        ],
        axis=0,
    )
    diff_abs_max = float(np.max(np.abs(diffs)))
    diff_abs_max = max(diff_abs_max, 1e-12)

    # Column 0: c_i − c_i ≡ 0 (sanity reference).
    ax = axes[1, 0]
    im_diff = ax.imshow(
        np.zeros((resolution, resolution)),
        origin="lower",
        cmap=CMAP_DIFF,
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
    )
    ax.set_title("$c_i - c_i$", fontsize=FONT_TITLE)
    ax.axis("off")

    for j in range(n_show):
        ax = axes[1, j + 1]
        im_diff = ax.imshow(
            diffs[j],
            origin="lower",
            cmap=CMAP_DIFF,
            vmin=-diff_abs_max,
            vmax=diff_abs_max,
        )
        ax.set_title(f"$\\hat{{c}}^{{({j+1})}} - c_i$", fontsize=FONT_TITLE)
        ax.axis("off")

    # Shared colorbars in dedicated axes on the right (prevents overlap).
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    cax_field = fig.add_axes([0.86, 0.55, 0.02, 0.33])
    cax_diff = fig.add_axes([0.86, 0.12, 0.02, 0.33])
    cbar_field = fig.colorbar(axes[0, 0].images[0], cax=cax_field)
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
    cbar_field.ax.tick_params(labelsize=FONT_TICK)
    cbar_diff.ax.tick_params(labelsize=FONT_TICK)

    _save_fig(fig, out_dir, "fig1_conditioning")
    plt.close(fig)


def plot_conditioning_errors(
    errors: NDArray,
    out_dir: Path,
) -> None:
    """Per-realisation relative conditioning error histogram.

    **What it shows.**  Distribution of relative L2 errors between each
    generated field (re-filtered to the macro scale) and the conditioning
    field.

    **Diagnostic value.**  Quantifies how well the generator preserves
    macro-scale structure.

    **Pass / fail.**  Mean relative error < 5 % is a pass.
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 2, SUBPLOT_HEIGHT))

    ax.hist(errors, bins=20, edgecolor="k", alpha=0.7)
    ax.axvline(np.mean(errors), color="r", ls="--", lw=1.5,
               label=f"mean={np.mean(errors):.4f}")
    ax.set_xlabel("Relative error $E^{\\mathrm{coarse}}$", fontsize=FONT_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.set_title("Coarse-field error distribution", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
    ax.grid(alpha=0.2)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig1b_conditioning_errors")
    plt.close(fig)


# ============================================================================
# Figure 2 – Sample realisations
# ============================================================================

def plot_sample_realizations(
    realizations: NDArray,
    gt_field: Optional[NDArray],
    resolution: int,
    out_dir: Path,
) -> None:
    """Sample realisations at the generator's native scale.

    **What it shows.**  A few generated micro-scale realisations side by
    side with the ground-truth field at the same scale.

    **Diagnostic value.**  Visual inspection of generated samples vs GT at
    the generator's native scale.  The generated fields should exhibit
    similar texture, value range, and spatial structure to the GT.

    **Pass / fail.**  No formal metric; look for obvious artefacts such as
    blurriness, grid patterns, or wildly different value ranges.
    """
    K = realizations.shape[0]
    n_show = min(4, K)
    n_cols = n_show + (1 if gt_field is not None else 0)

    # Shared colour limits across GT and all shown realisations.
    all_fields = [realizations[j] for j in range(n_show)]
    if gt_field is not None:
        all_fields.append(gt_field)
    shared_vmin = float(min(f.min() for f in all_fields))
    shared_vmax = float(max(f.max() for f in all_fields))

    # Use a stable height (do not shrink with n_cols) and dedicate space for the
    # shared colorbar on the right to avoid overlap/squeezing.
    fig, axes = plt.subplots(1, n_cols, figsize=(FIG_WIDTH, FIELD_ROW_HEIGHT + 0.85))
    axes = [axes] if n_cols == 1 else list(axes)

    idx = 0
    im = None
    if gt_field is not None:
        im = axes[idx].imshow(gt_field.reshape(resolution, resolution),
                              origin="lower", cmap=CMAP_FIELD,
                              vmin=shared_vmin, vmax=shared_vmax)
        axes[idx].set_title("Ground truth", fontsize=FONT_TITLE)
        axes[idx].axis("off")
        idx += 1

    for j in range(n_show):
        im = axes[idx].imshow(realizations[j].reshape(resolution, resolution),
                              origin="lower", cmap=CMAP_FIELD,
                              vmin=shared_vmin, vmax=shared_vmax)
        axes[idx].set_title(f"Realisation {j+1}", fontsize=FONT_TITLE)
        axes[idx].axis("off")
        idx += 1

    # Shared colorbar: place it in a dedicated axes on the right.
    if im is not None:
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_TICK)
    else:
        fig.tight_layout()
    _save_fig(fig, out_dir, "fig2_realizations")
    plt.close(fig)


# ============================================================================
# Figure 3 – Detail-band decomposition
# ============================================================================

def plot_detail_bands(
    obs_details: Dict[int, NDArray],
    gen_details: Dict[int, NDArray],
    resolution: int,
    out_dir: Path,
    gen_idx: int = 0,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """Detail-band images (observed vs generated).

    **What it shows.**  Side-by-side detail-band images showing spatial
    structure at each scale separation.  Top row: observed, bottom row:
    generated.  Each column is a detail band
    d_l(x) = u_{H_l} - u_{H_{l+1}}.

    **Diagnostic value.**  Reveals whether the generator captures the
    correct spatial texture at every inter-scale resolution.  Differences
    in spottiness, anisotropy, or amplitude are immediately visible.

    **Pass / fail.**  Visual: the two rows should look qualitatively
    similar in amplitude, characteristic blob size, and directionality.
    """
    bands = sorted(set(obs_details.keys()) & set(gen_details.keys()))
    n_bands = len(bands)

    fig, axes = plt.subplots(2, n_bands, figsize=(FIG_WIDTH, 2 * FIELD_ROW_HEIGHT))
    if n_bands == 1:
        axes = axes[:, None]

    for col, band in enumerate(bands):
        obs_2d = obs_details[band][0].reshape(resolution, resolution)
        gen_2d = gen_details[band][gen_idx].reshape(resolution, resolution)
        vmax = max(abs(obs_2d).max(), abs(gen_2d).max())
        label = _band_label(band, H_schedule)

        axes[0, col].imshow(obs_2d, origin="lower", cmap=CMAP_FIELD,
                            vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"Obs {label}", fontsize=FONT_TITLE)
        axes[0, col].axis("off")

        axes[1, col].imshow(gen_2d, origin="lower", cmap=CMAP_FIELD,
                            vmin=-vmax, vmax=vmax)
        axes[1, col].set_title(f"Gen {label}", fontsize=FONT_TITLE)
        axes[1, col].axis("off")

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig3_detail_bands")
    plt.close(fig)


# ============================================================================
# Figure 4 – One-point PDFs
# ============================================================================

def plot_pdfs(
    obs_details: Dict[int, NDArray],
    gen_details: Dict[int, NDArray],
    out_dir: Path,
    n_cols: int = N_COLS,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """One-point PDFs of detail-band values (observed vs generated).

    **What it shows.**  Histogram-based density curves (with light
    smoothing for presentation) of pixel values in each detail band,
    overlaid for the observed and generated ensembles (following
    Tran et al., Figs. 13--14 in spirit).

    **Diagnostic value.**  The most direct test of one-point accuracy.
    Tests whether the generator reproduces the correct marginal statistics
    of inter-scale fluctuations.

    **Pass / fail.**  Density curves should overlap closely; systematic
    shift, scale mismatch, or heavy-tail discrepancies indicate first-order
    errors.  Complement with the QQ plot (Fig 5) for tail sensitivity.
    """
    bands = sorted(set(obs_details.keys()) & set(gen_details.keys()))
    n_bands = len(bands)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    rng = np.random.default_rng(0)

    for i, band in enumerate(bands):
        ax = axes[i // n_cols, i % n_cols]

        x, y_obs, y_gen = _density_pair_curve(
            obs_details[band],
            gen_details[band],
            rng,
        )
        markevery = max(1, int(len(x) // 10))
        ax.plot(
            x, y_obs,
            color=C_OBS, lw=1.4, label="Obs",
            marker="o", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_obs, alpha=0.14, color=C_OBS)
        ax.plot(
            x, y_gen,
            color=C_GEN, lw=1.4, label="Gen",
            marker="^", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_gen, alpha=0.14, color=C_GEN)

        ax.set_title(_band_label(band, H_schedule), fontsize=FONT_TITLE)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    # Hide unused axes.
    for i in range(n_bands, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig4_pdfs")
    plt.close(fig)


# ============================================================================
# Figure 5 – QQ plots
# ============================================================================

def plot_qq(
    first_order_results: Dict[int, Dict],
    out_dir: Path,
    n_cols: int = N_COLS,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """Quantile-quantile plots per detail band.

    **What it shows.**  QQ diagnostic comparing the tails and body of the
    observed vs generated detail-value distributions.

    **Diagnostic value.**  More sensitive than histograms in the
    distribution tails.  Departure from the 45-degree line reveals
    systematic over- or under-dispersion at specific quantile levels.

    **Pass / fail.**  Points should lie close to the y = x reference line
    across the full quantile range.  S-shaped or hockey-stick deviations
    indicate tail mismatch.
    """
    bands = sorted(first_order_results.keys())
    n_bands = len(bands)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH, SUBPLOT_HEIGHT_SQ * n_rows),
    )
    axes = np.atleast_2d(axes)

    for i, band in enumerate(bands):
        ax = axes[i // n_cols, i % n_cols]
        obs_q = first_order_results[band]["qq_obs"]
        gen_q = first_order_results[band]["qq_gen"]

        ax.scatter(obs_q, gen_q, s=6, alpha=0.6, color=C_GEN)
        lims = [min(obs_q.min(), gen_q.min()), max(obs_q.max(), gen_q.max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Obs quantiles", fontsize=FONT_LABEL)
        ax.set_ylabel("Gen quantiles", fontsize=FONT_LABEL)
        ax.set_title(_band_label(band, H_schedule), fontsize=FONT_TITLE)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_bands, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig5_qq")
    plt.close(fig)


# ============================================================================
# Figure 6 – Directional correlation R(tau)  [Critical figure]
# ============================================================================

def plot_directional_correlation(
    second_order_results: Dict[int, Dict],
    pixel_size: float,
    out_dir: Path,
    n_cols: int = N_COLS,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """Directional normalised correlation R(tau) per detail band.

    **What it shows.**  R(tau e_1) and R(tau e_2) for each detail band,
    comparing the observed curve (solid/dashed blue) against the generated
    ensemble mean +/- one standard deviation (red with shaded envelope).

    **Diagnostic value.**  The core second-order diagnostic from Tran et
    al.  Tests whether spatial two-point structure is reproduced: correct
    correlation length, decay shape, and isotropy of the inter-scale
    detail fluctuations.

    **Pass / fail.**  The observed blue curves should lie within (or very
    close to) the red +/- sigma envelope.  The normalised J value printed
    in each title quantifies the integrated mismatch; J_norm < 0.05 is
    excellent, < 0.10 acceptable.
    """
    bands = sorted(second_order_results.keys())
    n_bands = len(bands)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for i, band in enumerate(bands):
        ax = axes[i // n_cols, i % n_cols]
        res = second_order_results[band]

        lags = res["gen_correlation"]["lags_pixels"].astype(np.float64) * pixel_size
        r_max = res["J"]["r_max_phys"]
        max_lag_idx = min(len(lags), int(r_max / pixel_size) + 10)
        lags = lags[:max_lag_idx]

        # Observed.
        ax.plot(lags, res["R_obs_e1"][:max_lag_idx], "-",
                color=C_OBS, lw=1.2, label="Obs $e_1$")
        ax.plot(lags, res["R_obs_e2"][:max_lag_idx], "--",
                color=C_OBS, lw=1.2, label="Obs $e_2$")

        # Generated mean +/- sigma.
        gc = res["gen_correlation"]
        m1 = gc["R_e1_mean"][:max_lag_idx]
        s1 = gc["R_e1_std"][:max_lag_idx]
        m2 = gc["R_e2_mean"][:max_lag_idx]
        s2 = gc["R_e2_std"][:max_lag_idx]

        ax.plot(lags, m1, "-", color=C_GEN, lw=1.2, label="Gen $e_1$")
        ax.fill_between(lags, m1 - s1, m1 + s1, color=C_FILL, alpha=0.25)

        ax.plot(lags, m2, "--", color=C_GEN, lw=1.2, label="Gen $e_2$")
        ax.fill_between(lags, m2 - s2, m2 + s2, color=C_FILL, alpha=0.15)

        ax.axhline(1 / np.e, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="grey", ls="-", lw=0.5, alpha=0.3)

        J_val = res["J"]["J_normalised"]
        label = _band_label(band, H_schedule)
        ax.set_title(
            f"{label}  ($J_{{\\mathrm{{norm}}}}$={J_val:.4f})",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel("$\\tau / D$", fontsize=FONT_LABEL)
        ax.set_ylabel("$R(\\tau)$", fontsize=FONT_LABEL)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8, loc="upper right")
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_bands, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig6_correlation")
    plt.close(fig)


# ============================================================================
# Figure 7 – J mismatch bar chart
# ============================================================================

def plot_J_bars(
    second_order_results: Dict[int, Dict],
    out_dir: Path,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """Tran J mismatch summary across detail bands.

    **What it shows.**  Bar chart of J_{e_1}, J_{e_2}, and the normalised
    aggregate J_norm for each detail band.

    **Diagnostic value.**  Provides a single-glance summary of the overall
    correlation error across all bands.  Bands with high J indicate scales
    where the generator fails to reproduce the spatial two-point structure.

    **Pass / fail.**  All bars should be small (J_norm < 0.05 excellent,
    < 0.10 acceptable).  A single outlier band warrants investigation of
    the corresponding R(tau) curve (Fig 6).
    """
    bands = sorted(second_order_results.keys())
    J_vals = [second_order_results[b]["J"]["J_normalised"] for b in bands]
    J_e1 = [second_order_results[b]["J"]["J_e1"] for b in bands]
    J_e2 = [second_order_results[b]["J"]["J_e2"] for b in bands]

    x = np.arange(len(bands))
    width = 0.25

    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 2, SUBPLOT_HEIGHT))
    ax.bar(x - width, J_e1, width, label="$J_{e_1}$", color=C_OBS, alpha=0.7)
    ax.bar(x, J_e2, width, label="$J_{e_2}$", color=C_GEN, alpha=0.7)
    ax.bar(x + width, J_vals, width, label="$J_{\\mathrm{norm}}$",
           color="grey", alpha=0.7)

    ax.set_xticks(x)
    labels = [_band_label(b, H_schedule) for b in bands]
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Mismatch", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
    ax.grid(axis="y", alpha=0.2)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig7_J_bars")
    plt.close(fig)


# ============================================================================
# Figure 8 – PSD curves
# ============================================================================

def plot_psd(
    spectral_results: Dict[int, Dict],
    out_dir: Path,
    n_cols: int = N_COLS,
    H_schedule: Optional[List[float]] = None,
) -> None:
    """Radially averaged power spectral density per detail band.

    **What it shows.**  Log-log PSD curves for observed and generated
    ensembles at each detail band, with the generated +/- sigma envelope.

    **Diagnostic value.**  Frequency-domain view of spatial structure per
    detail band.  Reveals whether the generator captures the correct
    spectral slope and any characteristic frequencies.

    **Pass / fail.**  Obs and gen PSD curves should overlap across the
    full wavenumber range.  Persistent divergence at high-k signals
    insufficient micro-structure generation.
    """
    bands = sorted(spectral_results.keys())
    n_bands = len(bands)
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for i, band in enumerate(bands):
        ax = axes[i // n_cols, i % n_cols]
        res = spectral_results[band]

        k = res["obs_psd"]["k_bins"]
        valid = k > 1e-12

        ax.loglog(k[valid], res["obs_psd"]["psd_mean"][valid],
                  color=C_OBS, lw=1.2, label="Obs")
        ax.loglog(k[valid], res["gen_psd"]["psd_mean"][valid],
                  color=C_GEN, lw=1.2, label="Gen")

        gen_lo = res["gen_psd"]["psd_mean"] - res["gen_psd"]["psd_std"]
        gen_hi = res["gen_psd"]["psd_mean"] + res["gen_psd"]["psd_std"]
        ax.fill_between(k[valid],
                        np.maximum(gen_lo[valid], 1e-30),
                        gen_hi[valid],
                        color=C_FILL, alpha=0.2)

        delta = res["psd_mismatch"]
        label = _band_label(band, H_schedule)
        ax.set_title(
            f"{label}  ($\\Delta_{{\\mathrm{{PSD}}}}$={delta:.3f})",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel("$k$", fontsize=FONT_LABEL)
        ax.set_ylabel("$S(k)$", fontsize=FONT_LABEL)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_bands, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig8_psd")
    plt.close(fig)


# ============================================================================
# Figure 9 – Diversity
# ============================================================================

def plot_diversity(
    diversity_results: Dict,
    out_dir: Path,
) -> None:
    """Inter-realisation diversity diagnostics.

    **What it shows.**  Left: histogram of pairwise normalised L2 distances
    between generated realisations at the microscale.  Right: per-band
    diversity ratios (generated / GT).

    **Diagnostic value.**  Detects mode collapse.  If the generator
    produces near-identical outputs, pairwise distances cluster near zero
    and diversity ratios fall well below 1.

    **Pass / fail.**  Diversity ratios should be in [0.5, 2.0] (green).
    Ratios below 0.5 (red) signal possible mode collapse; ratios above 2.0
    signal excessive noise.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))

    # Microscale pairwise distances.
    ax = axes[0]
    dists = diversity_results["microscale"]["distances"]
    if len(dists) > 0:
        ax.hist(dists, bins=30, edgecolor="k", alpha=0.7, color=C_GEN)
        ax.axvline(diversity_results["microscale"]["mean"],
                   color="r", ls="--", lw=1.5,
                   label=f"mean={diversity_results['microscale']['mean']:.4f}")
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
    ax.set_xlabel("Normalised $L_2$ distance", fontsize=FONT_LABEL)
    ax.set_title("Microscale inter-realisation distances", fontsize=FONT_TITLE)
    ax.grid(alpha=0.2)
    _set_tick_fontsize(ax)

    # Per-band diversity ratios.
    ax = axes[1]
    per_band = diversity_results.get("per_band", {})
    if per_band:
        bands = sorted(per_band.keys())
        ratios = [per_band[b].get("diversity_ratio", float("nan")) for b in bands]
        colors = ["green" if 0.5 <= r <= 2.0 else "red" for r in ratios]
        ax.bar(range(len(bands)), ratios, color=colors, alpha=0.7, edgecolor="k")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.axhline(0.5, color="orange", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(2.0, color="orange", ls=":", lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([f"Band {b}" for b in bands], fontsize=FONT_TICK)
        ax.set_ylabel("Diversity ratio (gen / GT)", fontsize=FONT_LABEL)
        ax.set_title("Detail-band diversity ratios", fontsize=FONT_TITLE)
        ax.grid(axis="y", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No GT comparison", ha="center", va="center",
                transform=ax.transAxes)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig9_diversity")
    plt.close(fig)


# ============================================================================
# Figure 10 – Direct-field PDFs (NEW)
# ============================================================================

def plot_direct_field_pdfs(
    eval_gt_fields: Dict[int, NDArray],
    gen_filtered_fields: Dict[int, NDArray],
    eval_H_schedule: List[float],
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """One-point PDFs of the direct filtered fields (not detail-band residuals).

    **What it shows.**  Histogram-based density curves (with light
    smoothing for presentation) of GT ensemble pixel values vs generated
    ensemble pixel values at each evaluation scale. One subplot per scale.

    **Diagnostic value.**  Shows whether the generator reproduces the
    correct marginal distribution at each spatial scale independently,
    without the differencing step used in detail-band analysis (Fig 4).
    A complement that catches errors in the absolute field distributions.

    **Pass / fail.**  Density curves should overlap closely at every scale.
    Systematic shift or variance mismatch at a particular scale indicates
    the generator's filtered output drifts from the GT distribution there.
    """
    scales = sorted(set(eval_gt_fields.keys()) & set(gen_filtered_fields.keys()))
    n_scales = len(scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    rng = np.random.default_rng(0)

    for i, scale in enumerate(scales):
        ax = axes[i // n_cols, i % n_cols]

        x, y_obs, y_gen = _density_pair_curve(
            eval_gt_fields[scale],
            gen_filtered_fields[scale],
            rng,
        )
        markevery = max(1, int(len(x) // 10))
        ax.plot(
            x, y_obs,
            color=C_OBS, lw=1.4, label="Obs",
            marker="o", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_obs, alpha=0.14, color=C_OBS)
        ax.plot(
            x, y_gen,
            color=C_GEN, lw=1.4, label="Gen",
            marker="^", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_gen, alpha=0.14, color=C_GEN)

        H_val = eval_H_schedule[scale] if scale < len(eval_H_schedule) else scale
        ax.set_title(f"$H = {H_val}$", fontsize=FONT_TITLE)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_scales, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig10_direct_field_pdfs")
    plt.close(fig)


# ============================================================================
# Figure 11 – Direct-field directional correlation (NEW)
# ============================================================================

def plot_direct_field_correlation(
    eval_gt_fields: Dict[int, NDArray],
    gen_filtered_fields: Dict[int, NDArray],
    eval_H_schedule: List[float],
    resolution: int,
    pixel_size: float,
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """Directional normalised correlation R(tau) of the direct filtered fields.

    **What it shows.**  R(tau e_1) and R(tau e_2) for the filtered fields
    at each evaluation scale, comparing the observed ensemble-mean curve
    (solid/dashed blue) against the generated ensemble mean +/- one
    standard deviation (red with shaded envelope).  One subplot per scale.

    **Diagnostic value.**  Complements the detail-band correlation (Fig 6)
    by showing two-point structure of the fields themselves at each scale,
    rather than the inter-scale residuals.  Useful for detecting scale-
    specific correlation-length errors that may cancel in the detail
    decomposition.

    **Pass / fail.**  Obs curves should fall within the gen +/- sigma
    envelope across the full lag range, analogous to Fig 6.
    """
    from scripts.fae.tran_evaluation.second_order import (
        directional_correlation,
        ensemble_directional_correlation,
    )

    scales = sorted(set(eval_gt_fields.keys()) & set(gen_filtered_fields.keys()))
    n_scales = len(scales)
    n_rows = (n_scales + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for i, scale in enumerate(scales):
        ax = axes[i // n_cols, i % n_cols]

        # Observed: correlation of ensemble-mean field.
        obs_mean_2d = np.mean(
            eval_gt_fields[scale], axis=0,
        ).reshape(resolution, resolution)
        R_obs_e1, R_obs_e2 = directional_correlation(obs_mean_2d)

        # Generated: ensemble statistics.
        gen_corr = ensemble_directional_correlation(
            gen_filtered_fields[scale], resolution,
        )

        max_lag_idx = resolution // 2
        lags = np.arange(max_lag_idx).astype(np.float64) * pixel_size

        # Observed.
        ax.plot(lags, R_obs_e1[:max_lag_idx], "-",
                color=C_OBS, lw=1.2, label="Obs $e_1$")
        ax.plot(lags, R_obs_e2[:max_lag_idx], "--",
                color=C_OBS, lw=1.2, label="Obs $e_2$")

        # Generated mean +/- sigma.
        m1 = gen_corr["R_e1_mean"][:max_lag_idx]
        s1 = gen_corr["R_e1_std"][:max_lag_idx]
        m2 = gen_corr["R_e2_mean"][:max_lag_idx]
        s2 = gen_corr["R_e2_std"][:max_lag_idx]

        ax.plot(lags, m1, "-", color=C_GEN, lw=1.2, label="Gen $e_1$")
        ax.fill_between(lags, m1 - s1, m1 + s1, color=C_FILL, alpha=0.25)

        ax.plot(lags, m2, "--", color=C_GEN, lw=1.2, label="Gen $e_2$")
        ax.fill_between(lags, m2 - s2, m2 + s2, color=C_FILL, alpha=0.15)

        ax.axhline(1 / np.e, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="grey", ls="-", lw=0.5, alpha=0.3)

        H_val = eval_H_schedule[scale] if scale < len(eval_H_schedule) else scale
        ax.set_title(f"$H = {H_val}$", fontsize=FONT_TITLE)
        ax.set_xlabel("$\\tau / D$", fontsize=FONT_LABEL)
        ax.set_ylabel("$R(\\tau)$", fontsize=FONT_LABEL)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8, loc="upper right")
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_scales, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig11_direct_field_correlation")
    plt.close(fig)


# ============================================================================
# Summary table (text)
# ============================================================================

def print_summary_table(
    cond_results: Dict,
    first_order_results: Dict[int, Dict],
    second_order_results: Dict[int, Dict],
    spectral_results: Dict[int, Dict],
    diversity_results: Dict,
) -> str:
    """Build and return a formatted summary table string."""
    lines: List[str] = []
    sep = "=" * 80
    lines.append(sep)
    lines.append("TRAN-ALIGNED EVALUATION SUMMARY")
    lines.append(sep)

    # Conditioning.
    lines.append(f"\n1. CONDITIONING CONSISTENCY")
    lines.append(f"   Mean E^coarse: {cond_results['mean']:.6f}")
    lines.append(f"   Median:        {cond_results['median']:.6f}")
    passed = cond_results["mean"] < 0.05
    lines.append(f"   Pass (<5%):    {'YES' if passed else 'NO'}")

    # First-order.
    bands = sorted(first_order_results.keys())
    lines.append(f"\n2. FIRST-ORDER STATISTICS (W1)")
    lines.append(f"   {'Band':>6} | {'W1':>10} | {'W1_norm':>10} | {'Mean_RE':>10} | {'Var_RE':>10}")
    lines.append(f"   {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for b in bands:
        w = first_order_results[b]["wasserstein1"]
        m = first_order_results[b]["moments"]["relative_error"]
        lines.append(
            f"   {b:>6} | {w['w1']:>10.6f} | {w['w1_normalised']:>10.6f} | "
            f"{m['mean']:>10.6f} | {m['variance']:>10.6f}"
        )

    # Second-order.
    lines.append(f"\n3. SECOND-ORDER STATISTICS (R(tau), J)")
    lines.append(
        f"   {'Band':>6} | {'J_norm':>10} | {'J_e1':>10} | {'J_e2':>10} | "
        f"{'xi_obs_e1':>10} | {'xi_gen_e1':>10} | {'iso_obs':>8} | {'iso_gen':>8}"
    )
    lines.append(f"   {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for b in bands:
        if b not in second_order_results:
            continue
        J = second_order_results[b]["J"]
        cl = second_order_results[b]["correlation_lengths"]
        iso = second_order_results[b]["isotropy"]
        lines.append(
            f"   {b:>6} | {J['J_normalised']:>10.6f} | {J['J_e1']:>10.6f} | "
            f"{J['J_e2']:>10.6f} | {cl['obs_e1']['xi_e']:>10.4f} | "
            f"{cl['gen_e1']['xi_e']:>10.4f} | "
            f"{'Y' if iso['obs']['is_isotropic'] else 'N':>8} | "
            f"{'Y' if iso['gen']['is_isotropic'] else 'N':>8}"
        )

    # Spectral.
    lines.append(f"\n4. SPECTRAL (PSD)")
    lines.append(f"   {'Band':>6} | {'dPSD':>10} | {'lam_obs':>10} | {'lam_gen':>10}")
    lines.append(f"   {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for b in bands:
        if b not in spectral_results:
            continue
        s = spectral_results[b]
        lines.append(
            f"   {b:>6} | {s['psd_mismatch']:>10.4f} | "
            f"{s['wavelength_obs']:>10.4f} | {s['wavelength_gen']:>10.4f}"
        )

    # Diversity.
    lines.append(f"\n5. DIVERSITY")
    micro = diversity_results["microscale"]
    lines.append(f"   Microscale mean dist:  {micro['mean']:.6f}")
    lines.append(f"   Microscale CV:         {micro['cv']:.6f}")
    if "diversity_ratio" in micro:
        lines.append(f"   Microscale div ratio:  {micro['diversity_ratio']:.4f}")

    per_band = diversity_results.get("per_band", {})
    for b in sorted(per_band.keys()):
        pb = per_band[b]
        collapse = pb.get("mode_collapse", False)
        dr = pb.get("diversity_ratio", float("nan"))
        lines.append(
            f"   Band {b}: CV={pb['cv']:.4f}, div_ratio={dr:.4f}, "
            f"collapse={'YES' if collapse else 'no'}"
        )

    lines.append(sep)
    text = "\n".join(lines)
    return text


# ============================================================================
# Figure 12 – Trajectory fields (backward SDE marginals at each knot time)
# ============================================================================

def plot_trajectory_fields(
    trajectory_fields: NDArray,
    gt_fields_by_index: Dict[int, NDArray],
    time_indices: NDArray,
    full_H_schedule: List[float],
    resolution: int,
    out_dir: Path,
) -> None:
    """Backward SDE trajectory fields at each knot time vs GT.

    **What it shows.**  Top row: ground-truth fields at each MSBM scale
    (one per knot time).  Bottom row: a single generated realisation at
    each knot time.  Columns are ordered from finest (left) to coarsest
    (right).

    **Diagnostic value.**  Visual check that the backward SDE produces
    fields with the correct spatial texture, value range, and smoothness
    at every intermediate scale — not just the final (finest) output.

    **Pass / fail.**  Visual: the two rows should look qualitatively
    similar at each scale.  The generated field should become
    progressively smoother from left to right, matching GT.
    """
    T_knots = trajectory_fields.shape[0]
    sample_idx = 0  # Show first GT sample and first gen realisation.

    fig, axes = plt.subplots(2, T_knots, figsize=(FIG_WIDTH, 2 * FIELD_ROW_HEIGHT))
    if T_knots == 1:
        axes = axes[:, None]

    for k in range(T_knots):
        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx

        gt_field = gt_fields_by_index[ds_idx][sample_idx]
        gen_field = trajectory_fields[k, 0]  # First realisation.

        all_vals = np.concatenate([gt_field.ravel(), gen_field.ravel()])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())

        ax_gt = axes[0, k]
        ax_gt.imshow(gt_field.reshape(resolution, resolution),
                     origin="lower", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"GT $H={H_val}$", fontsize=FONT_TITLE)
        ax_gt.axis("off")

        ax_gen = axes[1, k]
        im = ax_gen.imshow(gen_field.reshape(resolution, resolution),
                           origin="lower", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax_gen.set_title(f"Gen $H={H_val}$", fontsize=FONT_TITLE)
        ax_gen.axis("off")

    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    _save_fig(fig, out_dir, "fig12_trajectory_fields")
    plt.close(fig)


# ============================================================================
# Figure 13 – Trajectory PDFs (one-point distributions at each knot scale)
# ============================================================================

def plot_trajectory_pdfs(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """One-point PDFs of backward SDE trajectory fields vs GT at each knot scale.

    **What it shows.**  Histogram-based density curves (with light
    smoothing for presentation) of GT ensemble pixel values vs generated
    trajectory-field pixel values at each MSBM knot time (following
    Tran et al., Figs. 13--14 in spirit).

    **Diagnostic value.**  Tests whether the backward SDE produces fields
    with the correct marginal distribution at each intermediate scale.
    Unlike Figs 10--11, these are the SDE's *native* marginals, not
    post-hoc filtered versions of the final field.

    **Pass / fail.**  Density curves should overlap closely at every scale.
    Discrepancies reveal where in the backward SDE trajectory the
    distribution deviates from the target.
    """
    knots = sorted(trajectory_results.keys())
    n_knots = len(knots)
    n_rows = (n_knots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    rng = np.random.default_rng(0)

    for i, k in enumerate(knots):
        ax = axes[i // n_cols, i % n_cols]
        res = trajectory_results[k]

        x, y_obs, y_gen = _density_pair_curve(
            res["_obs_fields"],
            res["_gen_fields"],
            rng,
        )
        markevery = max(1, int(len(x) // 10))
        ax.plot(
            x, y_obs,
            color=C_OBS, lw=1.4, label="Obs",
            marker="o", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_obs, alpha=0.14, color=C_OBS)
        ax.plot(
            x, y_gen,
            color=C_GEN, lw=1.4, label="Gen",
            marker="^", markersize=2.2, markevery=markevery,
        )
        ax.fill_between(x, y_gen, alpha=0.14, color=C_GEN)

        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        w1_norm = res["wasserstein1"]["w1_normalised"]
        ax.set_title(f"$H={H_val}$  ($W_1^{{\\mathrm{{norm}}}}$={w1_norm:.4f})",
                     fontsize=FONT_TITLE)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_knots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig13_trajectory_pdfs")
    plt.close(fig)


# ============================================================================
# Figure 14 – Trajectory directional correlation R(tau)
# ============================================================================

def plot_trajectory_correlation(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
    pixel_size: float,
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """Directional R(tau) of backward SDE trajectory fields vs GT at each knot scale.

    **What it shows.**  R(tau e_1) and R(tau e_2) for the backward SDE
    trajectory fields at each MSBM knot time, comparing the observed
    ensemble-mean curve against the generated ensemble mean +/- sigma.

    **Diagnostic value.**  Tests whether the SDE produces fields with the
    correct spatial two-point structure at each intermediate scale.
    Unlike Fig 6, these are native SDE marginals, not inter-scale
    residuals.  Unlike Fig 11, these are not post-hoc filtered fields.

    **Pass / fail.**  Obs curves should fall within the gen +/- sigma
    envelope.  The J_norm value in each title quantifies the mismatch.
    """
    knots = sorted(trajectory_results.keys())
    n_knots = len(knots)
    n_rows = (n_knots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for i, k in enumerate(knots):
        ax = axes[i // n_cols, i % n_cols]
        res = trajectory_results[k]

        gc = res["gen_correlation"]
        max_lag_idx = len(gc["lags_pixels"]) // 2
        lags = gc["lags_pixels"][:max_lag_idx].astype(np.float64) * pixel_size

        # Observed.
        ax.plot(lags, res["R_obs_e1"][:max_lag_idx], "-",
                color=C_OBS, lw=1.2, label="Obs $e_1$")
        ax.plot(lags, res["R_obs_e2"][:max_lag_idx], "--",
                color=C_OBS, lw=1.2, label="Obs $e_2$")

        # Generated mean +/- sigma.
        m1 = gc["R_e1_mean"][:max_lag_idx]
        s1 = gc["R_e1_std"][:max_lag_idx]
        m2 = gc["R_e2_mean"][:max_lag_idx]
        s2 = gc["R_e2_std"][:max_lag_idx]

        ax.plot(lags, m1, "-", color=C_GEN, lw=1.2, label="Gen $e_1$")
        ax.fill_between(lags, m1 - s1, m1 + s1, color=C_FILL, alpha=0.25)

        ax.plot(lags, m2, "--", color=C_GEN, lw=1.2, label="Gen $e_2$")
        ax.fill_between(lags, m2 - s2, m2 + s2, color=C_FILL, alpha=0.15)

        ax.axhline(1 / np.e, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="grey", ls="-", lw=0.5, alpha=0.3)

        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        J_val = res["J"]["J_normalised"]
        ax.set_title(
            f"$H={H_val}$  ($J_{{\\mathrm{{norm}}}}$={J_val:.4f})",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel("$\\tau / D$", fontsize=FONT_LABEL)
        ax.set_ylabel("$R(\\tau)$", fontsize=FONT_LABEL)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8, loc="upper right")
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_knots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig14_trajectory_correlation")
    plt.close(fig)


# ============================================================================
# Figure 17 – Superposed trajectory directional correlation across H
# ============================================================================

def plot_trajectory_correlation_superposed(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
    pixel_size: float,
    out_dir: Path,
) -> None:
    """Superposed trajectory correlations across all knot scales.

    **What it shows.**  Two panels (e1 and e2), each superposing all knot-scale
    correlation curves so shifts across H are directly visible.

    **Styling.**  Curve shade encodes H (lighter -> smaller H, darker -> larger H).
    Generated mean curves use dashed lines; generated +/- sigma envelopes use
    a separate red-shaded family.
    """
    knots = sorted(trajectory_results.keys())
    if len(knots) == 0:
        return

    # Sort by physical H to make shading monotonic with scale.
    items: List[Tuple[float, int, Dict]] = []
    for k in knots:
        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else float(ds_idx)
        items.append((float(H_val), k, trajectory_results[k]))
    items.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT))
    ax_e1, ax_e2 = axes

    n_items = len(items)
    h_handles: List[Line2D] = []
    cmap_h = plt.get_cmap("cividis")
    cmap_band = plt.get_cmap("Reds")

    for rank, (H_val, _k, res) in enumerate(items):
        t = rank / max(n_items - 1, 1)
        line_color = cmap_h(0.25 + 0.65 * t)
        band_color = cmap_band(0.25 + 0.60 * t)
        band_alpha = 0.08 + 0.10 * t

        gc = res["gen_correlation"]
        max_lag_idx = len(gc["lags_pixels"]) // 2
        lags = gc["lags_pixels"][:max_lag_idx].astype(np.float64) * pixel_size

        # e1
        ax_e1.plot(lags, res["R_obs_e1"][:max_lag_idx], "-",
                   color=line_color, lw=1.15)
        m1 = gc["R_e1_mean"][:max_lag_idx]
        s1 = gc["R_e1_std"][:max_lag_idx]
        ax_e1.plot(lags, m1, "--", color=line_color, lw=1.15)
        ax_e1.fill_between(lags, m1 - s1, m1 + s1, color=band_color, alpha=band_alpha)

        # e2
        ax_e2.plot(lags, res["R_obs_e2"][:max_lag_idx], "-",
                   color=line_color, lw=1.15)
        m2 = gc["R_e2_mean"][:max_lag_idx]
        s2 = gc["R_e2_std"][:max_lag_idx]
        ax_e2.plot(lags, m2, "--", color=line_color, lw=1.15)
        ax_e2.fill_between(lags, m2 - s2, m2 + s2, color=band_color, alpha=band_alpha)

        h_handles.append(Line2D([0], [0], color=line_color, lw=1.4, label=f"H={H_val:.3g}"))

    for ax, direction in ((ax_e1, "$e_1$"), (ax_e2, "$e_2$")):
        ax.axhline(1 / np.e, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="grey", ls="-", lw=0.5, alpha=0.3)
        ax.set_title(f"Superposed $R(\\tau {direction})$", fontsize=FONT_TITLE)
        ax.set_xlabel("$\\tau / D$", fontsize=FONT_LABEL)
        ax.set_ylabel("$R(\\tau)$", fontsize=FONT_LABEL)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    style_handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="-", label="Obs"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="Gen mean"),
    ]
    leg_style = ax_e1.legend(
        handles=style_handles,
        loc="upper right",
        fontsize=FONT_LEGEND,
        framealpha=0.85,
        title="Curve type",
        title_fontsize=FONT_LEGEND,
    )
    ax_e1.add_artist(leg_style)
    ax_e1.legend(
        handles=h_handles,
        loc="lower left",
        fontsize=FONT_LEGEND,
        framealpha=0.85,
        title="Scale H (shade)",
        title_fontsize=FONT_LEGEND,
    )

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig17_trajectory_correlation_superposed")
    plt.close(fig)


# ============================================================================
# Figure 15 – Trajectory PSD
# ============================================================================

def plot_trajectory_psd(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """Radially averaged PSD of backward SDE trajectory fields vs GT at each knot scale.

    **What it shows.**  Log-log PSD curves for observed and generated
    ensembles at each MSBM knot time, with the generated +/- sigma
    envelope.

    **Diagnostic value.**  Frequency-domain view of the SDE's native
    marginal fields at each scale.  Reveals whether the SDE produces the
    correct spectral content at each intermediate time.

    **Pass / fail.**  Obs and gen PSD curves should overlap across the
    full wavenumber range at every knot time.
    """
    knots = sorted(trajectory_results.keys())
    n_knots = len(knots)
    n_rows = (n_knots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT * n_rows))
    axes = np.atleast_2d(axes)

    for i, k in enumerate(knots):
        ax = axes[i // n_cols, i % n_cols]
        res = trajectory_results[k]

        k_bins = res["obs_psd"]["k_bins"]
        valid = k_bins > 1e-12

        ax.loglog(k_bins[valid], res["obs_psd"]["psd_mean"][valid],
                  color=C_OBS, lw=1.2, label="Obs")
        ax.loglog(k_bins[valid], res["gen_psd"]["psd_mean"][valid],
                  color=C_GEN, lw=1.2, label="Gen")

        gen_lo = res["gen_psd"]["psd_mean"] - res["gen_psd"]["psd_std"]
        gen_hi = res["gen_psd"]["psd_mean"] + res["gen_psd"]["psd_std"]
        ax.fill_between(k_bins[valid],
                        np.maximum(gen_lo[valid], 1e-30),
                        gen_hi[valid],
                        color=C_FILL, alpha=0.2)

        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        delta = res["psd_mismatch"]
        ax.set_title(
            f"$H={H_val}$  ($\\Delta_{{\\mathrm{{PSD}}}}$={delta:.3f})",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel("$k$", fontsize=FONT_LABEL)
        ax.set_ylabel("$S(k)$", fontsize=FONT_LABEL)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_knots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig15_trajectory_psd")
    plt.close(fig)


# ============================================================================
# Figure 16 – Trajectory QQ plots
# ============================================================================

def plot_trajectory_qq(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
    out_dir: Path,
    n_cols: int = N_COLS,
) -> None:
    """Quantile-quantile plots for backward SDE trajectory fields at each knot scale.

    **What it shows.**  QQ diagnostic comparing tails and body of the
    observed vs generated trajectory-field distributions at each scale.

    **Diagnostic value.**  More sensitive than histograms in the tails.
    Departure from the y=x line reveals where the SDE marginal
    distribution deviates systematically from the target.

    **Pass / fail.**  Points should lie close to the y=x reference line.
    """
    knots = sorted(trajectory_results.keys())
    n_knots = len(knots)
    n_rows = (n_knots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH, SUBPLOT_HEIGHT_SQ * n_rows),
    )
    axes = np.atleast_2d(axes)

    for i, k in enumerate(knots):
        ax = axes[i // n_cols, i % n_cols]
        obs_q = trajectory_results[k]["qq_obs"]
        gen_q = trajectory_results[k]["qq_gen"]

        ax.scatter(obs_q, gen_q, s=6, alpha=0.6, color=C_GEN)
        lims = [min(obs_q.min(), gen_q.min()), max(obs_q.max(), gen_q.max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)

        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        ax.set_title(f"$H={H_val}$", fontsize=FONT_TITLE)
        ax.set_xlabel("Obs quantiles", fontsize=FONT_LABEL)
        ax.set_ylabel("Gen quantiles", fontsize=FONT_LABEL)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)

    for i in range(n_knots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, out_dir, "fig16_trajectory_qq")
    plt.close(fig)


# ============================================================================
# Trajectory summary table (text)
# ============================================================================

def print_trajectory_summary_table(
    trajectory_results: Dict[int, Dict],
    time_indices: NDArray,
    full_H_schedule: List[float],
) -> str:
    """Build and return a formatted summary table for trajectory evaluation."""
    lines: List[str] = []
    sep = "=" * 80
    lines.append(sep)
    lines.append("BACKWARD SDE TRAJECTORY EVALUATION SUMMARY")
    lines.append(sep)

    knots = sorted(trajectory_results.keys())

    lines.append(f"\n{'Knot':>6} | {'H':>6} | {'W1_norm':>10} | {'J_norm':>10} | "
                 f"{'dPSD':>10} | {'xi_obs_e1':>10} | {'xi_gen_e1':>10}")
    lines.append(f"{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-"
                 f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for k in knots:
        res = trajectory_results[k]
        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        w1_n = res["wasserstein1"]["w1_normalised"]
        J_n = res["J"]["J_normalised"]
        dpsd = res["psd_mismatch"]
        xi_obs = res["correlation_lengths"]["obs_e1"]["xi_e"]
        xi_gen = res["correlation_lengths"]["gen_e1"]["xi_e"]
        lines.append(
            f"{k:>6} | {H_val:>6.2f} | {w1_n:>10.6f} | {J_n:>10.6f} | "
            f"{dpsd:>10.4f} | {xi_obs:>10.4f} | {xi_gen:>10.4f}"
        )

    lines.append(sep)
    return "\n".join(lines)
