"""General-purpose visualization utilities for MMSFM field and covariance analysis."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from matplotlib import animation, gridspec, pyplot as plt
from matplotlib.figure import Figure

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore

__all__ = [
    "format_for_paper",
    "reconstruct_fields_from_coefficients",
    "plot_field_snapshots",
    "plot_field_evolution_gif",
    "plot_field_statistics",
    "plot_spatial_correlation",
    "plot_sample_comparison_grid",
    "plot_eigenvalue_spectra_comparison",
    "plot_covariance_heatmaps_comparison",
    "visualize_all_field_reconstructions",
    "compute_sample_correlation_matrix_with_eigen",
    "reconstruct_covariance_from_truncated_eigen",
    "compute_correlation_from_covariance",
    "relative_covariance_frobenius_distance",
    "plot_vector_field_2d_projection",
    "plot_vector_field_streamplot",
    "plot_interpolated_probability_paths",
]

_NUMERIC_EPS = 1e-12


def format_for_paper() -> None:
    """Apply publication-style defaults for matplotlib figures."""
    plt.rcParams.update({"image.cmap": "viridis"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "Utopia",
                "ITC Bookman",
                "Bookman",
                "Nimbus Roman No9 L",
                "Palatino",
                "Charter",
                "serif",
            ]
        }
    )
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"mathtext.fontset": "custom"})
    plt.rcParams.update({"mathtext.rm": "serif"})
    plt.rcParams.update({"mathtext.it": "serif:italic"})
    plt.rcParams.update({"mathtext.bf": "serif:bold"})
    plt.close("all")


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_tensor(
    array: Any, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        if dtype is not None and array.dtype != dtype:
            array = array.to(dtype=dtype)
        if device is not None and array.device != device:
            array = array.to(device)
        return array
    return torch.as_tensor(array, dtype=dtype, device=device)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _save_and_log_figure(
    fig: Figure,
    base_path: str,
    run: Any = None,
    wandb_key: str | None = None,
    *,
    dpi: int = 150,
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    outdir = os.path.dirname(base_path)
    _ensure_dir(outdir)

    for ext in formats:
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext.lower() == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(f"{base_path}.{ext}", **save_kwargs)

    if run is not None and wandb is not None and wandb_key is not None:
        run.log({wandb_key: wandb.Image(data_or_path=f"{base_path}.png", mode="RGB")})


def _save_and_log_animation(
    ani: animation.FuncAnimation,
    path: str,
    run: Any = None,
    wandb_key: str | None = None,
    *,
    fps: int = 5,
) -> None:
    outdir = os.path.dirname(path)
    _ensure_dir(outdir)
    ani.save(path, writer="imagemagick", fps=fps)

    if run is not None and wandb is not None and wandb_key is not None:
        run.log({wandb_key: wandb.Video(data_or_path=path, format="gif")})


def _select_time_indices(num_steps: int, target_steps: int) -> np.ndarray:
    if num_steps >= target_steps and target_steps > 1:
        return np.linspace(0, num_steps - 1, target_steps).astype(int)
    return np.arange(min(num_steps, target_steps), dtype=int)


def reconstruct_fields_from_coefficients(
    coeffs: np.ndarray | torch.Tensor,
    pca_info: Mapping[str, Any],
    resolution: int,
) -> np.ndarray:
    """Reconstruct spatial fields from PCA coefficients."""
    coeffs_np = _to_numpy(coeffs)

    mean = _to_numpy(pca_info["mean"]).reshape(-1)
    components = _to_numpy(pca_info["components"])
    eigenvalues = _to_numpy(pca_info["explained_variance"])
    is_whitened = bool(pca_info.get("is_whitened", True))

    if coeffs_np.ndim == 2:
        coeffs_np = coeffs_np[np.newaxis, ...]
        squeeze_time = True
    else:
        squeeze_time = False

    sqrt_eig = np.sqrt(np.maximum(eigenvalues, _NUMERIC_EPS))
    data_dim = resolution * resolution

    fields_per_time = []
    for coeff_block in coeffs_np:
        if is_whitened:
            scaled_coeffs = coeff_block * sqrt_eig[np.newaxis, :]
            reconstructed = scaled_coeffs @ components + mean
        else:
            reconstructed = coeff_block @ components + mean
        fields_2d = reconstructed.reshape(-1, resolution, resolution)
        fields_per_time.append(fields_2d)

    fields = np.asarray(fields_per_time)
    if squeeze_time:
        fields = fields.squeeze(0)
    if fields.shape[-2] * fields.shape[-1] != data_dim:
        raise ValueError("Reconstructed field has inconsistent dimensionality.")
    return fields


def plot_field_snapshots(
    fields: np.ndarray,
    zt: Sequence[float],
    outdir: str | None,
    run: Any,
    n_samples: int = 5,
    score: bool = False,
    cmap: str = "viridis",
    *,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Create static grid of field samples at selected time points."""
    T, N, _, _ = fields.shape
    num_marginals = len(zt)
    n_samples = min(n_samples, N)

    time_indices = _select_time_indices(T, num_marginals if num_marginals > 0 else T)
    num_cols = len(time_indices)

    # Limit figure size to fit letter paper (8.5 x 11 inches)
    col_width = min(3, 8.5 / max(num_cols, 1))
    row_height = min(3, 10.5 / max(n_samples, 1))
    fig = plt.figure(figsize=(col_width * num_cols, row_height * n_samples))
    grid = gridspec.GridSpec(n_samples, num_cols, figure=fig, hspace=0.3, wspace=0.3)

    vmin = np.percentile(fields, 1)
    vmax = np.percentile(fields, 99)

    for row in range(n_samples):
        for col, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(grid[row, col])
            im = ax.imshow(fields[t_idx, row], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                time_val = (
                    zt[col]
                    if col < len(zt)
                    else zt[0] + (zt[-1] - zt[0]) * t_idx / max(T - 1, 1)
                    if len(zt) >= 2
                    else zt[0] if zt else t_idx
                )
                ax.set_title(f"t = {time_val:.2f}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Sample {row + 1}", fontsize=10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"field_snapshots_{diffeq}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)

    if close:
        plt.close(fig)
        return None
    return fig


def plot_field_evolution_gif(
    fields: np.ndarray,
    zt: Sequence[float],
    outdir: str | None,
    run: Any,
    sample_idx: int = 0,
    score: bool = False,
    cmap: str = "viridis",
    fps: int = 5,
    *,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
) -> None:
    """Animate evolution of a single field through time."""
    T, _, _, _ = fields.shape
    t_interp = np.linspace(zt[0], zt[-1], T) if len(zt) >= 2 else np.arange(T)

    vmin = np.percentile(fields[:, sample_idx], 1)
    vmax = np.percentile(fields[:, sample_idx], 99)

    fig, ax = plt.subplots(figsize=(6, 6))  # Single square plot fits easily

    def animate(frame: int) -> list[Any]:
        ax.clear()
        im = ax.imshow(
            fields[frame, sample_idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        time_val = t_interp[frame] if frame < len(t_interp) else frame
        ax.set_title(f"Sample {sample_idx + 1}, t = {time_val:.2f}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return [im]

    ani = animation.FuncAnimation(fig, animate, frames=T, interval=1000 // max(fps, 1), blit=True)

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"field_evolution_{diffeq}_sample{sample_idx}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        path = os.path.join(outdir, f"{base_name}.gif")
        _save_and_log_animation(ani, path, run, wandb_entry, fps=fps)

    plt.close(fig)


def plot_field_statistics(
    fields: np.ndarray,
    zt: Sequence[float],
    testdata_fields: Sequence[np.ndarray],
    outdir: str | None,
    run: Any,
    score: bool = False,
    *,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Compare mean and standard deviation of generated vs. reference fields."""
    T, _, _, _ = fields.shape
    num_reference = len(testdata_fields)
    t_interp = np.linspace(zt[0], zt[-1], T) if len(zt) >= 2 else np.arange(T)

    gen_means = np.mean(fields, axis=(1, 2, 3))
    gen_stds = np.std(fields, axis=(1, 2, 3))

    test_means = np.array([np.mean(ref) for ref in testdata_fields[: len(zt)]])
    test_stds = np.array([np.std(ref) for ref in testdata_fields[: len(zt)]])

    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))  # Fit letter width

    axs[0].plot(t_interp, gen_means, "o-", label="Generated", markersize=6, alpha=0.7)
    axs[0].plot(zt[:num_reference], test_means[:num_reference], "s-", label="Test Data", markersize=8)
    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Mean Field Value", fontsize=12)
    axs[0].set_title("Mean Field Value Over Time", fontsize=14)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_interp, gen_stds, "o-", label="Generated", markersize=6, alpha=0.7)
    axs[1].plot(zt[:num_reference], test_stds[:num_reference], "s-", label="Test Data", markersize=8)
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("Std Field Value", fontsize=12)
    axs[1].set_title("Std Field Value Over Time", fontsize=14)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"field_statistics_{diffeq}"
    wandb_entry = wandb_key or f"evalplots/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)

    if close:
        plt.close(fig)
        return None
    return fig


def plot_spatial_correlation(
    fields: np.ndarray,
    zt: Sequence[float],
    outdir: str | None,
    run: Any,
    score: bool = False,
    *,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Average spatial autocorrelation of generated fields."""
    from scipy.signal import correlate2d  # imported lazily to avoid hard dependency

    T, N, _, _ = fields.shape
    n_times = min(5, T)
    time_indices = np.linspace(0, T - 1, n_times).astype(int) if T > 1 else np.array([0])
    t_interp = np.linspace(zt[0], zt[-1], T) if len(zt) >= 2 else np.arange(T)

    # Fit within letter width
    col_width = min(4, 8.5 / max(n_times, 1))
    fig, axs = plt.subplots(1, n_times, figsize=(col_width * n_times, col_width))
    if n_times == 1:
        axs = np.array([axs])

    for idx, t_idx in enumerate(time_indices):
        autocorrs = []
        for sample_idx in range(min(20, N)):
            field = fields[t_idx, sample_idx]
            field_norm = (field - field.mean()) / (field.std() + _NUMERIC_EPS)
            autocorr = correlate2d(field_norm, field_norm, mode="same")
            autocorr = autocorr / (autocorr.max() + _NUMERIC_EPS)
            autocorrs.append(autocorr)

        avg_autocorr = np.mean(autocorrs, axis=0)
        im = axs[idx].imshow(avg_autocorr, cmap="RdBu_r", vmin=-0.3, vmax=1.0, origin="lower")
        axs[idx].set_title(f"t = {t_interp[t_idx]:.2f}", fontsize=12)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        fig.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)

    fig.suptitle("Spatial Autocorrelation", fontsize=16)
    fig.tight_layout()

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"spatial_correlation_{diffeq}"
    wandb_entry = wandb_key or f"evalplots/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)

    if close:
        plt.close(fig)
        return None
    return fig


def plot_sample_comparison_grid(
    target_fields: Sequence[np.ndarray],
    generated_fields: np.ndarray,
    zt: Sequence[float],
    outdir: str | None,
    run: Any,
    *,
    score: bool = False,
    n_samples: int = 5,
    cmap: str = "viridis",
    diff_cmap: str = "RdBu_r",
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Visual comparison between ground-truth and generated fields."""
    if not target_fields or generated_fields.size == 0:
        return None

    num_marginals = len(target_fields)
    T = generated_fields.shape[0]
    time_indices = _select_time_indices(T, num_marginals if num_marginals > 0 else T)
    num_cols = min(n_samples, min(len(target_fields[0]), generated_fields.shape[1]))

    if num_cols == 0:
        return None

    # Fit within letter paper dimensions
    col_width = min(3, 8.5 / max(num_cols, 1))
    row_height = min(1.5, 10.5 / max(3 * len(time_indices), 1))
    fig, axes = plt.subplots(
        nrows=3 * len(time_indices),
        ncols=num_cols,
        figsize=(col_width * num_cols, row_height * 3 * len(time_indices)),
    )
    if axes.ndim == 1:
        axes = axes.reshape(3 * len(time_indices), num_cols)

    for t_pos, gen_idx in enumerate(time_indices):
        target_idx = min(t_pos, num_marginals - 1)
        gen_block = generated_fields[gen_idx, :num_cols]
        target_block = target_fields[target_idx][:num_cols]
        diff_block = target_block - gen_block

        combined = np.concatenate([target_block.reshape(-1), gen_block.reshape(-1)])
        vmin = np.percentile(combined, 1)
        vmax = np.percentile(combined, 99)
        vmax_diff = np.max(np.abs(diff_block)) + _NUMERIC_EPS

        for col in range(num_cols):
            base_row = 3 * t_pos
            ax_target = axes[base_row, col]
            ax_generated = axes[base_row + 1, col]
            ax_diff = axes[base_row + 2, col]

            ax_target.imshow(target_block[col], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax_generated.imshow(gen_block[col], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax_diff.imshow(
                diff_block[col],
                cmap=diff_cmap,
                vmin=-vmax_diff,
                vmax=vmax_diff,
                origin="lower",
            )

            for ax in (ax_target, ax_generated, ax_diff):
                ax.set_xticks([])
                ax.set_yticks([])

            if col == 0:
                time_val = zt[target_idx] if target_idx < len(zt) else zt[-1] if zt else gen_idx
                ax_target.set_ylabel(f"Target\n(t={time_val:.2f})", fontsize=9)
                ax_generated.set_ylabel("Generated", fontsize=9)
                ax_diff.set_ylabel("Diff (T-G)", fontsize=9)

            if t_pos == 0:
                ax_target.set_title(f"Sample {col + 1}", fontsize=11)

    fig.tight_layout()

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"field_sample_comparison_{diffeq}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)

    if close:
        plt.close(fig)
        return None
    return fig


def compute_sample_correlation_matrix_with_eigen(
    samples: torch.Tensor,
    *,
    truncate: bool = False,
    variance_threshold: float = 0.999,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute correlation matrix and eigen-decomposition statistics."""
    if samples.ndim != 2:
        raise ValueError("Input samples must be 2D with shape [N, D].")

    n_samples = samples.shape[0]
    if n_samples < 1:
        raise ValueError("At least one sample is required.")

    samples = samples.float()
    mean = samples.mean(dim=0, keepdim=True)
    centered = samples - mean

    if n_samples == 1:
        covariance = torch.zeros(
            (samples.shape[1], samples.shape[1]),
            dtype=samples.dtype,
            device=samples.device,
        )
    else:
        covariance = (centered.T @ centered) / (n_samples - 1)
    covariance = 0.5 * (covariance + covariance.T)

    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total_var = torch.clamp(eigvals.sum(), min=_NUMERIC_EPS)
    variance_ratio = torch.cumsum(eigvals, dim=0) / total_var

    if variance_threshold >= 1.0:
        n_components = eigvals.numel()
    else:
        threshold_tensor = torch.tensor(
            variance_threshold,
            dtype=variance_ratio.dtype,
            device=variance_ratio.device,
        )
        n_components = int(torch.searchsorted(variance_ratio, threshold_tensor, right=False).item()) + 1
    n_components = max(1, min(n_components, eigvals.numel()))

    if truncate:
        covariance = reconstruct_covariance_from_truncated_eigen(eigvals, eigvecs, n_components)

    correlation = compute_correlation_from_covariance(covariance)

    info = {
        "mean": mean.squeeze(0),
        "covariance": covariance,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "variance_ratio": variance_ratio,
        "n_components": n_components,
    }
    return correlation, info


def reconstruct_covariance_from_truncated_eigen(
    eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, k: int
) -> torch.Tensor:
    """Reconstruct covariance using top-k eigenpairs."""
    k = max(1, min(int(k), eigenvalues.shape[0]))
    vals = eigenvalues[:k]
    vecs = eigenvectors[:, :k]
    return (vecs * vals) @ vecs.T


def compute_correlation_from_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """Convert covariance matrix to correlation matrix."""
    diag = torch.diag(covariance)
    std = torch.sqrt(torch.clamp(diag, min=_NUMERIC_EPS))
    inv_std = 1.0 / std
    corr = covariance * inv_std[:, None] * inv_std[None, :]
    corr = torch.clamp(corr, min=-1.0, max=1.0)
    corr.fill_diagonal_(1.0)
    return corr


def relative_covariance_frobenius_distance(
    reference: torch.Tensor, estimate: torch.Tensor
) -> float:
    """Relative Frobenius norm between two covariance (or correlation) matrices."""
    diff_norm = torch.linalg.norm(reference - estimate, ord="fro")
    ref_norm = torch.linalg.norm(reference, ord="fro")
    if ref_norm <= _NUMERIC_EPS:
        return float("inf") if diff_norm > _NUMERIC_EPS else 0.0
    return (diff_norm / ref_norm).item()


def plot_eigenvalue_spectra_comparison(
    target_samples: Mapping[float, torch.Tensor],
    generated_samples: Mapping[float, torch.Tensor],
    time_points: Sequence[float],
    *,
    variance_threshold: float = 0.999,
    outdir: str | None = None,
    run: Any = None,
    score: bool = False,
    filename_prefix: str | None = None,
    wandb_prefix: str = "diagnostics",
    close: bool = True,
) -> tuple[Figure, Figure] | None:
    """Plot eigenvalue spectra and cumulative variance comparisons."""
    format_for_paper()

    n_times = len(time_points)
    if n_times == 0:
        raise ValueError("time_points must contain at least one value.")
    ncols = min(3, n_times)
    nrows = (n_times + ncols - 1) // ncols

    # Fit within letter paper dimensions
    col_width = min(5, 8.5 / max(ncols, 1))
    row_height = min(4, 10.5 / max(nrows, 1))
    fig_eigen, axes_eigen = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows))
    fig_cumvar, axes_cumvar = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows))

    axes_eigen = np.atleast_1d(axes_eigen).flatten()
    axes_cumvar = np.atleast_1d(axes_cumvar).flatten()

    for idx, time_point in enumerate(time_points):
        target_data = target_samples[time_point]
        generated_data = generated_samples[time_point]

        _, target_info = compute_sample_correlation_matrix_with_eigen(
            _to_tensor(target_data), truncate=False, variance_threshold=variance_threshold
        )
        _, gen_info = compute_sample_correlation_matrix_with_eigen(
            _to_tensor(generated_data), truncate=False, variance_threshold=variance_threshold
        )

        target_vals = target_info["eigenvalues"].cpu().numpy()
        gen_vals = gen_info["eigenvalues"].cpu().numpy()
        target_cumvar = target_info["variance_ratio"].cpu().numpy()
        gen_cumvar = gen_info["variance_ratio"].cpu().numpy()
        target_k = target_info["n_components"]
        gen_k = gen_info["n_components"]

        n_show = max(target_k, gen_k)
        x_axis = np.arange(1, n_show + 1)

        ax_eig = axes_eigen[idx]
        ax_eig.loglog(x_axis, target_vals[:n_show], "b-", label="Target", linewidth=2, alpha=0.7)
        ax_eig.loglog(x_axis, gen_vals[:n_show], "r--", label="Generated", linewidth=2, alpha=0.7)
        ax_eig.axvline(target_k, color="blue", linestyle=":", linewidth=1.5, alpha=0.5)
        ax_eig.axvline(gen_k, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
        ax_eig.set_xlabel("Mode Index")
        ax_eig.set_ylabel("Eigenvalue")
        ax_eig.set_title(f"t = {time_point:.2f}\nTarget k={target_k}, Gen k={gen_k}")
        ax_eig.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax_eig.legend(fontsize=8, loc="best")

        ax_cum = axes_cumvar[idx]
        ax_cum.semilogx(x_axis, target_cumvar[:n_show] * 100, "b-", label="Target", linewidth=2, alpha=0.7)
        ax_cum.semilogx(x_axis, gen_cumvar[:n_show] * 100, "r--", label="Generated", linewidth=2, alpha=0.7)
        ax_cum.axhline(variance_threshold * 100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax_cum.axvline(target_k, color="blue", linestyle=":", linewidth=1.5, alpha=0.5)
        ax_cum.axvline(gen_k, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
        ax_cum.set_xlabel("Number of Components")
        ax_cum.set_ylabel("Cumulative Variance (%)")
        ax_cum.set_ylim([90, 100.5])
        ax_cum.set_title(f"t = {time_point:.2f}")
        ax_cum.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax_cum.legend(fontsize=8, loc="lower right")

    for idx in range(n_times, len(axes_eigen)):
        axes_eigen[idx].set_visible(False)
        axes_cumvar[idx].set_visible(False)

    fig_eigen.suptitle("Eigenvalue Spectra Comparison", fontsize=14, y=0.995)
    fig_cumvar.suptitle("Cumulative Variance Explained", fontsize=14, y=0.995)
    fig_eigen.tight_layout()
    fig_cumvar.tight_layout()

    diffeq = "sde" if score else "ode"
    base = filename_prefix or f"covariance_diagnostics_{diffeq}"
    if outdir is not None:
        _save_and_log_figure(
            fig_eigen,
            os.path.join(outdir, f"{base}_eigenvalues"),
            run,
            f"{wandb_prefix}/{base}_eigenvalues",
        )
        _save_and_log_figure(
            fig_cumvar,
            os.path.join(outdir, f"{base}_cumulative_variance"),
            run,
            f"{wandb_prefix}/{base}_cumulative_variance",
        )

    if close:
        plt.close(fig_eigen)
        plt.close(fig_cumvar)
        return None
    return fig_eigen, fig_cumvar


def plot_covariance_heatmaps_comparison(
    target_samples: Mapping[float, torch.Tensor],
    generated_samples: Mapping[float, torch.Tensor],
    time_points: Sequence[float],
    *,
    max_dim_for_heatmap: int = 32,
    variance_threshold: float = 0.999,
    outdir: str | None = None,
    run: Any = None,
    score: bool = False,
    filename_prefix: str | None = None,
    wandb_prefix: str = "diagnostics",
    close: bool = True,
) -> Figure | None:
    """Plot truncated covariance/correlation heatmaps across time."""
    format_for_paper()
    n_times = len(time_points)
    if n_times == 0:
        raise ValueError("time_points must contain at least one value.")

    # Fit within letter paper dimensions (3 rows of heatmaps)
    col_width = min(5, 8.5 / max(n_times, 1))
    total_height = min(10.5, 4 * 3)  # 3 rows, max ~4 inches each
    fig, axes = plt.subplots(3, n_times, figsize=(col_width * n_times, total_height))
    if n_times == 1:
        axes = axes.reshape(3, 1)

    for col_idx, time_point in enumerate(time_points):
        target_data = _to_tensor(target_samples[time_point])
        generated_data = _to_tensor(generated_samples[time_point])

        _, target_info = compute_sample_correlation_matrix_with_eigen(
            target_data, truncate=False, variance_threshold=variance_threshold
        )
        _, generated_info = compute_sample_correlation_matrix_with_eigen(
            generated_data, truncate=False, variance_threshold=variance_threshold
        )

        target_k = target_info["n_components"]
        target_cov_trunc = reconstruct_covariance_from_truncated_eigen(
            target_info["eigenvalues"], target_info["eigenvectors"], target_k
        )
        generated_cov_trunc = reconstruct_covariance_from_truncated_eigen(
            generated_info["eigenvalues"], generated_info["eigenvectors"], target_k
        )

        target_corr_tensor = compute_correlation_from_covariance(target_cov_trunc)
        generated_corr_tensor = compute_correlation_from_covariance(generated_cov_trunc)
        rel_fro = relative_covariance_frobenius_distance(
            target_corr_tensor, generated_corr_tensor
        )

        target_corr = target_corr_tensor.cpu().numpy()
        generated_corr = generated_corr_tensor.cpu().numpy()

        max_dim = min(max_dim_for_heatmap, target_corr.shape[0], generated_corr.shape[0])
        target_corr = target_corr[:max_dim, :max_dim]
        generated_corr = generated_corr[:max_dim, :max_dim]
        diff_corr = target_corr - generated_corr

        im_target = axes[0, col_idx].imshow(target_corr, cmap="coolwarm", vmin=-1, vmax=1)
        axes[0, col_idx].set_title(f"t = {time_point:.2f}")
        if col_idx == 0:
            axes[0, col_idx].set_ylabel("Target\n(Truncated)", fontsize=10)
        plt.colorbar(im_target, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

        im_gen = axes[1, col_idx].imshow(generated_corr, cmap="coolwarm", vmin=-1, vmax=1)
        if col_idx == 0:
            axes[1, col_idx].set_ylabel("Generated\n(Truncated)", fontsize=10)
        plt.colorbar(im_gen, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

        vmax_diff = np.abs(diff_corr).max()
        im_diff = axes[2, col_idx].imshow(
            diff_corr, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff
        )
        if col_idx == 0:
            axes[2, col_idx].set_ylabel("Difference", fontsize=10)
        axes[2, col_idx].set_xlabel(f"k={target_k}\nRelFro={rel_fro:.4f}", fontsize=9)
        plt.colorbar(im_diff, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Truncated Correlation Matrix Comparison\n(Truncated at {variance_threshold * 100:.1f}% variance using Target k)",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    diffeq = "sde" if score else "ode"
    base_name = filename_prefix or f"truncated_covariance_{diffeq}"
    if outdir is not None:
        _save_and_log_figure(
            fig,
            os.path.join(outdir, base_name),
            run,
            f"{wandb_prefix}/{base_name}",
        )

    if close:
        plt.close(fig)
        return None
    return fig


def visualize_all_field_reconstructions(
    traj_coeffs: np.ndarray | torch.Tensor,
    testdata: Sequence[np.ndarray | torch.Tensor],
    pca_info: Mapping[str, Any],
    zt: Sequence[float],
    outdir: str,
    run: Any,
    score: bool = False,
    *,
    covariance_targets: Mapping[float, torch.Tensor] | None = None,
    covariance_generations: Mapping[float, torch.Tensor] | None = None,
    covariance_time_points: Sequence[float] | None = None,
    variance_threshold: float = 0.999,
    max_dim_for_heatmap: int = 32,
) -> None:
    """Entry point producing all field reconstruction diagnostics."""
    resolution = int(np.sqrt(int(pca_info["data_dim"])))

    print(f"Reconstructing fields with resolution {resolution}x{resolution}...")
    fields = reconstruct_fields_from_coefficients(traj_coeffs, pca_info, resolution)
    print(f"Generated fields shape: {fields.shape}")

    testdata_fields = [
        reconstruct_fields_from_coefficients(test_marginal, pca_info, resolution)
        for test_marginal in testdata
    ]

    print("Creating field visualizations...")
    plot_field_snapshots(fields, zt, outdir, run, n_samples=5, score=score)
    plot_field_evolution_gif(fields, zt, outdir, run, sample_idx=0, score=score, fps=5)
    for sample_idx in (1, 2):
        if sample_idx < fields.shape[1]:
            plot_field_evolution_gif(
                fields, zt, outdir, run, sample_idx=sample_idx, score=score, fps=5
            )
    if testdata_fields:
        plot_field_statistics(fields, zt, testdata_fields, outdir, run, score=score)
        plot_sample_comparison_grid(
            testdata_fields,
            fields,
            zt,
            outdir,
            run,
            score=score,
        )
    plot_spatial_correlation(fields, zt, outdir, run, score=score)

    if covariance_targets and covariance_generations:
        if covariance_time_points is None:
            covariance_time_points = sorted(
                set(covariance_targets.keys()) & set(covariance_generations.keys())
            )
        if covariance_time_points:
            plot_eigenvalue_spectra_comparison(
                covariance_targets,
                covariance_generations,
                covariance_time_points,
                variance_threshold=variance_threshold,
                outdir=outdir,
                run=run,
                score=score,
            )
            plot_covariance_heatmaps_comparison(
                covariance_targets,
                covariance_generations,
                covariance_time_points,
                max_dim_for_heatmap=max_dim_for_heatmap,
                variance_threshold=variance_threshold,
                outdir=outdir,
                run=run,
                score=score,
            )

    print("Field visualizations complete!")


def plot_vector_field_2d_projection(
    flow_model: Any,
    x0: torch.Tensor,
    xT: torch.Tensor,
    time_points: Sequence[float],
    dims: tuple[int, ...],
    outdir: str | None,
    run: Any,
    *,
    n_grid: int = 20,
    pca_components: int = 2,
    score: bool = False,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Visualize learned vector field via 2D PCA projection with quiver plot.
    
    Args:
        flow_model: Trained flow model
        x0: Starting distribution samples (N, C, H, W)
        xT: Target distribution samples (N, C, H, W)
        time_points: Time values to visualize
        dims: Data dimensions
        outdir: Output directory
        run: Logging object
        n_grid: Grid resolution for evaluation
        pca_components: Number of PCA components (must be 2)
        score: Whether this is score-based
        filename_prefix: Optional filename prefix
        wandb_key: Optional wandb key
        close: Whether to close figure
    
    Returns:
        Figure object if close=False, None otherwise
    """
    from sklearn.decomposition import PCA
    
    format_for_paper()
    
    # Flatten samples for PCA
    x0_flat = x0.view(x0.shape[0], -1).cpu().numpy()
    xT_flat = xT.view(xT.shape[0], -1).cpu().numpy()
    all_samples = np.vstack([x0_flat, xT_flat])
    
    # Fit PCA on combined data
    pca = PCA(n_components=pca_components)
    pca.fit(all_samples)
    
    # Project endpoints
    x0_proj = pca.transform(x0_flat)
    xT_proj = pca.transform(xT_flat)
    
    # Create grid in PCA space
    x_min, x_max = min(x0_proj[:, 0].min(), xT_proj[:, 0].min()), max(x0_proj[:, 0].max(), xT_proj[:, 0].max())
    y_min, y_max = min(x0_proj[:, 1].min(), xT_proj[:, 1].min()), max(x0_proj[:, 1].max(), xT_proj[:, 1].max())
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, n_grid),
        np.linspace(y_min - y_margin, y_max + y_margin, n_grid)
    )
    
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # Inverse transform to original space
    grid_points_high = pca.inverse_transform(grid_points_2d)
    grid_tensor = torch.from_numpy(grid_points_high).float().to(x0.device)
    grid_tensor = grid_tensor.view(-1, *dims)
    
    n_times = len(time_points)
    ncols = min(3, n_times)
    nrows = (n_times + ncols - 1) // ncols
    
    # Fit within letter paper dimensions
    col_width = min(6, 8.5 / max(ncols, 1))
    row_height = min(5, 10.5 / max(nrows, 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows))
    if n_times == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    
    flow_model.eval()
    with torch.no_grad():
        for idx, t in enumerate(time_points):
            ax = axes[idx]
            
            # Evaluate flow field at grid points
            t_tensor = torch.full((grid_tensor.shape[0],), t, device=grid_tensor.device, dtype=grid_tensor.dtype)
            velocities = flow_model(t_tensor, grid_tensor)
            
            # Project velocities to PCA space
            velocities_flat = velocities.view(velocities.shape[0], -1).cpu().numpy()
            velocities_proj = pca.transform(grid_points_high + velocities_flat) - grid_points_2d
            
            U = velocities_proj[:, 0].reshape(n_grid, n_grid)
            V = velocities_proj[:, 1].reshape(n_grid, n_grid)
            
            # Quiver plot
            ax.quiver(xx, yy, U, V, alpha=0.6, scale=None, scale_units='xy', angles='xy')
            
            # Overlay data points
            ax.scatter(x0_proj[:, 0], x0_proj[:, 1], c='blue', s=30, alpha=0.6, label='Source', edgecolors='k', linewidths=0.5)
            ax.scatter(xT_proj[:, 0], xT_proj[:, 1], c='red', s=30, alpha=0.6, label='Target', edgecolors='k', linewidths=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=10)
            ax.set_title(f't = {t:.2f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
    
    for idx in range(n_times, len(axes)):
        axes[idx].set_visible(False)
    
    diffeq = "sde" if score else "ode"
    fig.suptitle(f"Vector Field Projection (PCA 2D) - {diffeq.upper()}", fontsize=14, y=0.995)
    fig.tight_layout()
    
    base_name = filename_prefix or f"vector_field_projection_{diffeq}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)
    
    if close:
        plt.close(fig)
        return None
    return fig


def plot_vector_field_streamplot(
    flow_model: Any,
    x0: torch.Tensor,
    xT: torch.Tensor,
    time_points: Sequence[float],
    dims: tuple[int, ...],
    outdir: str | None,
    run: Any,
    *,
    n_grid: int = 30,
    pca_components: int = 2,
    score: bool = False,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Visualize learned vector field via 2D PCA projection with streamplot.
    
    Args:
        flow_model: Trained flow model
        x0: Starting distribution samples (N, C, H, W)
        xT: Target distribution samples (N, C, H, W)
        time_points: Time values to visualize
        dims: Data dimensions
        outdir: Output directory
        run: Logging object
        n_grid: Grid resolution for evaluation
        pca_components: Number of PCA components (must be 2)
        score: Whether this is score-based
        filename_prefix: Optional filename prefix
        wandb_key: Optional wandb key
        close: Whether to close figure
    
    Returns:
        Figure object if close=False, None otherwise
    """
    from sklearn.decomposition import PCA
    
    format_for_paper()
    
    # Flatten samples for PCA
    x0_flat = x0.view(x0.shape[0], -1).cpu().numpy()
    xT_flat = xT.view(xT.shape[0], -1).cpu().numpy()
    all_samples = np.vstack([x0_flat, xT_flat])
    
    # Fit PCA on combined data
    pca = PCA(n_components=pca_components)
    pca.fit(all_samples)
    
    # Project endpoints
    x0_proj = pca.transform(x0_flat)
    xT_proj = pca.transform(xT_flat)
    
    # Create grid in PCA space
    x_min, x_max = min(x0_proj[:, 0].min(), xT_proj[:, 0].min()), max(x0_proj[:, 0].max(), xT_proj[:, 0].max())
    y_min, y_max = min(x0_proj[:, 1].min(), xT_proj[:, 1].min()), max(x0_proj[:, 1].max(), xT_proj[:, 1].max())
    
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, n_grid),
        np.linspace(y_min - y_margin, y_max + y_margin, n_grid)
    )
    
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # Inverse transform to original space
    grid_points_high = pca.inverse_transform(grid_points_2d)
    grid_tensor = torch.from_numpy(grid_points_high).float().to(x0.device)
    grid_tensor = grid_tensor.view(-1, *dims)
    
    n_times = len(time_points)
    ncols = min(3, n_times)
    nrows = (n_times + ncols - 1) // ncols
    
    # Fit within letter paper dimensions
    col_width = min(6, 8.5 / max(ncols, 1))
    row_height = min(5, 10.5 / max(nrows, 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows))
    if n_times == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    
    flow_model.eval()
    with torch.no_grad():
        for idx, t in enumerate(time_points):
            ax = axes[idx]
            
            # Evaluate flow field at grid points
            t_tensor = torch.full((grid_tensor.shape[0],), t, device=grid_tensor.device, dtype=grid_tensor.dtype)
            velocities = flow_model(t_tensor, grid_tensor)
            
            # Project velocities to PCA space
            velocities_flat = velocities.view(velocities.shape[0], -1).cpu().numpy()
            velocities_proj = pca.transform(grid_points_high + velocities_flat) - grid_points_2d
            
            U = velocities_proj[:, 0].reshape(n_grid, n_grid)
            V = velocities_proj[:, 1].reshape(n_grid, n_grid)
            
            # Compute velocity magnitude for color
            speed = np.sqrt(U**2 + V**2)
            
            # Streamplot
            strm = ax.streamplot(xx, yy, U, V, color=speed, cmap='viridis', 
                                linewidth=1.5, density=1.5, arrowsize=1.2, arrowstyle='->')
            plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude', fraction=0.046, pad=0.04)
            
            # Overlay data points
            ax.scatter(x0_proj[:, 0], x0_proj[:, 1], c='cyan', s=40, alpha=0.8, 
                      label='Source', edgecolors='black', linewidths=1, marker='o')
            ax.scatter(xT_proj[:, 0], xT_proj[:, 1], c='magenta', s=40, alpha=0.8, 
                      label='Target', edgecolors='black', linewidths=1, marker='s')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=10)
            ax.set_title(f't = {t:.2f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
    
    for idx in range(n_times, len(axes)):
        axes[idx].set_visible(False)
    
    diffeq = "sde" if score else "ode"
    fig.suptitle(f"Vector Field Streamlines (PCA 2D) - {diffeq.upper()}", fontsize=14, y=0.995)
    fig.tight_layout()
    
    base_name = filename_prefix or f"vector_field_streamplot_{diffeq}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)
    
    if close:
        plt.close(fig)
        return None
    return fig


def plot_interpolated_probability_paths(
    marginals: Sequence[torch.Tensor],
    time_points: Sequence[float],
    *,
    n_samples: int = 10,
    n_eval_points: int = 50,
    spline_type: str = "pchip",
    outdir: str | None = None,
    run: Any = None,
    pca_components: int = 2,
    filename_prefix: str | None = None,
    wandb_key: str | None = None,
    close: bool = True,
) -> Figure | None:
    """Visualize interpolated probability paths using overlapping triplet interpolation.
    
    This function replicates the EXACT training-time probability paths created by 
    overlapping triplet interpolation: for each sliding window (k, k+1, k+2), a 
    component-wise spline interpolates across the 3 marginals, and segments are 
    stitched to form complete trajectories.
    
    Args:
        marginals: List of marginal distributions at different times, each (N, C, H, W) or (N, dim)
        time_points: Time values at marginals (e.g., [0.0, 0.2, 0.4, ..., 1.0])
        n_samples: Number of sample paths to visualize
        n_eval_points: Number of points to evaluate along each triplet segment
        spline_type: Type of spline ('pchip' for monotonic cubic Hermite, 'cubic' for natural cubic)
        outdir: Output directory
        run: Logging object
        pca_components: Number of PCA components (must be 2)
        filename_prefix: Optional filename prefix
        wandb_key: Optional wandb key
        close: Whether to close figure
    
    Returns:
        Figure object if close=False, None otherwise
    """
    from scipy.interpolate import PchipInterpolator, CubicSpline
    from sklearn.decomposition import PCA
    
    format_for_paper()
    
    # Convert marginals to flattened numpy arrays
    marginals_flat = []
    for marginal in marginals:
        if isinstance(marginal, torch.Tensor):
            marginals_flat.append(marginal.view(marginal.shape[0], -1).cpu().numpy())
        else:
            marginals_flat.append(np.asarray(marginal).reshape(marginal.shape[0], -1))
    
    # Combine all marginals for PCA
    all_samples = np.vstack(marginals_flat)
    
    # Fit PCA on combined data
    pca = PCA(n_components=pca_components)
    pca.fit(all_samples)
    
    # Project all marginals
    marginals_proj = [pca.transform(marg) for marg in marginals_flat]
    
    # Select spline function
    if spline_type.lower() == "pchip":
        spline_fn = PchipInterpolator
        spline_name = "Monotonic Cubic Hermite (PCHIP)"
    elif spline_type.lower() == "cubic":
        spline_fn = CubicSpline
        spline_name = "Natural Cubic Spline"
    else:
        raise ValueError(f"Unknown spline type: {spline_type}")
    
    # Create figure - fit within letter width
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5))
    
    n_marginals = len(marginals_flat)
    n_triplets = n_marginals - 2
    n_to_plot = min(n_samples, marginals_flat[0].shape[0])
    
    if n_triplets < 1:
        raise ValueError(f"Need at least 3 marginals for triplet interpolation, got {n_marginals}")
    
    # Left plot: Individual triplet-stitched paths
    ax_paths = axes[0]
    
    for sample_idx in range(n_to_plot):
        # Build complete path by stitching triplet segments
        complete_path_pixel = []
        complete_path_times = []
        
        for triplet_idx in range(n_triplets):
            # Get three consecutive marginals for this triplet
            triplet_marginals = [
                marginals_flat[triplet_idx + offset][sample_idx]
                for offset in range(3)
            ]
            triplet_times = time_points[triplet_idx:triplet_idx + 3]
            
            # Interpolate within this triplet using component-wise splines
            t_eval = np.linspace(triplet_times[0], triplet_times[-1], n_eval_points)
            
            interpolated_segment = np.zeros((n_eval_points, triplet_marginals[0].shape[0]))
            for dim_idx in range(triplet_marginals[0].shape[0]):
                dim_values = np.array([m[dim_idx] for m in triplet_marginals])
                spline = spline_fn(triplet_times, dim_values)
                interpolated_segment[:, dim_idx] = spline(t_eval)
            
            # Add to complete path (avoid duplicate endpoints between triplets)
            if triplet_idx == 0:
                complete_path_pixel.append(interpolated_segment)
                complete_path_times.append(t_eval)
            else:
                # Skip first point to avoid duplication
                complete_path_pixel.append(interpolated_segment[1:])
                complete_path_times.append(t_eval[1:])
        
        # Concatenate all segments
        complete_path_pixel = np.vstack(complete_path_pixel)
        complete_path_times = np.concatenate(complete_path_times)
        
        # Project complete path to PCA space
        path_proj = pca.transform(complete_path_pixel)
        
        # Plot path with color gradient by time
        n_points = len(complete_path_times)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        for i in range(n_points - 1):
            ax_paths.plot(
                path_proj[i:i+2, 0], 
                path_proj[i:i+2, 1], 
                c=colors[i], 
                alpha=0.6, 
                linewidth=1.5
            )
    
    # Overlay all marginal endpoints
    colors_marginals = plt.cm.tab10(np.linspace(0, 1, n_marginals))
    for marg_idx in range(n_marginals):
        marg_proj = marginals_proj[marg_idx]
        ax_paths.scatter(
            marg_proj[:n_to_plot, 0], 
            marg_proj[:n_to_plot, 1], 
            c=[colors_marginals[marg_idx]], 
            s=50, 
            alpha=0.7, 
            label=f't={time_points[marg_idx]:.2f}',
            edgecolors='black', 
            linewidths=1, 
            zorder=5
        )
    
    ax_paths.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax_paths.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax_paths.set_title(f'Triplet-Stitched Probability Paths\n({n_to_plot} samples, {n_triplets} overlapping triplets)', fontsize=12)
    ax_paths.legend(loc='best', fontsize=8, ncol=2)
    ax_paths.grid(True, alpha=0.3)
    
    # Right plot: Density of paths (shows where paths concentrate)
    ax_density = axes[1]
    
    # Collect all path points (use more samples for better density estimation)
    all_path_points = []
    n_density_samples = min(50, marginals_flat[0].shape[0])
    
    for sample_idx in range(n_density_samples):
        complete_path_pixel = []
        
        for triplet_idx in range(n_triplets):
            triplet_marginals = [
                marginals_flat[triplet_idx + offset][sample_idx]
                for offset in range(3)
            ]
            triplet_times = time_points[triplet_idx:triplet_idx + 3]
            
            t_eval = np.linspace(triplet_times[0], triplet_times[-1], n_eval_points)
            
            interpolated_segment = np.zeros((n_eval_points, triplet_marginals[0].shape[0]))
            for dim_idx in range(triplet_marginals[0].shape[0]):
                dim_values = np.array([m[dim_idx] for m in triplet_marginals])
                spline = spline_fn(triplet_times, dim_values)
                interpolated_segment[:, dim_idx] = spline(t_eval)
            
            if triplet_idx == 0:
                complete_path_pixel.append(interpolated_segment)
            else:
                complete_path_pixel.append(interpolated_segment[1:])
        
        complete_path_pixel = np.vstack(complete_path_pixel)
        path_proj = pca.transform(complete_path_pixel)
        all_path_points.append(path_proj)
    
    all_path_points = np.vstack(all_path_points)
    
    # Create 2D histogram for density
    hist, xedges, yedges = np.histogram2d(
        all_path_points[:, 0], 
        all_path_points[:, 1], 
        bins=40
    )
    
    im = ax_density.imshow(
        hist.T, 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
        origin='lower', 
        cmap='YlOrRd', 
        aspect='auto',
        alpha=0.8
    )
    plt.colorbar(im, ax=ax_density, label='Path Density', fraction=0.046, pad=0.04)
    
    # Overlay all marginals on density plot
    for marg_idx in range(n_marginals):
        marg_proj = marginals_proj[marg_idx]
        ax_density.scatter(
            marg_proj[:, 0], 
            marg_proj[:, 1], 
            c=[colors_marginals[marg_idx]], 
            s=30, 
            alpha=0.5, 
            edgecolors='black', 
            linewidths=0.5, 
            zorder=5
        )
    
    ax_density.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax_density.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax_density.set_title('Path Density Heatmap', fontsize=12)
    ax_density.grid(True, alpha=0.3)
    
    fig.suptitle(f'Component-wise {spline_name} with Overlapping Triplets', fontsize=14, y=0.98)
    fig.tight_layout()
    
    base_name = filename_prefix or f"interpolated_paths_triplet_{spline_type}"
    wandb_entry = wandb_key or f"visualizations/{base_name}"
    if outdir is not None:
        _save_and_log_figure(fig, os.path.join(outdir, base_name), run, wandb_entry)
    
    if close:
        plt.close(fig)
        return None
    return fig
