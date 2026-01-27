# %%
# %% [markdown]
# # Tran Inclusions Dataset Visualization
#
# This notebook visualizes the tran_inclusions.npz dataset across time:
# - Raw fields at different time scales (micro → meso → macro)
# - PCA coefficient distributions
# - Field statistics and spatial correlation
# - Comparison of raw vs PCA-reconstructed fields
#
# Dataset info:
# - 376 PCA components explaining 100% variance (n_components=0.999)
# - Standard PCA (not whitened)
# - Min-max scaled to [0,1] with epsilon=1e-6

# %%
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add parent directory to path
path_root = Path.cwd().parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from scripts.images.field_visualization import format_for_paper

print("Imports successful!")

# %% [markdown]
# ## Paper Formatting

# %%
format_for_paper()
print("✓ Applied paper formatting")

# %% [markdown]
# ## Load Dataset

# %%
# Load the dataset
data_path = "./data/tran_inclusions.npz"
data_path = Path(path_root) / data_path if not Path(data_path).exists() else Path(data_path)
print(f"Loading dataset from: {data_path}")

npz = np.load(data_path)

# Extract metadata
data_dim = int(npz["data_dim"])
resolution = int(np.sqrt(data_dim))
dataset_format = str(npz["dataset_format"])
scale_mode = str(npz["scale_mode"])
scaling_epsilon = float(npz["scaling_epsilon"])
data_generator = str(npz["data_generator"])
is_whitened = bool(npz["is_whitened"])
whitening_epsilon = float(npz["whitening_epsilon"])

# PCA info
pca_components = npz["pca_components"]  # (n_components, data_dim)
pca_mean = npz["pca_mean"]  # (data_dim,)
pca_explained_variance = npz["pca_explained_variance"]  # (n_components,)
n_components = pca_components.shape[0]

# Held-out information
held_out_indices = npz["held_out_indices"]
held_out_times = npz["held_out_times"]

# Get time values from marginal keys
raw_keys = sorted(
    [k for k in npz.files if k.startswith("raw_marginal_")],
    key=lambda x: float(x.split("_")[-1]),
)
coeff_keys = sorted(
    [k for k in npz.files if k.startswith("marginal_") and not k.startswith("raw_")],
    key=lambda x: float(x.split("_")[-1]),
)

times = np.array([float(k.split("_")[-1]) for k in raw_keys])
n_times = len(times)

# Load raw fields
raw_fields_list = []
for key in raw_keys:
    fields_flat = npz[key]  # (N, data_dim)
    fields_2d = fields_flat.reshape(-1, resolution, resolution)
    raw_fields_list.append(fields_2d)

raw_fields = np.stack(raw_fields_list, axis=0)  # (T, N, H, W)
n_samples = raw_fields.shape[1]

# Load PCA coefficients
coeff_marginals_list = [npz[key] for key in coeff_keys]
coeff_marginals = np.stack(coeff_marginals_list, axis=0)  # (T, N, n_components)

print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"Data generator: {data_generator}")
print(f"Resolution: {resolution}x{resolution}")
print(f"Data dimension: {data_dim}")
print(f"Number of time steps: {n_times}")
print(f"Number of samples: {n_samples}")
print(f"Time values: {times}")
print(f"\nPCA Configuration:")
print(f"  Number of components: {n_components}")
print(f"  Is whitened: {is_whitened}")
print(f"  Variance explained: {pca_explained_variance.sum():.6f}")
print(f"\nScaling:")
print(f"  Scale mode: {scale_mode}")
print(f"  Scaling epsilon: {scaling_epsilon}")
print(f"\nHeld-out information:")
print(f"  Held-out indices: {held_out_indices}")
print(f"  Held-out times: {held_out_times}")
print(f"  PCA was fitted EXCLUDING:")
print(f"    - t=0 (microscale) - auto-excluded for tran_inclusion")
print(f"    - Indices {held_out_indices}: t={held_out_times}")
print("="*70)

# %% [markdown]
# ## Reconstruct Fields from PCA

# %%
def reconstruct_from_pca(coeffs, pca_components, pca_mean, is_whitened, pca_variance=None):
    """Reconstruct fields from PCA coefficients.

    Args:
        coeffs: PCA coefficients, shape (..., n_components)
        pca_components: PCA components, shape (n_components, data_dim)
        pca_mean: PCA mean, shape (data_dim,)
        is_whitened: Whether coefficients are whitened
        pca_variance: PCA explained variance (needed if whitened)

    Returns:
        Reconstructed fields, shape (..., data_dim)
    """
    original_shape = coeffs.shape
    coeffs_flat = coeffs.reshape(-1, coeffs.shape[-1])

    if is_whitened:
        # Undo whitening: multiply by sqrt(eigenvalues)
        if pca_variance is None:
            raise ValueError("pca_variance required for whitened coefficients")
        coeffs_flat = coeffs_flat * np.sqrt(pca_variance)

    # Reconstruct: x = mean + coeffs @ components
    reconstructed = coeffs_flat @ pca_components + pca_mean

    # Reshape back
    return reconstructed.reshape(original_shape[:-1] + (pca_mean.shape[0],))

# Reconstruct fields from PCA coefficients
print("Reconstructing fields from PCA coefficients...")
reconstructed_flat = reconstruct_from_pca(
    coeff_marginals,
    pca_components,
    pca_mean,
    is_whitened,
    pca_explained_variance,
)
reconstructed_fields = reconstructed_flat.reshape(n_times, n_samples, resolution, resolution)
print(f"✓ Reconstructed fields shape: {reconstructed_fields.shape}")

# %% [markdown]
# ## Raw Field Visualization Across Time

# %%
# Create output directory
output_dir = Path("figures/dataset_viz")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# %%
def plot_field_grid_across_time(
    fields: np.ndarray,
    times: np.ndarray,
    n_samples_to_show: int = 5,
    title: str = "Fields Across Time",
    cmap: str = 'viridis',
):
    """Plot a grid of fields: rows=samples, cols=time steps.

    Args:
        fields: Shape (T, N, H, W)
        times: Time values, shape (T,)
        n_samples_to_show: Number of samples to display
        title: Figure title
        cmap: Colormap
    """
    T, N, H, W = fields.shape
    n_samples = min(n_samples_to_show, N)

    fig, axes = plt.subplots(n_samples, T, figsize=(2*T, 2*n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Use global colormap limits across all fields
    vmin, vmax = fields.min(), fields.max()

    for sample_idx in range(n_samples):
        for t_idx in range(T):
            ax = axes[sample_idx, t_idx]
            im = ax.imshow(fields[t_idx, sample_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles (time values)
            if sample_idx == 0:
                ax.set_title(f't={times[t_idx]:.4f}', fontsize=9)

            # Row labels (sample index)
            if t_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx}', fontsize=9)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Field Value')

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    return fig

# %%
print("\n" + "="*60)
print("PLOTTING RAW FIELDS ACROSS TIME")
print("="*60)

fig = plot_field_grid_across_time(
    raw_fields,
    times,
    n_samples_to_show=5,
    title=f"Raw Fields: {data_generator.upper()} (Micro → Meso → Macro)",
    cmap='viridis',
)
save_path = output_dir / "raw_fields_across_time.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## PCA Reconstruction Quality

# %%
# Compute reconstruction error
def compute_reconstruction_rmse(original, reconstructed):
    """Compute RMSE between original and reconstructed fields."""
    diff = original - reconstructed
    rmse_per_sample = np.sqrt(np.mean(diff**2, axis=(2, 3)))  # (T, N)
    return rmse_per_sample

rmse_per_sample = compute_reconstruction_rmse(raw_fields, reconstructed_fields)
rmse_mean = rmse_per_sample.mean(axis=1)  # (T,)
rmse_std = rmse_per_sample.std(axis=1)  # (T,)

print("\n" + "="*60)
print("PCA RECONSTRUCTION QUALITY")
print("="*60)
print(f"\nOverall RMSE: {rmse_per_sample.mean():.4e}")
print(f"\nPer-time RMSE:")
for t_idx, t_val in enumerate(times):
    print(f"  t={t_val:.4f}: {rmse_mean[t_idx]:.3e} ± {rmse_std[t_idx]:.3e}")

# %%
# Plot reconstruction error
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(times, rmse_mean, yerr=rmse_std, fmt='o-', capsize=4, linewidth=2)
ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title(f'PCA Reconstruction Error ({n_components} components)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_path = output_dir / "pca_reconstruction_error.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Visual Comparison: Raw vs PCA Reconstructed

# %%
def plot_raw_vs_reconstructed(
    raw_fields: np.ndarray,
    recon_fields: np.ndarray,
    times: np.ndarray,
    sample_idx: int = 0,
):
    """Compare raw vs reconstructed fields side by side.

    Args:
        raw_fields: Shape (T, N, H, W)
        recon_fields: Shape (T, N, H, W)
        times: Time values
        sample_idx: Which sample to visualize
    """
    T = raw_fields.shape[0]

    fig, axes = plt.subplots(3, T, figsize=(2.5*T, 7.5))

    vmin = min(raw_fields[:, sample_idx].min(), recon_fields[:, sample_idx].min())
    vmax = max(raw_fields[:, sample_idx].max(), recon_fields[:, sample_idx].max())

    for t_idx in range(T):
        # Raw
        ax = axes[0, t_idx]
        im_raw = ax.imshow(raw_fields[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f't={times[t_idx]:.4f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if t_idx == 0:
            ax.set_ylabel('Raw', fontsize=11, fontweight='bold')

        # Reconstructed
        ax = axes[1, t_idx]
        im_recon = ax.imshow(recon_fields[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if t_idx == 0:
            ax.set_ylabel('PCA Recon', fontsize=11)

        # Difference
        diff = raw_fields[t_idx, sample_idx] - recon_fields[t_idx, sample_idx]
        vmax_diff = max(abs(diff.min()), abs(diff.max())) if diff.max() != diff.min() else 1e-10
        ax = axes[2, t_idx]
        im_diff = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if t_idx == 0:
            ax.set_ylabel('Difference', fontsize=11)

        # RMSE annotation
        rmse = np.sqrt(np.mean(diff**2))
        ax.text(0.5, -0.12, f'RMSE: {rmse:.2e}', transform=ax.transAxes,
                ha='center', fontsize=8)

    # Colorbars
    fig.subplots_adjust(right=0.92)
    cax1 = fig.add_axes([0.94, 0.65, 0.015, 0.25])
    fig.colorbar(im_raw, cax=cax1, label='Field Value')

    cax2 = fig.add_axes([0.94, 0.1, 0.015, 0.25])
    fig.colorbar(im_diff, cax=cax2, label='Error')

    fig.suptitle(f'Sample {sample_idx}: Raw vs PCA Reconstruction', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    return fig

# %%
print("\n" + "="*60)
print("RAW vs PCA RECONSTRUCTED COMPARISON")
print("="*60)

for sample_idx in range(min(3, n_samples)):
    fig = plot_raw_vs_reconstructed(raw_fields, reconstructed_fields, times, sample_idx=sample_idx)
    save_path = output_dir / f"raw_vs_pca_sample{sample_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path.name}")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Field Statistics Across Time

# %%
def plot_field_statistics(fields: np.ndarray, times: np.ndarray, title: str = "Field Statistics"):
    """Plot mean, std, min, max of fields across time.

    Args:
        fields: Shape (T, N, H, W)
        times: Time values
    """
    T = fields.shape[0]

    # Compute statistics
    means = np.array([fields[t].mean() for t in range(T)])
    stds = np.array([fields[t].std() for t in range(T)])
    mins = np.array([fields[t].min() for t in range(T)])
    maxs = np.array([fields[t].max() for t in range(T)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean
    ax = axes[0, 0]
    ax.plot(times, means, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Mean Field Value')
    ax.set_title('Mean')
    ax.grid(True, alpha=0.3)

    # Std
    ax = axes[0, 1]
    ax.plot(times, stds, 'o-', linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Std Dev')
    ax.set_title('Standard Deviation')
    ax.grid(True, alpha=0.3)

    # Min
    ax = axes[1, 0]
    ax.plot(times, mins, 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Min Field Value')
    ax.set_title('Minimum')
    ax.grid(True, alpha=0.3)

    # Max
    ax = axes[1, 1]
    ax.plot(times, maxs, 'o-', linewidth=2, markersize=6, color='red')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Max Field Value')
    ax.set_title('Maximum')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("FIELD STATISTICS ACROSS TIME")
print("="*60)

fig = plot_field_statistics(raw_fields, times, title="Raw Field Statistics Across Time")
save_path = output_dir / "field_statistics.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Field Value Distributions

# %%
def plot_field_distributions(fields: np.ndarray, times: np.ndarray, n_bins: int = 50):
    """Plot histograms of field values at each time step.

    Args:
        fields: Shape (T, N, H, W)
        times: Time values
        n_bins: Number of histogram bins
    """
    T = fields.shape[0]

    # Select time steps to display
    if T > 8:
        time_indices = np.linspace(0, T - 1, 8, dtype=int)
    else:
        time_indices = np.arange(T)

    n_cols = len(time_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    for col, t_idx in enumerate(time_indices):
        ax = axes[col]

        vals = fields[t_idx].ravel()
        ax.hist(vals, bins=n_bins, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Field Value', fontsize=9)
        if col == 0:
            ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f't={times[t_idx]:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Field Value Distributions Across Time', fontsize=14, y=0.995)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("FIELD VALUE DISTRIBUTIONS")
print("="*60)

fig = plot_field_distributions(raw_fields, times)
save_path = output_dir / "field_distributions.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## PCA Coefficient Distributions

# %%
def plot_pca_coeff_distributions(coeffs: np.ndarray, times: np.ndarray, n_dims: int = 6):
    """Plot distributions of PCA coefficients for the first few dimensions.

    Args:
        coeffs: Shape (T, N, n_components)
        times: Time values
        n_dims: Number of PCA dimensions to show
    """
    T, N, K = coeffs.shape
    n_dims = min(n_dims, K)

    fig, axes = plt.subplots(n_dims, T, figsize=(1.8*T, 2*n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :]

    for dim_idx in range(n_dims):
        for t_idx in range(T):
            ax = axes[dim_idx, t_idx]

            vals = coeffs[t_idx, :, dim_idx]
            ax.hist(vals, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles
            if dim_idx == 0:
                ax.set_title(f't={times[t_idx]:.3f}', fontsize=8)

            # Row labels
            if t_idx == 0:
                ax.set_ylabel(f'PC {dim_idx}', fontsize=9)

    fig.suptitle('PCA Coefficient Distributions (First 6 Components)', fontsize=14, y=0.995)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("PCA COEFFICIENT DISTRIBUTIONS")
print("="*60)

fig = plot_pca_coeff_distributions(coeff_marginals, times, n_dims=6)
save_path = output_dir / "pca_coeff_distributions.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## PCA Coefficient Statistics

# %%
def plot_pca_coeff_statistics(coeffs: np.ndarray, times: np.ndarray, n_dims: int = 8):
    """Plot mean and std of PCA coefficients over time.

    Args:
        coeffs: Shape (T, N, n_components)
        times: Time values
        n_dims: Number of dimensions to show
    """
    T, N, K = coeffs.shape
    n_dims = min(n_dims, K)

    fig, axes = plt.subplots(2, n_dims, figsize=(2.5*n_dims, 8))

    for dim_idx in range(n_dims):
        # Mean
        ax = axes[0, dim_idx]
        means = coeffs[:, :, dim_idx].mean(axis=1)  # (T,)
        ax.plot(times, means, 'o-', linewidth=2)
        ax.set_xlabel('Time t', fontsize=9)
        ax.set_ylabel(f'Mean (PC {dim_idx})', fontsize=9)
        ax.set_title(f'PC {dim_idx}: Mean', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Std
        ax = axes[1, dim_idx]
        stds = coeffs[:, :, dim_idx].std(axis=1)  # (T,)
        ax.plot(times, stds, 'o-', linewidth=2, color='orange')
        ax.set_xlabel('Time t', fontsize=9)
        ax.set_ylabel(f'Std (PC {dim_idx})', fontsize=9)
        ax.set_title(f'PC {dim_idx}: Std', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('PCA Coefficient Statistics Over Time', fontsize=14, y=0.995)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("PCA COEFFICIENT STATISTICS")
print("="*60)

fig = plot_pca_coeff_statistics(coeff_marginals, times, n_dims=8)
save_path = output_dir / "pca_coeff_statistics.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## PCA Variance Explained

# %%
# Plot cumulative variance explained
variance_ratio = pca_explained_variance / pca_explained_variance.sum()
cumsum_variance = np.cumsum(variance_ratio)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
ax = axes[0]
n_show = min(50, n_components)
ax.bar(range(n_show), variance_ratio[:n_show], alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Component Index', fontsize=12)
ax.set_ylabel('Variance Explained Ratio', fontsize=12)
ax.set_title(f'PCA Variance Explained (First {n_show} Components)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Cumulative variance
ax = axes[1]
ax.plot(range(n_components), cumsum_variance, linewidth=2, color='darkblue')
ax.axhline(y=0.99, color='r', linestyle='--', linewidth=1, label='99% threshold')
ax.axhline(y=0.999, color='orange', linestyle='--', linewidth=1, label='99.9% threshold')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
ax.set_title('Cumulative Variance Explained', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Find components for 99% and 99.9%
n_99 = np.argmax(cumsum_variance >= 0.99) + 1
n_999 = np.argmax(cumsum_variance >= 0.999) + 1
ax.annotate(f'{n_99} components', xy=(n_99, 0.99), xytext=(n_99 + 20, 0.97),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
ax.annotate(f'{n_999} components', xy=(n_999, 0.999), xytext=(n_999 + 20, 0.985),
            arrowprops=dict(arrowstyle='->', color='orange'), fontsize=9)

plt.tight_layout()
save_path = output_dir / "pca_variance_explained.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

print(f"\n99% variance explained by: {n_99} components")
print(f"99.9% variance explained by: {n_999} components")
print(f"Total components used: {n_components}")
print(f"Total variance explained: {cumsum_variance[-1]:.6f}")

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*70)
print("DATASET VISUALIZATION SUMMARY")
print("="*70)
print(f"\n📁 Output directory: {output_dir}\n")

print("✓ Generated Figures:")
print("  - raw_fields_across_time.png: Grid of raw fields at all time steps")
print("  - pca_reconstruction_error.png: RMSE of PCA reconstruction vs time")
print("  - raw_vs_pca_sample*.png: Side-by-side comparison for first 3 samples")
print("  - field_statistics.png: Mean, std, min, max over time")
print("  - field_distributions.png: Histograms of field values at each time")
print("  - pca_coeff_distributions.png: PCA coefficient distributions")
print("  - pca_coeff_statistics.png: PCA coefficient mean/std evolution")
print("  - pca_variance_explained.png: PCA variance explained analysis")

print("\n📊 Key Dataset Properties:")
print(f"  - Data generator: {data_generator}")
print(f"  - Resolution: {resolution}×{resolution}")
print(f"  - Number of samples: {n_samples}")
print(f"  - Number of time steps: {n_times}")
print(f"  - PCA components: {n_components} (explains {cumsum_variance[-1]*100:.2f}% variance)")
print(f"  - Scaling: {scale_mode} with epsilon={scaling_epsilon}")
print(f"  - Whitening: {'Yes' if is_whitened else 'No'}")

print("\n📈 PCA Quality:")
print(f"  - Overall reconstruction RMSE: {rmse_per_sample.mean():.4e}")
print(f"  - 99% variance requires: {n_99} components")
print(f"  - 99.9% variance requires: {n_999} components")

print("\n⚠️  Held-out Information:")
print(f"  - PCA was NOT fitted on:")
print(f"    • t=0 (microscale) - automatically excluded for tran_inclusion")
print(f"    • Indices {held_out_indices.tolist()}: times {held_out_times.tolist()}")

print("="*70)

# %%
