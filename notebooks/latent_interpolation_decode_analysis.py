# %%
# %% [markdown]
# # Latent Space Interpolation Decoding Analysis
#
# This notebook analyzes the diffeomorphic autoencoder's ability to decode intermediate
# interpolated points in latent space. We test whether smooth interpolation in latent
# space (z) corresponds to smooth, physically meaningful trajectories in ambient space.
#
# **Pipeline**:
# 1. Load diffeomorphic autoencoder and PCA data
# 2. Encode reference marginals to latent space
# 3. Create interpolated paths between marginal knots (pairwise/triplet)
# 4. Decode interpolated latent points at dense time grid
# 5. Lift to GRF spatial fields and analyze quality
# 6. Compare interpolated fields with reference data (where available)
#
# **Key Question**: Can the autoencoder successfully decode points along smooth
# interpolation paths in latent space, or does it only work well at the marginal knots?

# %%
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, CubicSpline

# Add parent directory to path
path_root = Path.cwd().parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from scripts.utils import get_device, build_zt
from scripts.latent_msbm_main import load_autoencoder, _load_data
from scripts.pca.pca_visualization_utils import parse_args_file
from scripts.images.field_visualization import (
    format_for_paper,
    reconstruct_fields_from_coefficients,
    plot_field_snapshots,
    plot_field_evolution_gif,
)
import argparse

print("Imports successful!")

# %% [markdown]
# ## Paper Formatting

# %%
format_for_paper()
print("✓ Applied paper formatting to all plots")

# %% [markdown]
# ## Configuration
#
# Set paths to your trained model and dataset.

# %%
# USER: Update these paths to your training output directory and data
msbm_dir = Path("/data1/jy384/research/MMSFM/results/2026-01-26T18-10-14-42")

# Dataset path (must contain raw_marginal_* for field reconstruction)
data_path = "./data/tran_inclusions.npz"

# Autoencoder checkpoint
ae_checkpoint = "results/2026-01-23T17-16-56-39/geodesic_autoencoder_best.pth"
ae_type = "diffeo"  # "geodesic" or "diffeo"

# Data split parameters
test_size = None  # Use training config default
seed = 42
nogpu = False

# Interpolation settings
interp_mode = "pairwise"  # "pairwise" or "triplet"
spline = "pchip"  # "linear", "pchip", or "cubic"
n_interp_steps = 100  # Number of intermediate points per interval

# Visualization settings
n_field_samples = 5  # Number of field samples to visualize
figsize_scale = 3.0

device = get_device(nogpu)
print(f"Using device: {device}")

# %% [markdown]
# ## Load Configuration from Training

# %%
# Load training configuration
train_cfg = {}
args_path = msbm_dir / "args.txt"
if args_path.exists():
    train_cfg = parse_args_file(args_path)
    print(f"Loaded training config from {args_path}")
else:
    print(f"Warning: args.txt not found in {msbm_dir}")

def _cfg(key: str, fallback):
    return train_cfg.get(key, fallback)

def _resolve_maybe(path_like):
    if path_like is None:
        return None
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return str(p)
    for base in (Path.cwd().parent, msbm_dir):
        cand = (base / p).resolve()
        if cand.exists():
            return str(cand)
    return str(p.resolve())

# Resolve paths
data_path = _resolve_maybe(_cfg("data_path", data_path))
ae_checkpoint = _resolve_maybe(_cfg("ae_checkpoint", ae_checkpoint))
ae_type = _cfg("ae_type", ae_type)
test_size = test_size if test_size is not None else _cfg("test_size", 0.2)

# AE parameters
latent_dim_override = _cfg("latent_dim_override", 376)
ae_ode_method = _cfg("ae_ode_method", "dopri5")

print(f"Data path: {data_path}")
print(f"AE checkpoint: {ae_checkpoint}")
print(f"AE type: {ae_type}")
print(f"Interpolation mode: {interp_mode}")
print(f"Spline type: {spline}")

# %% [markdown]
# ## Load PCA Metadata for Field Reconstruction

# %%
print("Loading PCA metadata from dataset...")
npz = np.load(data_path)

# Extract PCA reconstruction info
pca_components = npz["pca_components"]  # (n_components, data_dim)
pca_mean = npz["pca_mean"]  # (data_dim,)
pca_explained_variance = npz["pca_explained_variance"]  # (n_components,)
is_whitened = bool(npz["is_whitened"])
data_dim = int(npz["data_dim"])
resolution = int(np.sqrt(data_dim))

# Build pca_info dict for reconstruction functions
pca_info = {
    "mean": pca_mean,
    "components": pca_components,
    "explained_variance": pca_explained_variance,
    "is_whitened": is_whitened,
    "data_dim": data_dim,
}

print(f"✓ PCA metadata loaded:")
print(f"  - Resolution: {resolution}x{resolution}")
print(f"  - Data dimension: {data_dim}")
print(f"  - Number of PCA components: {pca_components.shape[0]}")
print(f"  - Is whitened: {is_whitened}")
print(f"  - Variance explained: {pca_explained_variance.sum():.4f}")

# %%
# Load raw marginal fields (ground truth for comparison)
print("\nLoading reference GRF fields from dataset...")

raw_keys = sorted(
    [k for k in npz.files if k.startswith("raw_marginal_")],
    key=lambda x: float(x.split("_")[-1]),
)
coeff_keys = sorted(
    [k for k in npz.files if k.startswith("marginal_") and not k.startswith("raw_")],
    key=lambda x: float(x.split("_")[-1]),
)

# Get time values from marginal keys
zt_values = np.array([float(k.split("_")[-1]) for k in coeff_keys])

# Load raw fields (N, data_dim) → reshape to (N, H, W)
raw_fields_list = []
for key in raw_keys:
    fields_flat = npz[key]  # (N, data_dim)
    fields_2d = fields_flat.reshape(-1, resolution, resolution)
    raw_fields_list.append(fields_2d)

raw_fields = np.stack(raw_fields_list, axis=0)  # (T, N, H, W)
print(f"✓ Raw fields shape: {raw_fields.shape} (T, N, H, W)")

# Load PCA coefficient marginals
coeff_marginals_list = [npz[key] for key in coeff_keys]
coeff_marginals = np.stack(coeff_marginals_list, axis=0)  # (T, N, n_components)
print(f"✓ PCA coefficient marginals shape: {coeff_marginals.shape}")

npz.close()

# %% [markdown]
# ## Load Data and Autoencoder

# %%
# Load data (uses same PCA coefficients, train/test split)
load_ns = argparse.Namespace(
    data_path=data_path,
    test_size=test_size,
    seed=seed,
    use_cache_data=_cfg("use_cache_data", False),
    selected_cache_path=_cfg("selected_cache_path", None),
    cache_dir=_cfg("cache_dir", None),
)
x_train, x_test, zt = _load_data(load_ns)
T = int(x_test.shape[0])
zt_np = np.asarray(zt, dtype=np.float32)

print(f"\nLoaded data:")
print(f"  T={T} marginals")
print(f"  x_train.shape={x_train.shape}")
print(f"  x_test.shape={x_test.shape}")
print(f"  zt: {zt_np}")

# Load autoencoder
encoder, decoder, ae_config = load_autoencoder(
    Path(ae_checkpoint),
    device,
    ae_type=ae_type,
    latent_dim_override=latent_dim_override,
    ode_method_override=ae_ode_method,
)
latent_dim = int(ae_config["latent_dim"])
print(f"\n✓ Loaded autoencoder: latent_dim={latent_dim}")

# %% [markdown]
# ## Encode Reference Marginals

# %%
print("Encoding reference marginals to latent space...")

# Use test set for analysis
x_ref = x_test  # (T, N, D)
n_samples = int(x_ref.shape[1])

# Encode each marginal
with torch.no_grad():
    latent_marginals = []
    for t_idx in range(T):
        x_t = torch.from_numpy(x_ref[t_idx]).float().to(device)  # (N, D)
        t_val = torch.full((n_samples,), zt_np[t_idx], device=device)
        z_t = encoder(x_t, t_val)  # (N, K)
        latent_marginals.append(z_t.cpu().numpy())

    latent_marginals = np.stack(latent_marginals, axis=0)  # (T, N, K)

print(f"✓ Latent marginals shape: {latent_marginals.shape}")
print(f"  Latent statistics: mean={latent_marginals.mean():.4f}, std={latent_marginals.std():.4f}")

# %% [markdown]
# ## Interpolation in Latent Space
#
# Create smooth interpolated paths between marginal knots using the specified spline method.

# %%
def create_pairwise_interpolation(
    latent_marginals: np.ndarray,
    zt: np.ndarray,
    n_steps_per_interval: int,
    spline_type: str = "pchip",
) -> tuple[np.ndarray, np.ndarray]:
    """Create pairwise interpolated trajectory in latent space.

    Args:
        latent_marginals: Latent marginal knots, shape (T, N, K)
        zt: Marginal time values, shape (T,)
        n_steps_per_interval: Number of interpolation steps per interval
        spline_type: "linear", "pchip", or "cubic"

    Returns:
        t_interp: Interpolated time values, shape (S,)
        z_interp: Interpolated latent points, shape (S, N, K)
    """
    T, N, K = latent_marginals.shape

    # Create dense time grid
    t_interp_parts = []
    z_interp_parts = []

    for i in range(T - 1):
        t_start, t_end = zt[i], zt[i + 1]
        t_interval = np.linspace(t_start, t_end, n_steps_per_interval + 1)

        # For each sample, interpolate in latent space
        z_interval = np.zeros((n_steps_per_interval + 1, N, K), dtype=np.float32)

        for sample_idx in range(N):
            # Get latent points for this sample at knots i and i+1
            z_start = latent_marginals[i, sample_idx]  # (K,)
            z_end = latent_marginals[i + 1, sample_idx]  # (K,)

            if spline_type == "linear":
                # Linear interpolation
                alpha = np.linspace(0, 1, n_steps_per_interval + 1)[:, None]
                z_interval[:, sample_idx, :] = (1 - alpha) * z_start + alpha * z_end
            else:
                # Spline interpolation (dimension-wise)
                for dim in range(K):
                    knot_times = np.array([t_start, t_end])
                    knot_values = np.array([z_start[dim], z_end[dim]])

                    if spline_type == "pchip":
                        interp = PchipInterpolator(knot_times, knot_values)
                    elif spline_type == "cubic":
                        interp = CubicSpline(knot_times, knot_values)
                    else:
                        raise ValueError(f"Unknown spline type: {spline_type}")

                    z_interval[:, sample_idx, dim] = interp(t_interval)

        # Avoid duplicates at interval boundaries
        if i > 0:
            t_interval = t_interval[1:]
            z_interval = z_interval[1:]

        t_interp_parts.append(t_interval)
        z_interp_parts.append(z_interval)

    t_interp = np.concatenate(t_interp_parts)
    z_interp = np.concatenate(z_interp_parts, axis=0)

    return t_interp, z_interp


def create_triplet_interpolation(
    latent_marginals: np.ndarray,
    zt: np.ndarray,
    n_steps_per_interval: int,
    spline_type: str = "pchip",
) -> tuple[np.ndarray, np.ndarray]:
    """Create triplet interpolated trajectory in latent space.

    Uses three consecutive knots for each interval (except boundaries).

    Args:
        latent_marginals: Latent marginal knots, shape (T, N, K)
        zt: Marginal time values, shape (T,)
        n_steps_per_interval: Number of interpolation steps per interval
        spline_type: "pchip" or "cubic" (linear not supported for triplet)

    Returns:
        t_interp: Interpolated time values, shape (S,)
        z_interp: Interpolated latent points, shape (S, N, K)
    """
    T, N, K = latent_marginals.shape

    if spline_type == "linear":
        raise ValueError("Linear interpolation not supported for triplet mode")

    t_interp_parts = []
    z_interp_parts = []

    for i in range(T - 1):
        t_start, t_end = zt[i], zt[i + 1]
        t_interval = np.linspace(t_start, t_end, n_steps_per_interval + 1)

        z_interval = np.zeros((n_steps_per_interval + 1, N, K), dtype=np.float32)

        for sample_idx in range(N):
            # Determine knot indices for triplet
            if i == 0:
                # First interval: use knots 0, 1, 2
                knot_indices = [0, 1, 2] if T >= 3 else [0, 1]
            elif i == T - 2:
                # Last interval: use knots T-3, T-2, T-1
                knot_indices = [T - 3, T - 2, T - 1] if T >= 3 else [T - 2, T - 1]
            else:
                # Middle interval: use knots i-1, i, i+1, i+2
                knot_indices = [i - 1, i, i + 1, i + 2] if i + 2 < T else [i - 1, i, i + 1]

            knot_times = zt[knot_indices]

            for dim in range(K):
                knot_values = latent_marginals[knot_indices, sample_idx, dim]

                if spline_type == "pchip":
                    interp = PchipInterpolator(knot_times, knot_values)
                elif spline_type == "cubic":
                    interp = CubicSpline(knot_times, knot_values)
                else:
                    raise ValueError(f"Unknown spline type: {spline_type}")

                z_interval[:, sample_idx, dim] = interp(t_interval)

        if i > 0:
            t_interval = t_interval[1:]
            z_interval = z_interval[1:]

        t_interp_parts.append(t_interval)
        z_interp_parts.append(z_interval)

    t_interp = np.concatenate(t_interp_parts)
    z_interp = np.concatenate(z_interp_parts, axis=0)

    return t_interp, z_interp


# %%
print(f"Creating {interp_mode} interpolation with {spline} spline...")

if interp_mode == "pairwise":
    t_interp, z_interp = create_pairwise_interpolation(
        latent_marginals, zt_np, n_interp_steps, spline
    )
elif interp_mode == "triplet":
    t_interp, z_interp = create_triplet_interpolation(
        latent_marginals, zt_np, n_interp_steps, spline
    )
else:
    raise ValueError(f"Unknown interp_mode: {interp_mode}")

print(f"✓ Interpolated trajectory created:")
print(f"  Time points: {len(t_interp)}")
print(f"  Latent shape: {z_interp.shape} (S, N, K)")
print(f"  Time range: [{t_interp.min():.4f}, {t_interp.max():.4f}]")

# %% [markdown]
# ## Decode Interpolated Latent Points
#
# Decode the dense interpolated latent trajectory back to ambient (PCA coefficient) space.

# %%
print("Decoding interpolated latent points to ambient space...")

S, N, K = z_interp.shape
ambient_interp = np.zeros((S, N, pca_components.shape[0]), dtype=np.float32)

with torch.no_grad():
    batch_size = 256
    for s in range(S):
        z_s = torch.from_numpy(z_interp[s]).float().to(device)  # (N, K)
        t_s = torch.full((N,), t_interp[s], device=device)

        # Decode in batches if needed
        if N <= batch_size:
            x_s = decoder(z_s, t_s)  # (N, D)
            ambient_interp[s] = x_s.cpu().numpy()
        else:
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                z_batch = z_s[i:end_i]
                t_batch = t_s[i:end_i]
                x_batch = decoder(z_batch, t_batch)
                ambient_interp[s, i:end_i] = x_batch.cpu().numpy()

        if (s + 1) % 50 == 0 or s == S - 1:
            print(f"  Decoded {s + 1}/{S} time steps")

print(f"✓ Decoded ambient trajectory shape: {ambient_interp.shape}")

# %% [markdown]
# ## Lift to GRF Field Space

# %%
print("Lifting interpolated PCA coefficients to GRF field space...")

# Lift interpolated coefficients → fields
fields_interp = reconstruct_fields_from_coefficients(ambient_interp, pca_info, resolution)
print(f"✓ Interpolated fields shape: {fields_interp.shape} (S, N, H, W)")

# Also lift reference marginals for comparison
fields_ref_marginals = reconstruct_fields_from_coefficients(x_ref, pca_info, resolution)
print(f"✓ Reference marginal fields shape: {fields_ref_marginals.shape} (T, N, H, W)")

# %% [markdown]
# ## Sample Interpolated Trajectory at Marginal Times

# %%
def sample_at_times(traj: np.ndarray, t_traj: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Sample trajectory at target times via nearest neighbor."""
    sampled = []
    for t in t_target:
        idx = np.argmin(np.abs(t_traj - t))
        sampled.append(traj[idx])
    return np.stack(sampled, axis=0)


# Sample interpolated trajectory at marginal times for comparison
fields_interp_at_marginals = sample_at_times(fields_interp, t_interp, zt_np)
print(f"✓ Interpolated fields at marginals shape: {fields_interp_at_marginals.shape}")

# %% [markdown]
# ## Analysis: Field Quality at Marginal Times

# %%
def compute_field_rmse(ref: np.ndarray, gen: np.ndarray) -> dict:
    """Compute RMSE between reference and generated fields."""
    T, N, H, W = ref.shape
    diff = ref - gen
    rmse_per_sample_time = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))  # (T, N)

    return {
        'rmse_mean_per_time': rmse_per_sample_time.mean(axis=1),  # (T,)
        'rmse_std_per_time': rmse_per_sample_time.std(axis=1),  # (T,)
        'overall_rmse': np.mean(rmse_per_sample_time),
    }


# Compare interpolated fields at marginals vs reference marginals
errors_at_marginals = compute_field_rmse(fields_ref_marginals, fields_interp_at_marginals)

print("\n" + "="*70)
print("INTERPOLATION DECODING ERROR AT MARGINAL TIMES")
print("="*70)
print(f"Overall RMSE: {errors_at_marginals['overall_rmse']:.4e}")
print("\nPer-time breakdown:")
print(f"{'Time':>8}  {'RMSE (mean ± std)':>25}")
print("-" * 40)
for t_idx, t_val in enumerate(zt_np):
    rmse_mean = errors_at_marginals['rmse_mean_per_time'][t_idx]
    rmse_std = errors_at_marginals['rmse_std_per_time'][t_idx]
    print(f"{t_val:>8.4f}  {rmse_mean:.3e} ± {rmse_std:.3e}")

# %% [markdown]
# ## Analysis: Field Quality at Intermediate Times
#
# Check if fields at intermediate (non-marginal) times are smooth and reasonable.

# %%
# Find intermediate time indices (not at marginals)
intermediate_mask = np.ones(len(t_interp), dtype=bool)
for t_marg in zt_np:
    idx = np.argmin(np.abs(t_interp - t_marg))
    intermediate_mask[idx] = False

t_intermediate = t_interp[intermediate_mask]
fields_intermediate = fields_interp[intermediate_mask]

print(f"Intermediate time points: {len(t_intermediate)} (total: {len(t_interp)}, marginals: {T})")

# Compute statistics
print("\n" + "="*70)
print("FIELD STATISTICS COMPARISON")
print("="*70)

print("\nAt marginal times:")
for t_idx, t_val in enumerate(zt_np):
    f = fields_ref_marginals[t_idx]
    print(f"  t={t_val:.4f}: mean={f.mean():.4f}, std={f.std():.4f}, range=[{f.min():.4f}, {f.max():.4f}]")

print("\nAt intermediate times (interpolated):")
# Sample a few intermediate times
n_show = min(10, len(t_intermediate))
indices = np.linspace(0, len(t_intermediate) - 1, n_show, dtype=int)
for idx in indices:
    t_val = t_intermediate[idx]
    f = fields_intermediate[idx]
    print(f"  t={t_val:.4f}: mean={f.mean():.4f}, std={f.std():.4f}, range=[{f.min():.4f}, {f.max():.4f}]")

# %% [markdown]
# ## Visualization: Field Evolution

# %%
# Create output directory
eval_dir = msbm_dir / "eval" / "latent_interpolation"
eval_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {eval_dir}")

# Mock run object for saving
class LocalRun:
    def log(self, *args, **kwargs):
        pass

local_run = LocalRun()

# %%
# Visualize a single sample's field evolution over time
sample_idx = 0

# Select time indices to visualize
if len(t_interp) > 20:
    vis_indices = np.linspace(0, len(t_interp) - 1, 20, dtype=int)
else:
    vis_indices = np.arange(len(t_interp))

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.ravel()

for i, t_idx in enumerate(vis_indices):
    ax = axes[i]
    field = fields_interp[t_idx, sample_idx]
    im = ax.imshow(field, cmap='viridis', origin='lower')
    ax.set_title(f't={t_interp[t_idx]:.4f}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f'Interpolated Field Evolution (Sample {sample_idx})', fontsize=14, y=0.995)
plt.tight_layout()
save_path = eval_dir / f"field_evolution_interpolated_sample{sample_idx}.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Comparison: Marginals vs Interpolated

# %%
# Compare reference marginals with interpolated fields at marginal times
sample_idx = 0

fig, axes = plt.subplots(3, T, figsize=(3 * T, 9))

vmin = min(fields_ref_marginals[:, sample_idx].min(), fields_interp_at_marginals[:, sample_idx].min())
vmax = max(fields_ref_marginals[:, sample_idx].max(), fields_interp_at_marginals[:, sample_idx].max())

for t_idx in range(T):
    # Reference
    ax = axes[0, t_idx]
    ax.imshow(fields_ref_marginals[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f't={zt_np[t_idx]:.3f}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('Reference', fontsize=11, fontweight='bold')

    # Interpolated
    ax = axes[1, t_idx]
    ax.imshow(fields_interp_at_marginals[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('Interpolated', fontsize=11)

    # Difference
    diff = fields_ref_marginals[t_idx, sample_idx] - fields_interp_at_marginals[t_idx, sample_idx]
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    ax = axes[2, t_idx]
    ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('Difference', fontsize=11)
    rmse = np.sqrt(np.mean(diff ** 2))
    ax.text(0.5, -0.15, f'RMSE: {rmse:.2e}', transform=ax.transAxes, ha='center', fontsize=8)

plt.suptitle(f'Reference vs Interpolated at Marginals (Sample {sample_idx})', fontsize=14, y=0.98)
plt.tight_layout()
save_path = eval_dir / f"marginal_comparison_sample{sample_idx}.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Smoothness Analysis
#
# Analyze the smoothness of the interpolated trajectory by examining temporal derivatives.

# %%
def compute_temporal_gradient(fields: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute temporal gradient magnitude of fields.

    Args:
        fields: Field trajectory, shape (S, N, H, W)
        times: Time values, shape (S,)

    Returns:
        Gradient magnitude, shape (S-1, N)
    """
    S, N, H, W = fields.shape
    grads = np.zeros((S - 1, N), dtype=np.float32)

    for i in range(S - 1):
        dt = times[i + 1] - times[i]
        df = fields[i + 1] - fields[i]  # (N, H, W)
        grad_norm = np.sqrt(np.mean(df ** 2, axis=(1, 2))) / dt  # (N,)
        grads[i] = grad_norm

    return grads


# Compute temporal gradients
grads_interp = compute_temporal_gradient(fields_interp, t_interp)
print(f"Temporal gradient shape: {grads_interp.shape}")

# Plot temporal gradient over time
fig, ax = plt.subplots(figsize=(12, 6))

# Average across samples
grad_mean = grads_interp.mean(axis=1)
grad_std = grads_interp.std(axis=1)
t_grad = (t_interp[:-1] + t_interp[1:]) / 2

ax.plot(t_grad, grad_mean, 'b-', linewidth=2, label='Mean gradient')
ax.fill_between(t_grad, grad_mean - grad_std, grad_mean + grad_std, alpha=0.3, color='blue')

# Mark marginal times
for t_marg in zt_np:
    ax.axvline(t_marg, color='red', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('Temporal Gradient Magnitude', fontsize=12)
ax.set_title(f'Field Temporal Smoothness ({interp_mode} {spline})', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = eval_dir / "temporal_smoothness.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Latent Space Analysis

# %%
from sklearn.decomposition import PCA as SklearnPCA

# Project latent trajectories to 2D for visualization
latent_all = z_interp.reshape(-1, K)  # (S*N, K)
pca = SklearnPCA(n_components=2)
pca.fit(latent_all)

# Project marginal knots and interpolated points
latent_marginals_flat = latent_marginals.reshape(-1, K)
marginals_proj = pca.transform(latent_marginals_flat).reshape(T, N, 2)
interp_proj = pca.transform(latent_all).reshape(S, N, 2)

# Plot a few sample trajectories
fig, ax = plt.subplots(figsize=(10, 8))

# Plot marginal knots
colors = plt.cm.viridis(np.linspace(0, 1, T))
for t_idx in range(T):
    ax.scatter(marginals_proj[t_idx, :, 0], marginals_proj[t_idx, :, 1],
               c=[colors[t_idx]], alpha=0.5, s=50, label=f't={zt_np[t_idx]:.3f}')

# Plot interpolated trajectories for a few samples
n_traj_show = min(5, N)
for sample_idx in range(n_traj_show):
    traj = interp_proj[:, sample_idx, :]  # (S, 2)
    ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, marker='x', zorder=10)

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title(f'Latent Space Interpolation ({interp_mode} {spline})', fontsize=14)
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = eval_dir / "latent_trajectories_pca.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Decode-Encode Round-Trip Analysis
#
# Test if re-encoding the decoded interpolated fields recovers the original latent points.

# %%
print("Testing decode-encode round-trip consistency...")

# Pick a subset of intermediate time points
n_test_times = min(10, len(t_intermediate))
test_indices = np.linspace(0, len(t_intermediate) - 1, n_test_times, dtype=int)

roundtrip_errors = []

with torch.no_grad():
    for idx in test_indices:
        # Get interpolated latent point
        t_val = t_intermediate[idx]
        z_orig = z_interp[intermediate_mask][idx]  # (N, K)

        # Decode
        z_torch = torch.from_numpy(z_orig).float().to(device)
        t_torch = torch.full((N,), t_val, device=device)
        x_decoded = decoder(z_torch, t_torch)  # (N, D)

        # Re-encode
        z_reencoded = encoder(x_decoded, t_torch)  # (N, K)

        # Compute error
        z_reencoded_np = z_reencoded.cpu().numpy()
        error = np.sqrt(np.mean((z_orig - z_reencoded_np) ** 2, axis=1))  # (N,)
        roundtrip_errors.append(error.mean())

roundtrip_errors = np.array(roundtrip_errors)

print(f"\n{'='*70}")
print("DECODE-ENCODE ROUND-TRIP ANALYSIS")
print(f"{'='*70}")
print(f"Mean round-trip error: {roundtrip_errors.mean():.4e}")
print(f"Std round-trip error: {roundtrip_errors.std():.4e}")
print(f"Max round-trip error: {roundtrip_errors.max():.4e}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(roundtrip_errors)), roundtrip_errors, 'o-', linewidth=2)
ax.set_xlabel('Test Point Index', fontsize=12)
ax.set_ylabel('Round-trip RMSE', fontsize=12)
ax.set_title('Decode-Encode Round-Trip Error at Intermediate Times', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = eval_dir / "roundtrip_error.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*70)
print("LATENT INTERPOLATION DECODING ANALYSIS SUMMARY")
print("="*70)

print(f"\n📁 Output directory: {eval_dir}\n")

print("✓ Configuration:")
print(f"  - Autoencoder type: {ae_type}")
print(f"  - Latent dimension: {latent_dim}")
print(f"  - Interpolation mode: {interp_mode}")
print(f"  - Spline type: {spline}")
print(f"  - Steps per interval: {n_interp_steps}")

print("\n✓ Data:")
print(f"  - Marginals (T): {T}")
print(f"  - Samples (N): {N}")
print(f"  - Interpolated time points: {len(t_interp)}")
print(f"  - Intermediate time points: {len(t_intermediate)}")

print("\n✓ Key Findings:")
print(f"  - Error at marginal times: {errors_at_marginals['overall_rmse']:.4e}")
print(f"  - Mean temporal gradient: {grad_mean.mean():.4e}")
print(f"  - Round-trip error (mean): {roundtrip_errors.mean():.4e}")

print("\n✓ Generated Plots:")
print("  - field_evolution_interpolated_sample*.png")
print("  - marginal_comparison_sample*.png")
print("  - temporal_smoothness.png")
print("  - latent_trajectories_pca.png")
print("  - roundtrip_error.png")

print("\n📊 Interpretation:")
if errors_at_marginals['overall_rmse'] < 1e-2:
    print("  ✓ EXCELLENT: Interpolation decoding is highly accurate at marginals")
elif errors_at_marginals['overall_rmse'] < 5e-2:
    print("  ✓ GOOD: Interpolation decoding is reasonably accurate at marginals")
else:
    print("  ⚠ WARNING: Significant error at marginal times - check interpolation/AE")

if roundtrip_errors.mean() < 1e-3:
    print("  ✓ EXCELLENT: Decode-encode round-trip is consistent")
elif roundtrip_errors.mean() < 1e-2:
    print("  ✓ GOOD: Decode-encode round-trip has acceptable error")
else:
    print("  ⚠ WARNING: High round-trip error - AE may not preserve latent structure")

grad_variation = grad_std.mean() / (grad_mean.mean() + 1e-10)
if grad_variation < 0.5:
    print("  ✓ SMOOTH: Temporal evolution is smooth and consistent")
else:
    print("  ⚠ NOTICE: High temporal gradient variation - check smoothness")

print("="*70)

# %%
