#!/usr/bin/env python3
"""Score Model Diagnostics Script

This script generates diagnostic visualizations for the score model to help
understand why backward SDE trajectories diverge from the data distribution.

Based on configurations from visualize_conditional_flow_paths_diffeo.ipynb

Usage:
    python score_diagnostics.py --help
    python score_diagnostics.py --flow_checkpoint PATH --score_checkpoint PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mmsfm.models import TimeFiLMMLP
from scripts.latent_flow_main import (
    ExponentialContractingSchedule,
    load_autoencoder,
    LatentFlowMatcher,
    plot_score_vector_field,
    plot_velocity_vs_score_magnitude,
)
from scripts.pca_precomputed_utils import load_pca_data
from scripts.utils import build_zt, get_device


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate score model diagnostic visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with custom checkpoints
  python score_diagnostics.py \\
    --flow_checkpoint results/run1/latent_flow_model.pth \\
    --score_checkpoint results/run1/score_model.pth

  # Full customization
  python score_diagnostics.py \\
    --ae_checkpoint results/run1/geodesic_autoencoder.pth \\
    --flow_checkpoint results/run1/latent_flow_model.pth \\
    --score_checkpoint results/run1/score_model.pth \\
    --output_dir results/my_diagnostics \\
    --sigma_0 0.1 --decay_rate 2.0 \\
    --grid_size 30
        """
    )

    # Paths
    parser.add_argument(
        "--ae_checkpoint",
        type=Path,
        default=REPO_ROOT / "results/2026-01-21T23-15-12-14/geodesic_autoencoder.pth",
        help="Path to autoencoder checkpoint",
    )
    parser.add_argument(
        "--flow_checkpoint",
        type=Path,
        default=REPO_ROOT / "results/2026-01-22T19-17-58-10/latent_flow_model_best_traj.pth",
        help="Path to flow model checkpoint",
    )
    parser.add_argument(
        "--score_checkpoint",
        type=Path,
        default=REPO_ROOT / "results/2026-01-22T19-17-58-10/score_model_best_score.pth",
        help="Path to score model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=REPO_ROOT / "data/tran_inclusions.npz",
        help="Path to data file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results/score_diagnostics",
        help="Output directory for diagnostic plots",
    )

    # Model config
    parser.add_argument(
        "--ae_type",
        type=str,
        default="diffeo",
        choices=["geodesic", "diffeo"],
        help="Autoencoder type",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[1024, 1024, 1024],
        help="Hidden layer dimensions for flow/score models",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=32,
        help="Time embedding dimension",
    )

    # Noise schedule
    parser.add_argument(
        "--sigma_0",
        type=float,
        default=0.05,
        help="Initial noise level",
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=3.0,
        help="Noise decay rate",
    )

    # Training config
    parser.add_argument(
        "--score_param",
        type=str,
        default="scaled",
        choices=["scaled", "raw"],
        help="Score parameterization (must match training)",
    )
    parser.add_argument(
        "--interp_mode",
        type=str,
        default="pairwise",
        choices=["pairwise", "triplet"],
        help="Interpolation mode",
    )
    parser.add_argument(
        "--spline",
        type=str,
        default="pchip",
        choices=["linear", "pchip", "cubic"],
        help="Spline type",
    )

    # Visualization parameters
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Latent dimensions to visualize (e.g., 0 1)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="Grid resolution for vector fields",
    )
    parser.add_argument(
        "--n_magnitude_samples",
        type=int,
        default=1000,
        help="Number of samples for magnitude comparison",
    )

    # Other
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test data split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )

    return parser.parse_args()


# ============================================================================
# Configuration
# ============================================================================

args = parse_args()

# Extract configuration from args
AE_TYPE = args.ae_type
DATA_PATH = args.data_path
AE_CHECKPOINT = args.ae_checkpoint
FLOW_CHECKPOINT = args.flow_checkpoint
SCORE_CHECKPOINT = args.score_checkpoint
OUTPUT_DIR = args.output_dir

HIDDEN_DIMS = args.hidden_dims
TIME_DIM = args.time_dim

SIGMA_0 = args.sigma_0
DECAY_RATE = args.decay_rate

SCORE_PARAMETERIZATION = args.score_param
INTERP_MODE = args.interp_mode
SPLINE = args.spline

DIMS = tuple(args.dims)
GRID_SIZE = args.grid_size
N_MAGNITUDE_SAMPLES = args.n_magnitude_samples

TEST_SIZE = args.test_size
SEED = args.seed
USE_GPU = not args.no_gpu

# ============================================================================

print(f"Repository root: {REPO_ROOT}")
print("\nConfiguration:")
print(f"  AE checkpoint: {AE_CHECKPOINT}")
print(f"  Flow checkpoint: {FLOW_CHECKPOINT}")
print(f"  Score checkpoint: {SCORE_CHECKPOINT}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Noise schedule: sigma_0={SIGMA_0}, decay_rate={DECAY_RATE}")
print(f"  Score parameterization: {SCORE_PARAMETERIZATION}")
print(f"  Visualization dims: {DIMS}")

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
device_str = get_device(not USE_GPU)
device = torch.device(device_str)
print(f"Using device: {device}")

# Output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading PCA data...")
data_tuple = load_pca_data(
    str(DATA_PATH),
    TEST_SIZE,
    SEED,
    return_indices=True,
    return_full=True,
    return_times=True,
)
data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

# Drop first marginal
if len(full_marginals) > 0:
    data = data[1:]
    testdata = testdata[1:]
    full_marginals = full_marginals[1:]
    if marginal_times is not None:
        marginal_times = marginal_times[1:]

# Build time array
marginals = list(range(len(full_marginals)))
zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
T = len(zt)

# Stack data
frames = np.stack(full_marginals, axis=0).astype(np.float32)
x_train = frames[:, train_idx, :].astype(np.float32)
x_test = frames[:, test_idx, :].astype(np.float32)

print(f"Data shapes: x_train={x_train.shape}, x_test={x_test.shape}")
print(f"Time points: T={T}, zt={zt}")

# ============================================================================
# Load Autoencoder
# ============================================================================

print(f"\nLoading {AE_TYPE} autoencoder from {AE_CHECKPOINT}...")
encoder, decoder, ae_config = load_autoencoder(
    AE_CHECKPOINT,
    device_str,
    ae_type=AE_TYPE,
)
latent_dim = ae_config["latent_dim"]
print(f"Autoencoder type: {ae_config.get('type', 'unknown')}")
print(f"Latent dimension: {latent_dim}")
print(f"Ambient dimension: {ae_config['ambient_dim']}")

# ============================================================================
# Load Flow Models
# ============================================================================

print(f"\nLoading flow models...")

def load_flow_models(
    flow_checkpoint: Path,
    score_checkpoint: Path,
    latent_dim: int,
    hidden_dims: list[int],
    time_dim: int,
    device: str,
) -> tuple[nn.Module, nn.Module]:
    """Load trained flow and score models."""
    velocity_model = TimeFiLMMLP(
        dim_x=latent_dim,
        dim_out=latent_dim,
        w=hidden_dims[0] if hidden_dims else 256,
        depth=len(hidden_dims),
        t_dim=time_dim,
    ).to(device)

    score_model = TimeFiLMMLP(
        dim_x=latent_dim,
        dim_out=latent_dim,
        w=hidden_dims[0] if hidden_dims else 256,
        depth=len(hidden_dims),
        t_dim=time_dim,
    ).to(device)

    velocity_model.load_state_dict(torch.load(flow_checkpoint, map_location=device))
    score_model.load_state_dict(torch.load(score_checkpoint, map_location=device))

    velocity_model.eval()
    score_model.eval()

    for p in velocity_model.parameters():
        p.requires_grad = False
    for p in score_model.parameters():
        p.requires_grad = False

    return velocity_model, score_model

velocity_model, score_model = load_flow_models(
    FLOW_CHECKPOINT,
    SCORE_CHECKPOINT,
    latent_dim,
    HIDDEN_DIMS,
    TIME_DIM,
    device_str,
)
print("Flow models loaded successfully")

# ============================================================================
# Build Noise Schedule
# ============================================================================

schedule = ExponentialContractingSchedule(
    sigma_0=SIGMA_0,
    decay_rate=DECAY_RATE,
)
print(f"\nNoise schedule: sigma_0={SIGMA_0}, decay_rate={DECAY_RATE}")
print(f"  sigma(0) = {schedule.sigma_t(torch.tensor(0.0)).item():.4f}")
print(f"  sigma(1) = {schedule.sigma_t(torch.tensor(1.0)).item():.4f}")

# ============================================================================
# Create Flow Matcher and Encode Marginals
# ============================================================================

print("\nCreating flow matcher and encoding marginals...")
flow_matcher = LatentFlowMatcher(
    encoder=encoder,
    decoder=decoder,
    schedule=schedule,
    zt=zt,
    interp_mode=INTERP_MODE,
    spline=SPLINE,
    score_parameterization=SCORE_PARAMETERIZATION,
    device=device_str,
)

flow_matcher.encode_marginals(x_train, x_test)
print(f"Encoded marginals: train={flow_matcher.latent_train.shape}, test={flow_matcher.latent_test.shape}")

# ============================================================================
# Diagnostic Visualizations
# ============================================================================

print("\n" + "="*70)
print("Generating Score Model Diagnostics")
print("="*70)

# 1. Score Vector Field
print("\n1. Plotting score vector field...")
plot_score_vector_field(
    score_model=score_model,
    schedule=schedule,
    latent_data=flow_matcher.latent_train.cpu().numpy(),
    save_path=OUTPUT_DIR / "score_vector_field.png",
    device=device_str,
    grid_size=GRID_SIZE,
    t_values=[0.0, 0.5, 1.0],
    dims=DIMS,
    score_parameterization=SCORE_PARAMETERIZATION,
    run=None,
)
print(f"  Saved: {OUTPUT_DIR / 'score_vector_field.png'}")

# 2. Velocity vs Score Magnitude Comparison
print("\n2. Plotting velocity vs score magnitude comparison...")
plot_velocity_vs_score_magnitude(
    velocity_model=velocity_model,
    score_model=score_model,
    schedule=schedule,
    latent_data=flow_matcher.latent_train.cpu().numpy(),
    save_path=OUTPUT_DIR / "velocity_vs_score_magnitude.png",
    device=device_str,
    n_samples=N_MAGNITUDE_SAMPLES,
    score_parameterization=SCORE_PARAMETERIZATION,
    run=None,
)
print(f"  Saved: {OUTPUT_DIR / 'velocity_vs_score_magnitude.png'}")

# 3. Backward SDE Drift Field Comparison
print("\n3. Plotting backward SDE drift field...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

d0, d1 = DIMS
K = latent_dim

# Compute bounds from data
all_latent = flow_matcher.latent_train.cpu().numpy().reshape(-1, K)
x_min, x_max = all_latent[:, d0].min(), all_latent[:, d0].max()
y_min, y_max = all_latent[:, d1].min(), all_latent[:, d1].max()
margin = 0.1
x_min -= margin * (x_max - x_min)
x_max += margin * (x_max - x_min)
y_min -= margin * (y_max - y_min)
y_max += margin * (y_max - y_min)

t_values_field = [0.2, 0.5, 0.8]

velocity_model.eval()
score_model.eval()

for ax, t_val in zip(axes, t_values_field):
    # Create grid
    xx = np.linspace(x_min, x_max, GRID_SIZE)
    yy = np.linspace(y_min, y_max, GRID_SIZE)
    XX, YY = np.meshgrid(xx, yy)

    # Build full latent vectors (set other dims to mean)
    mean_other = all_latent.mean(axis=0)
    grid_points = np.zeros((GRID_SIZE * GRID_SIZE, K), dtype=np.float32)
    grid_points[:, d0] = XX.ravel()
    grid_points[:, d1] = YY.ravel()
    for d in range(K):
        if d not in DIMS:
            grid_points[:, d] = mean_other[d]

    # Evaluate velocity and score
    with torch.no_grad():
        y_grid = torch.from_numpy(grid_points).float().to(device)
        t_grid = torch.full((len(grid_points),), t_val, device=device)

        v = velocity_model(y_grid, t=t_grid)
        s_theta = score_model(y_grid, t=t_grid)

        # Convert score to backward SDE drift contribution
        if SCORE_PARAMETERIZATION == "scaled":
            # s_scaled is already (g^2/2)*s_raw, add directly
            score_term = s_theta
        else:
            # Multiply by (g^2/2)
            g_t = schedule.sigma_t(t_grid).unsqueeze(-1)
            score_term = (g_t ** 2 / 2.0) * s_theta

        # Backward drift: -v + score_term
        backward_drift = -v + score_term

    U_backward = backward_drift[:, d0].cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    V_backward = backward_drift[:, d1].cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    magnitude = np.sqrt(U_backward**2 + V_backward**2)

    # Plot backward drift field
    quiv = ax.quiver(XX, YY, U_backward, V_backward, magnitude, alpha=0.7, cmap='coolwarm')
    plt.colorbar(quiv, ax=ax, label='Drift Magnitude')

    # Overlay data distribution at nearest marginal
    t_idx = np.argmin(np.abs(zt - t_val))
    ax.scatter(
        flow_matcher.latent_train[t_idx, ::20, d0].cpu().numpy(),
        flow_matcher.latent_train[t_idx, ::20, d1].cpu().numpy(),
        c='black', alpha=0.2, s=5, label='Data'
    )

    ax.set_xlabel(f"Latent dim {d0}")
    ax.set_ylabel(f"Latent dim {d1}")
    ax.set_title(f"Backward SDE Drift (-v + score) at t={t_val:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Backward SDE Drift Field\n(Should point toward earlier marginals)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backward_sde_drift_field.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR / 'backward_sde_drift_field.png'}")

# 4. Score Model Training Coverage Analysis
print("\n4. Analyzing score model training coverage...")
print("   This answers: 'How far is actual data from where the score model was trained?'")

# ============================================================================
# TRAINING COVERAGE LOGIC:
#
# The score model was trained on samples from p(y_t | z0, z1):
#   1. Sample pair (z0, z1) from consecutive marginals
#   2. Sample time t uniformly
#   3. Compute interpolation mean: mu_t = (1-alpha)*z0 + alpha*z1
#   4. Add Gaussian noise: y_t = mu_t + sigma_t * epsilon
#   5. Train score to predict -epsilon/sigma_t (denoising direction)
#
# These y_t points form a NARROW TUBE around the interpolation paths because:
#   - They only connect consecutive marginals (not all pairs)
#   - sigma_t is small (sigma_0=0.05), so noise spread is tiny
#   - Result: training samples don't fill the full latent space
#
# We compare:
#   - Where the score saw training data (y_t samples)
#   - Where the actual marginals lie (encoded data)
#   - If distance is large → score must extrapolate during backward SDE
# ============================================================================

# Sample training points using THE EXACT SAME PROCEDURE as during training
N_COVERAGE = 2000
print(f"   Sampling {N_COVERAGE} training points using flow matcher...")
t_batch, y_t_batch, u_t_batch, eps_batch = flow_matcher.sample_location_and_conditional_flow(
    N_COVERAGE, return_noise=True
)
print(f"   These are the exact positions where score model saw training examples.")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Training coverage in latent space
ax = axes[0]
colors_marginal = plt.cm.viridis(np.linspace(0, 1, len(zt)))

# Plot actual marginal distributions (faint)
for t_idx in range(len(zt)):
    ax.scatter(
        flow_matcher.latent_train[t_idx, :, d0].cpu().numpy(),
        flow_matcher.latent_train[t_idx, :, d1].cpu().numpy(),
        c=[colors_marginal[t_idx]], alpha=0.1, s=3,
        label=f"Marginal t={zt[t_idx]:.2f}" if t_idx % 2 == 0 else None
    )

# Plot training samples (bright)
scatter = ax.scatter(
    y_t_batch[:, d0].cpu().numpy(),
    y_t_batch[:, d1].cpu().numpy(),
    c=t_batch.cpu().numpy(),
    cmap='plasma',
    s=15,
    alpha=0.5,
    edgecolor='black',
    linewidth=0.3,
    label='Training samples y_t'
)
plt.colorbar(scatter, ax=ax, label='time t')
ax.set_xlabel(f"Latent dim {d0}")
ax.set_ylabel(f"Latent dim {d1}")
ax.set_title("Score Model Training Coverage\n(Do training samples overlap marginals?)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: Distance from training manifold
ax = axes[1]

# For each marginal data point, compute minimum distance to ANY training sample
print("   Computing distances from marginal data to training samples...")
distances = []
for t_idx in range(len(zt)):
    marginal_data = flow_matcher.latent_train[t_idx].cpu().numpy()  # (N, K) - actual data

    # Subsample marginal data for efficiency (200 points per marginal)
    n_subsample = min(200, marginal_data.shape[0])
    subsample_idx = np.random.choice(marginal_data.shape[0], size=n_subsample, replace=False)
    subsample = marginal_data[subsample_idx]

    # Compute pairwise Euclidean distances: D[i,j] = ||marginal[i] - training[j]||_2
    from scipy.spatial.distance import cdist
    dists = cdist(subsample, y_t_batch.cpu().numpy())  # (n_subsample, N_COVERAGE)

    # For each marginal point, find distance to NEAREST training sample
    min_dists = dists.min(axis=1)  # (n_subsample,)
    distances.extend(min_dists)

ax.hist(distances, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(np.median(distances), color='red', linestyle='--',
           label=f'Median: {np.median(distances):.2f}')
ax.set_xlabel("Min Distance to Training Sample")
ax.set_ylabel("Count")
ax.set_title("Distance from Marginals to Training Coverage")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "score_training_coverage.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR / 'score_training_coverage.png'}")

# ============================================================================
# Summary and Interpretation
# ============================================================================

print("\n" + "="*70)
print("Score Model Diagnostics Complete")
print("="*70)
print(f"\nAll diagnostics saved to: {OUTPUT_DIR}")

print("\n" + "="*70)
print("KEY METRICS")
print("="*70)
print(f"  Latent dimension: {latent_dim}")
print(f"  Score parameterization: {SCORE_PARAMETERIZATION}")
print(f"  Noise schedule: sigma_0={SIGMA_0}, decay_rate={DECAY_RATE}")
print(f"    -> sigma(t=0) = {SIGMA_0:.4f}")
print(f"    -> sigma(t=1) = {SIGMA_0 * np.exp(-DECAY_RATE):.4f}")
print(f"  Training coverage: median distance to samples = {np.median(distances):.3f}")

print("\n" + "="*70)
print("DIAGNOSTIC FILES")
print("="*70)
print(f"  1. {OUTPUT_DIR / 'score_vector_field.png'}")
print("     Raw score field ∇log p(y_t) overlaid on data")
print("     NOTE: Magnitudes are HUGE due to division by g(t)^2")
print("     What matters: DIRECTION (should point toward data)")
print()
print(f"  2. {OUTPUT_DIR / 'velocity_vs_score_magnitude.png'}")
print("     Magnitude comparison: velocity vs score CONTRIBUTION in backward SDE")
print("     This uses SCALED score (actual drift term), not raw score")
print()
print(f"  3. {OUTPUT_DIR / 'backward_sde_drift_field.png'}")
print("     Complete backward SDE drift: -v + score_term")
print("     Should point toward earlier marginals to reverse forward flow")
print()
print(f"  4. {OUTPUT_DIR / 'score_training_coverage.png'}")
print("     Where score model saw training data vs where marginals are")
print("     Large distance → extrapolation required → potential divergence")

print("\n" + "="*70)
print("UNDERSTANDING THE SCORE MAGNITUDE PARADOX")
print("="*70)
print("You may notice:")
print("  - score_vector_field.png shows HUGE magnitudes far from data")
print("  - velocity_vs_score_magnitude.png shows TINY score contribution")
print()
print("This is NOT a contradiction! Here's why:")
print()
print("TWO SCORE REPRESENTATIONS:")
print("  1. Raw score:    s_raw = ∇_y log p(y_t)  [natural quantity]")
print("  2. Scaled score: s_scaled = (g²/2) * s_raw  [used in SDE]")
print()
print("The score_vector_field.png visualizes RAW score for interpretability:")
print("  - Conversion: s_raw = s_scaled / (g²/2)")
print(f"  - At t=1: g² = {(SIGMA_0 * np.exp(-DECAY_RATE))**2:.2e}")
print(f"  - Dividing by g²/2 amplifies magnitude by ~{1/(2*(SIGMA_0 * np.exp(-DECAY_RATE))**2):.0f}×")
print("  - Result: raw score has enormous magnitude (but correct direction)")
print()
print("The backward SDE uses SCALED score directly (no division):")
print("  - Drift = -v(y,t) + s_scaled(y,t)")
print("  - s_scaled = score_model(y, t)  [direct output]")
print("  - Magnitude is SMALL because it includes the tiny g² factor")
print()
print("WHAT ACTUALLY MATTERS:")
print("  ✓ Direction of raw score (should point toward data)")
print("  ✓ Magnitude of scaled score vs velocity (for backward SDE)")
print("  ✗ Magnitude of raw score (artifact of visualization)")

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("Score Vector Field:")
print("  ✓ GOOD: Arrows point INWARD toward high-density regions")
print("  ✗ BAD:  Arrows point OUTWARD or flip erratically")
print("  → Ignore magnitude! Only direction matters.")
print()
print("Velocity vs Score Magnitude:")
print("  - Ratio < 0.1:  Score too weak to correct trajectory deviations")
print("  - Ratio ~ 1.0:  Balanced - score can correct drift")
print("  - Ratio > 10:   Score dominates - may cause instability")
print()
print("Backward SDE Drift Field:")
print("  ✓ GOOD: Drift points toward earlier marginals")
print("  ✗ BAD:  Drift points away or has erratic behavior")
print()
print("Training Coverage:")
print("  - Distance < 0.5:  Training samples densely cover marginals")
print("  - Distance > 1.0:  Marginals outside training support")
print("  → Large distance requires extrapolation → potential divergence")

print("\n" + "="*70)
print("RECOMMENDED NEXT STEPS")
print("="*70)

median_dist = np.median(distances)
if median_dist > 1.0:
    print("⚠ WARNING: Large median distance ({:.2f}) indicates poor training coverage".format(median_dist))
    print()
    print("Recommendations:")
    print("  1. Use forward ODE (which already works well)")
    print("  2. Increase noise schedule: sigma_0=0.2, decay_rate=1.0")
    print("     → Larger noise → stronger score → better correction")
    print("  3. Train with broader distribution (--score_mode trajectory)")
    print("  4. Increase score weight (--score_weight 10.0)")
else:
    print("✓ Training coverage is reasonable (median distance: {:.2f})".format(median_dist))
    print()
    print("If backward SDE still diverges, check:")
    print("  - Score/velocity ratio (should be > 0.1 for effective correction)")
    print("  - Score vector directions (should point toward data)")
    print("  - Consider increasing noise schedule for stronger score")

print("\n" + "="*70)
print("See SCORE_DIAGNOSTICS_EXPLAINED.md for detailed explanations")
print("="*70)
