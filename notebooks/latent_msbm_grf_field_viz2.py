# %%
# %% [markdown]
# # Latent MSBM: GRF Field Space Visualization
# 
# This notebook visualizes **conditionally generated** trajectories from trained latent MSBM 
# policies, lifted from latent space → PCA coefficients → GRF spatial fields.
# 
# **Pipeline**:
# 1. Load trained latent MSBM policies (forward/backward)
# 2. Generate conditional trajectories using SDE rollout
# 3. Decode latent → ambient (PCA coefficients)
# 4. Lift PCA coefficients → GRF spatial fields
# 5. Visualize generated fields and compare with reference data
# 
# Based on the utilities in `scripts/images/images_plot.py` and `scripts/images/field_visualization.py`.

# %%
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
path_root = Path.cwd().parent
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from scripts.utils import get_device
from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.noise_schedule import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule
from scripts.latent_msbm_main import load_autoencoder, _load_data
from scripts.pca.pca_visualization_utils import parse_args_file
from scripts.images.field_visualization import (
    format_for_paper,
    reconstruct_fields_from_coefficients,
    plot_field_snapshots,
    plot_field_evolution_gif,
    plot_spatial_correlation,
    plot_sample_comparison_grid,
)
import argparse

print("Imports successful!")

# %% [markdown]
# ## Paper Formatting

# %%
# Apply paper formatting from field_visualization module
format_for_paper()
print("✓ Applied paper formatting to all plots")

# %% [markdown]
# ## Configuration
# 
# Set the paths to your trained MSBM model and dataset.

# %%
# USER: Update these paths to your MSBM training output directory and data
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

# Visualization settings
n_infer = 50  # Number of samples for conditional generation
n_field_samples = 5  # Number of field samples to visualize in snapshots
n_traj_steps = 1500  # Number of steps for trajectory generation
figsize_per_subplot = 5  # Size of each subplot

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

# MSBM parameters
hidden = _cfg("hidden", [1024, 1024, 1024])
time_dim = _cfg("time_dim", 32)
policy_arch = _cfg("policy_arch", "resnet")
var = _cfg("var", 0.5)
var_schedule = _cfg("var_schedule", "constant")
var_decay_rate = _cfg("var_decay_rate", 2.0)
var_time_ref = _cfg("var_time_ref", None)
t_scale = _cfg("t_scale", 1.0)
interval = _cfg("interval", 100)
use_t_idx = _cfg("use_t_idx", False)
latent_dim_override = _cfg("latent_dim_override", 376)
ae_ode_method = _cfg("ae_ode_method", "dopri5")

print(f"Data path: {data_path}")
print(f"AE checkpoint: {ae_checkpoint}")
print(f"AE type: {ae_type}")
print(f"Policy architecture: {policy_arch}")

# %% [markdown]
# ## Load PCA Metadata for Field Reconstruction

# %%
# Load PCA information from the dataset for field reconstruction
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
# Load data for MSBM (uses same PCA coefficients, train/test split)
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

print(f"\nLoaded MSBM data:")
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
# ## Build MSBM Agent and Load Policies

# %%
# Create noise schedule
t_ref_default = float(max(1.0, (T - 1) * float(t_scale)))
t_ref = float(var_time_ref) if var_time_ref is not None else t_ref_default

if var_schedule == "constant":
    sigma_schedule = ConstantSigmaSchedule(float(var))
else:
    sigma_schedule = ExponentialContractingSigmaSchedule(
        sigma_0=float(var),
        decay_rate=float(var_decay_rate),
        t_ref=float(t_ref),
    )

# Build agent
agent = LatentMSBMAgent(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    zt=list(map(float, zt_np.tolist())),
    policy_arch=policy_arch,
    hidden_dims=hidden,
    time_dim=time_dim,
    var=var,
    sigma_schedule=sigma_schedule,
    t_scale=t_scale,
    interval=interval,
    use_t_idx=use_t_idx,
    lr=1e-4,
    lr_gamma=1.0,
    use_ema=False,
    device=device,
)

# Encode marginals
agent.encode_marginals(x_train, x_test)
print(f"✓ Encoded marginals: latent_test.shape={agent.latent_test.shape}")

# Load policy checkpoints
def _pick_policy_checkpoint(msbm_dir: Path, which: str) -> Path:
    if which == "forward":
        candidates = ["latent_msbm_policy_forward_ema.pth", "latent_msbm_policy_forward.pth"]
    elif which == "backward":
        candidates = ["latent_msbm_policy_backward_ema.pth", "latent_msbm_policy_backward.pth"]
    else:
        raise ValueError(which)
    
    for name in candidates:
        p = msbm_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {which} policy checkpoint in {msbm_dir}")

ckpt_f = _pick_policy_checkpoint(msbm_dir, which="forward")
ckpt_b = _pick_policy_checkpoint(msbm_dir, which="backward")

agent.z_f.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=False), strict=True)
agent.z_b.load_state_dict(torch.load(ckpt_b, map_location=device, weights_only=False), strict=True)
agent.z_f.eval()
agent.z_b.eval()

print(f"\n✓ Loaded forward policy from {ckpt_f.name}")
print(f"✓ Loaded backward policy from {ckpt_b.name}")

# %% [markdown]
# ## Prepare Reference Data

# %%
# Sample test indices
rng = np.random.default_rng(seed)
n_test = int(agent.latent_test.shape[1])
n_infer = min(n_infer, n_test, int(x_test.shape[1]))
idx = rng.choice(n_test, size=n_infer, replace=False)

# Get latent reference data
latent_ref = agent.latent_test[:, idx].detach().cpu().numpy()  # (T, N, K)
print(f"Using {n_infer} samples for visualization")
print(f"Latent reference shape: {latent_ref.shape}")

# Reference ambient coefficients
ambient_reference = x_test[:, idx, :]  # (T, N, D)
print(f"Reference ambient shape: {ambient_reference.shape}")

# %% [markdown]
# ## Conditional Trajectory Generation
# 
# Generate trajectories using the trained forward and backward MSBM policies.
# 
# **IMPORTANT**: The backward policy is trained on time-reversed interval labels (MSBM flip).
# To sample a trajectory from the last marginal back to the first, we must use the same
# reversed-time convention as during training.

# %%
@torch.no_grad()
def generate_knots(
    agent: LatentMSBMAgent,
    policy: torch.nn.Module,
    y_init: torch.Tensor,
    direction: str,
    drift_clip_norm: float | None = None,
) -> torch.Tensor:
    """Generate a trajectory that records only marginal knot states (T, N, K).
    
    This matches the `_generate_knots` function from `latent_msbm_eval.py`.
    
    Args:
        agent: The MSBM agent with SDE utilities
        policy: The policy network (z_f for forward, z_b for backward)
        y_init: Initial latent points, shape (N, K)
        direction: "forward" or "backward"
        drift_clip_norm: Optional drift clipping for stability
    
    Returns:
        Knot trajectory, shape (T, N, K)
    """
    ts_rel = agent.ts  # local time grid on [0, 1]
    y = y_init.clone()
    
    knots: list[torch.Tensor] = []
    
    if direction == "forward":
        knots.append(y.clone())
        for i in range(agent.t_dists.numel() - 1):
            t0 = agent.t_dists[i]
            t1 = agent.t_dists[i + 1]
            _, y = agent.sde.sample_traj(
                ts_rel, policy, y, t0, t_final=t1, save_traj=False, drift_clip_norm=drift_clip_norm
            )
            knots.append(y.clone())
            
    elif direction == "backward":
        # The backward policy is trained on time-reversed interval labels (MSBM flip).
        # To sample from last marginal to first, run the backward policy with reversed time labels.
        knots.append(y.clone())
        num_intervals = int(agent.t_dists.numel() - 1)
        for i in range(num_intervals - 1, -1, -1):
            # Compute reversed time labels as used during training
            rev_i = (num_intervals - 1) - i
            t0_rev = agent.t_dists[rev_i]
            t1_rev = agent.t_dists[rev_i + 1]
            _, y = agent.sde.sample_traj(
                ts_rel, policy, y, t0_rev, t_final=t1_rev, save_traj=False, drift_clip_norm=drift_clip_norm
            )
            knots.append(y.clone())
        # Reverse the list so it goes from t=0 to t=1
        knots = list(reversed(knots))
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return torch.stack(knots, dim=0)


def _map_internal_to_zt(
    t_internal: np.ndarray,
    t_dists: np.ndarray,
    zt: np.ndarray,
) -> np.ndarray:
    """Map internal MSBM times (t_dists scale) to AE times (zt in [0,1]) piecewise-linearly."""
    td = torch.from_numpy(t_dists.astype(np.float32))
    z = torch.from_numpy(zt.astype(np.float32))
    t = torch.from_numpy(t_internal.astype(np.float32))
    
    idx = torch.bucketize(t, td[1:])
    idx = torch.clamp(idx, 0, td.numel() - 2)
    
    t0 = td[idx]
    t1 = td[idx + 1]
    z0 = z[idx]
    z1 = z[idx + 1]
    
    denom = torch.where((t1 - t0).abs() < 1e-8, torch.full_like(t1, 1e-8), (t1 - t0))
    alpha = torch.clamp((t - t0) / denom, 0.0, 1.0)
    out = z0 + alpha * (z1 - z0)
    return out.numpy()


@torch.no_grad()
def generate_dense_trajectory(
    agent: LatentMSBMAgent,
    policy: torch.nn.Module,
    y_init: torch.Tensor,
    start_idx: int,
    end_idx: int,
    stride: int = 10,
    drift_clip_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense rollout between two marginal indices.
    
    This matches `_generate_dense` from `latent_msbm_eval.py`.
    
    Returns:
        t_internal: (S_total,) internal MSBM time values
        latent: (S_total, N, K)
    """
    ts_rel = agent.ts
    keep = list(range(0, int(ts_rel.numel()) - 1, int(stride))) + [int(ts_rel.numel()) - 1]
    
    y = y_init.clone()
    t_internal_parts: list[np.ndarray] = []
    traj_parts: list[np.ndarray] = []
    
    if start_idx == end_idx:
        t0 = float(agent.t_dists[start_idx].item())
        return np.asarray([t0], dtype=np.float32), y.detach().cpu().numpy()[None, ...]
    
    step = 1 if end_idx > start_idx else -1
    idxs = list(range(int(start_idx), int(end_idx), step))
    
    if step == 1:
        pairs = [(i, i + 1) for i in idxs]
    else:
        pairs = [(i, i - 1) for i in idxs]
    
    num_intervals = int(agent.t_dists.numel() - 1)
    
    for seg_i, (i0, i1) in enumerate(pairs):
        t0 = agent.t_dists[i0]
        t1 = agent.t_dists[i1]
        
        # For backward rollouts, use reversed time labels
        t_init = t0
        t_final = t1
        if i1 < i0:  # backward direction
            i = int(i0 - 1)
            rev_i = (num_intervals - 1) - i
            t_init = agent.t_dists[rev_i]
            t_final = agent.t_dists[rev_i + 1]
        
        traj, y = agent.sde.sample_traj(
            ts_rel, policy, y, t_init, t_final=t_final, save_traj=True, drift_clip_norm=drift_clip_norm
        )
        
        traj_np = traj[:, keep, :].detach().cpu().numpy()  # (N, S_keep, K)
        times_np = (float(t0.item()) + (ts_rel[keep].cpu().numpy() * (float(t1.item()) - float(t0.item())))).astype(np.float32)
        
        if seg_i > 0:
            traj_np = traj_np[:, 1:, :]
            times_np = times_np[1:]
        
        traj_parts.append(np.transpose(traj_np, (1, 0, 2)))  # (S_keep, N, K)
        t_internal_parts.append(times_np)
    
    t_internal = np.concatenate(t_internal_parts, axis=0)
    latent = np.concatenate(traj_parts, axis=0)
    
    # Ensure increasing time
    if t_internal.shape[0] >= 2 and t_internal[1] < t_internal[0]:
        t_internal = t_internal[::-1].copy()
        latent = latent[::-1].copy()
    
    return t_internal, latent

# %%
print("Generating conditional trajectories...")

# Get starting points from first marginal (for forward) and last marginal (for backward)
y_start_forward = agent.latent_test[0, idx].clone()  # (N, K)
y_start_backward = agent.latent_test[-1, idx].clone()  # (N, K)

print(f"  Forward: starting from t=0 with {n_infer} samples")
print(f"  Backward: starting from t=1 with {n_infer} samples")

# Generate forward trajectory knots
print("\nGenerating forward policy trajectories (knot sampling)...")
knots_forward = generate_knots(
    agent,
    policy=agent.z_f,
    y_init=y_start_forward,
    direction="forward",
)
print(f"✓ Forward knots shape: {knots_forward.shape}")  # (T, N, K)

# Convert to numpy and create time arrays
traj_forward = knots_forward.cpu().numpy()  # (T, N, K)
times_forward = np.array(agent.zt)  # AE times for each marginal

# Generate backward trajectory knots
print("\nGenerating backward policy trajectories (knot sampling)...")
knots_backward = generate_knots(
    agent,
    policy=agent.z_b,
    y_init=y_start_backward,
    direction="backward",
)
print(f"✓ Backward knots shape: {knots_backward.shape}")  # (T, N, K)

traj_backward = knots_backward.cpu().numpy()  # (T, N, K)
times_backward = np.array(agent.zt)  # Same time grid

print(f"\nLatent trajectories generated.")
print(f"  Forward: {traj_forward.shape}, times {times_forward.min():.4f} to {times_forward.max():.4f}")
print(f"  Backward: {traj_backward.shape}, times {times_backward.min():.4f} to {times_backward.max():.4f}")

# %% [markdown]
# ## Decode Generated Trajectories to Ambient Space

# %%
print("Decoding generated trajectories to ambient PCA coefficient space...")

# Decode forward trajectories
ambient_forward = agent.decode_trajectories(traj_forward, times_forward)
print(f"✓ Forward ambient shape: {ambient_forward.shape}")

# Decode backward trajectories
ambient_backward = agent.decode_trajectories(traj_backward, times_backward)
print(f"✓ Backward ambient shape: {ambient_backward.shape}")

# %% [markdown]
# ## Lift PCA Coefficients to GRF Field Space
# 
# Using the `reconstruct_fields_from_coefficients` function from field_visualization.py.
# We lift:
# - Forward policy generated trajectories
# - Backward policy generated trajectories
# - Reference (ground truth) data for comparison
# 
# **Note on min-max scaling and PCA reconstruction:**
# 
# The original GRF fields were min-max scaled to [0,1] before PCA was applied.
# After decoding latent trajectories and applying inverse PCA:
# 1. PCA reconstruction adds the mean and applies inverse whitening
# 2. Generated latent points may have slightly different statistics than training data
# 3. The resulting reconstructed fields may fall slightly outside [0,1]
# 
# This is **expected behavior** - the model generates in the latent space, and small
# deviations in the latent→ambient→field pipeline can produce out-of-range values.
# For visualization, fields can be clipped to [0,1] if desired.


# %%
def lift_coefficients_to_fields(
    coeffs: np.ndarray,
    pca_info: dict,
    resolution: int,
) -> np.ndarray:
    """Lift PCA coefficients to spatial field space.
    
    Args:
        coeffs: PCA coefficients, shape (T, N, n_components) or (N, n_components)
        pca_info: Dictionary with 'mean', 'components', 'explained_variance', 'is_whitened'
        resolution: Spatial resolution (fields are resolution x resolution)
    
    Returns:
        Reconstructed fields, shape (T, N, H, W) or (N, H, W)
    """
    return reconstruct_fields_from_coefficients(coeffs, pca_info, resolution)


def sample_trajectory_at_marginals(
    traj: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Sample trajectory at specific target times via nearest-neighbor lookup.
    
    Args:
        traj: Full trajectory, shape (S, N, D)
        times: Trajectory time values, shape (S,)
        target_times: Target times to sample at, shape (T,)
    
    Returns:
        Sampled trajectory, shape (T, N, D)
    """
    sampled = []
    for t in target_times:
        idx = np.argmin(np.abs(times - t))
        sampled.append(traj[idx])
    return np.stack(sampled, axis=0)


# %%
print("Lifting generated trajectories to GRF field space...")
print("="*60)

# Since we now use knot sampling, trajectories are already at marginal times
# No need to resample - ambient_forward/backward are shape (T, N, D) at marginal times
print("\nGenerated trajectories are already at marginal knots.")
print(f"  Forward at marginals shape: {ambient_forward.shape}")
print(f"  Backward at marginals shape: {ambient_backward.shape}")

# Lift forward policy generated coefficients → fields
print("\nLifting forward policy trajectories...")
forward_fields = lift_coefficients_to_fields(ambient_forward, pca_info, resolution)
print(f"✓ Forward generated fields shape: {forward_fields.shape}")

# Lift backward policy generated coefficients → fields
print("\nLifting backward policy trajectories...")
backward_fields = lift_coefficients_to_fields(ambient_backward, pca_info, resolution)
print(f"✓ Backward generated fields shape: {backward_fields.shape}")

# Lift reference coefficients → fields (ground truth)
print("\nLifting reference (ground truth) data...")
reference_fields = lift_coefficients_to_fields(ambient_reference, pca_info, resolution)
print(f"✓ Reference fields shape: {reference_fields.shape}")

# Prepare reference data list for comparison functions
testdata_fields = [reference_fields[t_idx] for t_idx in range(T)]
print(f"✓ Reference data: {len(testdata_fields)} marginals")

print("="*60)

# %% [markdown]
# ## Field Visualization: Snapshots
# 
# Visualize field samples at selected time points.
# - **REFERENCE**: Ground truth fields from data
# - **Forward Policy**: Fields generated via forward MSBM policy
# - **Backward Policy**: Fields generated via backward MSBM policy

# %%
# Create output directory
eval_dir = msbm_dir / "eval" / "grf_fields"
eval_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {eval_dir}")

# %%
# Define a local run object for saving (we'll skip wandb logging)
class LocalRun:
    """Mock run object that skips wandb logging."""
    def log(self, *args, **kwargs):
        pass  # No-op

local_run = LocalRun()

# %%
# Plot REFERENCE field snapshots (ground truth)
print("\n" + "="*60)
print("GENERATING REFERENCE (GROUND TRUTH) FIELD SNAPSHOTS")
print("="*60)
plot_field_snapshots(
    reference_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    n_samples=min(n_field_samples, reference_fields.shape[1]),
    score=False,
    cmap='viridis',
    filename_prefix='REFERENCE_field_snapshots',
    close=False,
)
plt.suptitle("REFERENCE Fields (Ground Truth)", fontsize=14, y=0.995, fontweight='bold')
plt.tight_layout()
plt.savefig(eval_dir / "REFERENCE_field_snapshots.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: REFERENCE_field_snapshots.png")
plt.show()

# %%
# Plot FORWARD POLICY generated field snapshots
print("\n" + "="*60)
print("GENERATING FORWARD POLICY FIELD SNAPSHOTS")
print("="*60)
plot_field_snapshots(
    forward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    n_samples=min(n_field_samples, forward_fields.shape[1]),
    score=False,
    cmap='viridis',
    filename_prefix='forward_policy_field_snapshots',
    close=False,
)
plt.suptitle("Forward Policy Generated Fields (MSBM → Latent → PCA → GRF)", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(eval_dir / "forward_policy_field_snapshots.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: forward_policy_field_snapshots.png")
plt.show()

# %%
# Plot BACKWARD POLICY generated field snapshots
print("\n" + "="*60)
print("GENERATING BACKWARD POLICY FIELD SNAPSHOTS")
print("="*60)
plot_field_snapshots(
    backward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    n_samples=min(n_field_samples, backward_fields.shape[1]),
    score=False,
    cmap='viridis',
    filename_prefix='backward_policy_field_snapshots',
    close=False,
)
plt.suptitle("Backward Policy Generated Fields (MSBM → Latent → PCA → GRF)", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(eval_dir / "backward_policy_field_snapshots.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: backward_policy_field_snapshots.png")
plt.show()

# %% [markdown]
# ## Field Evolution Animation

# %%
print("\n" + "="*60)
print("GENERATING FIELD EVOLUTION ANIMATIONS")
print("="*60)

# Animate REFERENCE field evolution (sample 0)
print("\nReference animations...")
plot_field_evolution_gif(
    reference_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    sample_idx=0,
    score=False,
    cmap='viridis',
    fps=5,
    filename_prefix='REFERENCE_field_evolution_sample0',
)
print(f"✓ Saved: REFERENCE_field_evolution_sample0.gif")

# Animate forward policy field evolution (sample 0)
print("\nForward policy animations...")
plot_field_evolution_gif(
    forward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    sample_idx=0,
    score=False,
    cmap='viridis',
    fps=5,
    filename_prefix='forward_policy_field_evolution_sample0',
)
print(f"✓ Saved: forward_policy_field_evolution_sample0.gif")

# Animate backward policy field evolution (sample 0)
print("\nBackward policy animations...")
plot_field_evolution_gif(
    backward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    sample_idx=0,
    score=False,
    cmap='viridis',
    fps=5,
    filename_prefix='backward_policy_field_evolution_sample0',
)
print(f"✓ Saved: backward_policy_field_evolution_sample0.gif")

# Additional samples for forward policy
for sample_idx in [1, 2]:
    if sample_idx < forward_fields.shape[1]:
        plot_field_evolution_gif(
            forward_fields,
            zt_np.tolist(),
            str(eval_dir),
            local_run,
            sample_idx=sample_idx,
            score=False,
            cmap='viridis',
            fps=5,
            filename_prefix=f'forward_policy_field_evolution_sample{sample_idx}',
        )
        print(f"✓ Saved: forward_policy_field_evolution_sample{sample_idx}.gif")


# %% [markdown]
# ## Spatial Correlation Analysis

# %%
print("\n" + "="*60)
print("GENERATING SPATIAL AUTOCORRELATION ANALYSIS")
print("="*60)

print("\nReference spatial autocorrelation...")
plot_spatial_correlation(
    reference_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    score=False,
    filename_prefix='REFERENCE_spatial_correlation',
    close=False,
)
plt.suptitle("REFERENCE: Spatial Autocorrelation", fontsize=14, fontweight='bold')
plt.savefig(eval_dir / "REFERENCE_spatial_correlation.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: REFERENCE_spatial_correlation.png")
plt.show()

# %%
print("\nForward policy spatial autocorrelation...")
plot_spatial_correlation(
    forward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    score=False,
    filename_prefix='forward_policy_spatial_correlation',
    close=False,
)
plt.suptitle("Forward Policy: Spatial Autocorrelation", fontsize=14)
plt.savefig(eval_dir / "forward_policy_spatial_correlation.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: forward_policy_spatial_correlation.png")
plt.show()

# %%
print("\nBackward policy spatial autocorrelation...")
plot_spatial_correlation(
    backward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    score=False,
    filename_prefix='backward_policy_spatial_correlation',
    close=False,
)
plt.suptitle("Backward Policy: Spatial Autocorrelation", fontsize=14)
plt.savefig(eval_dir / "backward_policy_spatial_correlation.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: backward_policy_spatial_correlation.png")
plt.show()

# %% [markdown]
# ## Sample Comparison Grid
# 
# Side-by-side comparison of reference vs generated fields with difference maps.

# %%
print("\n" + "="*60)
print("GENERATING SAMPLE COMPARISON GRIDS")
print("="*60)

# Compare Forward Policy vs Reference
print("\nComparing Forward Policy vs Reference...")
plot_sample_comparison_grid(
    testdata_fields,
    forward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    score=False,
    n_samples=min(5, forward_fields.shape[1]),
    cmap='viridis',
    diff_cmap='RdBu_r',
    filename_prefix='forward_vs_reference_comparison',
    close=False,
)
plt.suptitle("Field Comparison: REFERENCE vs Forward Policy Generated", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(eval_dir / "forward_vs_reference_comparison.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: forward_vs_reference_comparison.png")
plt.show()

# %%
# Compare Backward Policy vs Reference
print("\nComparing Backward Policy vs Reference...")
plot_sample_comparison_grid(
    testdata_fields,
    backward_fields,
    zt_np.tolist(),
    str(eval_dir),
    local_run,
    score=False,
    n_samples=min(5, backward_fields.shape[1]),
    cmap='viridis',
    diff_cmap='RdBu_r',
    filename_prefix='backward_vs_reference_comparison',
    close=False,
)
plt.suptitle("Field Comparison: REFERENCE vs Backward Policy Generated", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(eval_dir / "backward_vs_reference_comparison.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: backward_vs_reference_comparison.png")
plt.show()

# %% [markdown]
# ## Custom Field Comparison Visualization
# 
# A more detailed view comparing specific samples across time:
# - Reference (ground truth)
# - Forward policy generated
# - Backward policy generated
# - Difference maps with RMSE

# %%
def plot_field_trajectory_comparison(
    reference: np.ndarray,
    generated: np.ndarray,
    zt: np.ndarray,
    sample_idx: int = 0,
    direction: str = "forward",
    figsize_scale: float = 3.0,
) -> plt.Figure:
    """Plot a single sample's field trajectory: reference, generated, and difference.
    
    Args:
        reference: Reference fields, shape (T, N, H, W)
        generated: Generated fields, shape (T, N, H, W)
        zt: Time values, shape (T,)
        sample_idx: Which sample to visualize
        direction: "forward" or "backward" (for labeling)
        figsize_scale: Scale factor for figure size
    
    Returns:
        Matplotlib figure
    """
    T = reference.shape[0]
    
    # Select time steps to visualize (limit to 7 for readability)
    if T > 7:
        time_indices = np.linspace(0, T - 1, 7, dtype=int)
    else:
        time_indices = np.arange(T)
    
    n_cols = len(time_indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(figsize_scale * n_cols, figsize_scale * 3))
    
    # Compute global color limits
    ref_sample = reference[:, sample_idx]
    gen_sample = generated[:, sample_idx]
    vmin = min(ref_sample.min(), gen_sample.min())
    vmax = max(ref_sample.max(), gen_sample.max())
    
    diff = ref_sample - gen_sample
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    
    for col, t_idx in enumerate(time_indices):
        t_val = zt[t_idx]
        
        # Reference
        ax = axes[0, col]
        im_ref = ax.imshow(ref_sample[t_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f't = {t_val:.3f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('REFERENCE', fontsize=11, fontweight='bold')
        
        # Generated
        ax = axes[1, col]
        im_gen = ax.imshow(gen_sample[t_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f'{direction.capitalize()} Policy', fontsize=11)
        
        # Difference
        ax = axes[2, col]
        im_diff = ax.imshow(diff[t_idx], cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Difference', fontsize=11)
        
        # RMSE annotation
        rmse = np.sqrt(np.mean(diff[t_idx] ** 2))
        ax.text(0.5, -0.12, f'RMSE: {rmse:.2e}', transform=ax.transAxes,
                ha='center', fontsize=8)
    
    # Colorbars
    cax1 = fig.add_axes([0.92, 0.65, 0.015, 0.25])
    fig.colorbar(im_ref, cax=cax1, label='Field Value')
    
    cax2 = fig.add_axes([0.92, 0.1, 0.015, 0.25])
    fig.colorbar(im_diff, cax=cax2, label='Error')
    
    fig.suptitle(f'Sample {sample_idx}: REFERENCE vs {direction.capitalize()} Policy', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return fig

# %%
print("\n" + "="*60)
print("GENERATING DETAILED FIELD TRAJECTORY COMPARISONS")
print("="*60)

# Plot for first 3 samples - Forward Policy
print("\nForward Policy vs Reference:")
for sample_idx in range(min(3, forward_fields.shape[1])):
    fig = plot_field_trajectory_comparison(
        reference_fields,
        forward_fields,
        zt_np,
        sample_idx=sample_idx,
        direction="forward",
    )
    save_path = eval_dir / f"forward_trajectory_sample{sample_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path.name}")
    plt.show()
    plt.close(fig)

# %%
# Plot for first 3 samples - Backward Policy
print("\nBackward Policy vs Reference:")
for sample_idx in range(min(3, backward_fields.shape[1])):
    fig = plot_field_trajectory_comparison(
        reference_fields,
        backward_fields,
        zt_np,
        sample_idx=sample_idx,
        direction="backward",
    )
    save_path = eval_dir / f"backward_trajectory_sample{sample_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {save_path.name}")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Error Analysis
# 
# Quantitative analysis of generation error across marginals and samples.
# Compare both forward and backward policies against reference.

# %%
def compute_field_errors(
    reference: np.ndarray,
    generated: np.ndarray,
) -> dict:
    """Compute various error metrics between reference and generated fields.
    
    Args:
        reference: Reference fields, shape (T, N, H, W)
        generated: Generated fields, shape (T, N, H, W)
    
    Returns:
        Dictionary of error metrics
    """
    T, N, H, W = reference.shape
    
    # Per-sample, per-time RMSE
    diff = reference - generated
    rmse_per_sample_time = np.sqrt(np.mean(diff ** 2, axis=(2, 3)))  # (T, N)
    
    # Relative error: ||err||_2 / ||ref||_2 per (T, N).
    # Use an RMS-normalized denominator so the ratio is consistent with RMSE.
    # (If you use an L2 denominator with an RMSE numerator, you introduce an extra sqrt(H*W) factor.)
    ref_rms = np.sqrt(np.mean(reference ** 2, axis=(2, 3)))  # (T, N)
    rel_error = rmse_per_sample_time / np.maximum(ref_rms, 1e-10)
    
    return {
        'rmse_mean_per_time': rmse_per_sample_time.mean(axis=1),  # (T,)
        'rmse_std_per_time': rmse_per_sample_time.std(axis=1),  # (T,)
        'rel_error_mean_per_time': rel_error.mean(axis=1),  # (T,)
        'rel_error_std_per_time': rel_error.std(axis=1),  # (T,)
        'overall_rmse': np.mean(rmse_per_sample_time),
        'overall_rel_error': np.mean(rel_error),
    }

# %%
print("\n" + "="*60)
print("COMPUTING GENERATION ERRORS")
print("="*60)

# Compute errors for both policies
errors_forward = compute_field_errors(reference_fields, forward_fields)
errors_backward = compute_field_errors(reference_fields, backward_fields)

print(f"\n{'='*70}")
print("GENERATION ERROR SUMMARY: FORWARD POLICY")
print(f"{'='*70}")
print(f"Overall RMSE: {errors_forward['overall_rmse']:.4e}")
print(f"Overall Relative Error: {errors_forward['overall_rel_error']:.4f} ({errors_forward['overall_rel_error']*100:.2f}%)")

print(f"\n{'='*70}")
print("GENERATION ERROR SUMMARY: BACKWARD POLICY")
print(f"{'='*70}")
print(f"Overall RMSE: {errors_backward['overall_rmse']:.4e}")
print(f"Overall Relative Error: {errors_backward['overall_rel_error']:.4f} ({errors_backward['overall_rel_error']*100:.2f}%)")

# %%
# Detailed per-time breakdown
print(f"\n{'='*70}")
print("FORWARD POLICY - Per-Time Breakdown")
print(f"{'='*70}")
print(f"{'Time':>8}  {'RMSE (mean ± std)':>20}  {'Rel. Error (mean ± std)':>25}")
print("-" * 70)
for t_idx, t_val in enumerate(zt_np):
    rmse_mean = errors_forward['rmse_mean_per_time'][t_idx]
    rmse_std = errors_forward['rmse_std_per_time'][t_idx]
    rel_mean = errors_forward['rel_error_mean_per_time'][t_idx]
    rel_std = errors_forward['rel_error_std_per_time'][t_idx]
    print(f"{t_val:>8.4f}  {rmse_mean:.3e} ± {rmse_std:.3e}  {rel_mean:.4f} ± {rel_std:.4f}")

print(f"\n{'='*70}")
print("BACKWARD POLICY - Per-Time Breakdown")
print(f"{'='*70}")
print(f"{'Time':>8}  {'RMSE (mean ± std)':>20}  {'Rel. Error (mean ± std)':>25}")
print("-" * 70)
for t_idx, t_val in enumerate(zt_np):
    rmse_mean = errors_backward['rmse_mean_per_time'][t_idx]
    rmse_std = errors_backward['rmse_std_per_time'][t_idx]
    rel_mean = errors_backward['rel_error_mean_per_time'][t_idx]
    rel_std = errors_backward['rel_error_std_per_time'][t_idx]
    print(f"{t_val:>8.4f}  {rmse_mean:.3e} ± {rmse_std:.3e}  {rel_mean:.4f} ± {rel_std:.4f}")

# %%
# Plot error over time - comparing both policies
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE over time
ax = axes[0]
ax.errorbar(
    zt_np,
    errors_forward['rmse_mean_per_time'],
    yerr=errors_forward['rmse_std_per_time'],
    fmt='o-',
    capsize=3,
    label='Forward Policy',
    color='blue',
)
ax.errorbar(
    zt_np + 0.01,  # slight offset for visibility
    errors_backward['rmse_mean_per_time'],
    yerr=errors_backward['rmse_std_per_time'],
    fmt='s-',
    capsize=3,
    label='Backward Policy',
    color='orange',
)
ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Field Generation RMSE Over Time', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Relative error over time
ax = axes[1]
ax.errorbar(
    zt_np,
    errors_forward['rel_error_mean_per_time'] * 100,
    yerr=errors_forward['rel_error_std_per_time'] * 100,
    fmt='o-',
    capsize=3,
    label='Forward Policy',
    color='blue',
)
ax.errorbar(
    zt_np + 0.01,
    errors_backward['rel_error_mean_per_time'] * 100,
    yerr=errors_backward['rel_error_std_per_time'] * 100,
    fmt='s-',
    capsize=3,
    label='Backward Policy',
    color='orange',
)
ax.set_xlabel('Time t', fontsize=12)
ax.set_ylabel('Relative Error (%)', fontsize=12)
ax.set_title('Relative Generation Error Over Time', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = eval_dir / "generation_error_over_time.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Marginal Distribution Comparison
# 
# Compare the distribution of field values at each marginal.
# Shows Reference vs Forward Policy vs Backward Policy.

# %%
def plot_field_distributions(
    reference: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    zt: np.ndarray,
    n_bins: int = 50,
) -> plt.Figure:
    """Plot histograms of field value distributions at each marginal.
    
    Args:
        reference: Reference fields, shape (T, N, H, W)
        forward: Forward policy generated fields, shape (T, N, H, W)
        backward: Backward policy generated fields, shape (T, N, H, W)
        zt: Time values
        n_bins: Number of histogram bins
    
    Returns:
        Matplotlib figure
    """
    T = reference.shape[0]
    
    # Select time steps
    if T > 6:
        time_indices = np.linspace(0, T - 1, 6, dtype=int)
    else:
        time_indices = np.arange(T)
    
    n_cols = len(time_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]
    
    for col, t_idx in enumerate(time_indices):
        ax = axes[col]
        
        ref_vals = reference[t_idx].ravel()
        fwd_vals = forward[t_idx].ravel()
        bwd_vals = backward[t_idx].ravel()
        
        # Compute histogram range
        vmin = min(ref_vals.min(), fwd_vals.min(), bwd_vals.min())
        vmax = max(ref_vals.max(), fwd_vals.max(), bwd_vals.max())
        bins = np.linspace(vmin, vmax, n_bins)
        
        ax.hist(ref_vals, bins=bins, alpha=0.5, density=True, label='REFERENCE', color='green')
        ax.hist(fwd_vals, bins=bins, alpha=0.4, density=True, label='Forward', color='blue')
        ax.hist(bwd_vals, bins=bins, alpha=0.4, density=True, label='Backward', color='orange')
        
        ax.set_xlabel('Field Value', fontsize=10)
        if col == 0:
            ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f't = {zt[t_idx]:.3f}', fontsize=11)
        if col == n_cols - 1:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Field Value Distributions: Reference vs Generated', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("GENERATING FIELD VALUE DISTRIBUTION COMPARISON")
print("="*60)
fig = plot_field_distributions(reference_fields, forward_fields, backward_fields, zt_np)
save_path = eval_dir / "field_distributions.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*70)
print("GRF FIELD VISUALIZATION SUMMARY")
print("="*70)
print(f"\n📁 Output directory: {eval_dir}\n")

print("✓ REFERENCE Field Snapshots:")
print("  - REFERENCE_field_snapshots.png")

print("\n✓ Generated Field Snapshots:")
print("  - forward_policy_field_snapshots.png")
print("  - backward_policy_field_snapshots.png")

print("\n✓ Field Evolution Animations:")
print("  - REFERENCE_field_evolution_sample0.gif")
print("  - forward_policy_field_evolution_sample*.gif")
print("  - backward_policy_field_evolution_sample0.gif")

print("\n✓ Comparison Plots (Generated vs Reference):")
print("  - forward_vs_reference_comparison.png")
print("  - backward_vs_reference_comparison.png")
print("  - generation_error_over_time.png")
print("  - field_distributions.png")

print("\n✓ Spatial Correlation:")
print("  - REFERENCE_spatial_correlation.png")
print("  - forward_policy_spatial_correlation.png")
print("  - backward_policy_spatial_correlation.png")

print("\n✓ Detailed Trajectory Comparisons:")
print("  - forward_trajectory_sample*.png")
print("  - backward_trajectory_sample*.png")

print("\n📊 Key Metrics:")
print(f"  - Number of marginals (T): {T}")
print(f"  - Samples visualized: {n_infer}")
print(f"  - Trajectory steps per interval: {max(10, n_traj_steps // (T - 1))}")
print(f"  - Field resolution: {resolution}×{resolution}")
print(f"  - PCA components: {pca_components.shape[0]}")

print("\n📈 FORWARD POLICY ERROR:")
print(f"  - Overall RMSE: {errors_forward['overall_rmse']:.4e}")
print(f"  - Overall Relative Error: {errors_forward['overall_rel_error']*100:.2f}%")

print("\n📈 BACKWARD POLICY ERROR:")
print(f"  - Overall RMSE: {errors_backward['overall_rmse']:.4e}")
print(f"  - Overall Relative Error: {errors_backward['overall_rel_error']*100:.2f}%")

print("\n🎨 Plot format: Publication-ready (Times New Roman font, 150 DPI)")
print("="*70)


# %%
print("\n" + "="*60)
print("AUTOENCODER RECONSTRUCTION BASELINE")
print("="*60)

# Encode reference → latent → decode → ambient
# This tests the round-trip accuracy of the autoencoder

# The latent_ref is already the encoded version of x_test
# Decode it back to ambient space
reconstructed_ambient = agent.decode_trajectories(latent_ref, zt_np)
print(f"Reconstructed ambient shape: {reconstructed_ambient.shape}")

# Lift to fields
reconstructed_fields = lift_coefficients_to_fields(reconstructed_ambient, pca_info, resolution)
print(f"Reconstructed fields shape: {reconstructed_fields.shape}")

# Compute reconstruction error
errors_ae_recon = compute_field_errors(reference_fields, reconstructed_fields)


# %% [markdown]
# # ## Latent Space Trajectory Visualization
# # 
# # Visualize the generated trajectories in latent space to diagnose issues.
# # Compare:
# # - Reference latent marginals (ground truth encoded data)
# # - Forward policy generated trajectories
# # - Backward policy generated trajectories

# %%
from sklearn.decomposition import PCA

def visualize_latent_trajectories(
    latent_ref: np.ndarray,
    traj_forward: np.ndarray,
    traj_backward: np.ndarray,
    times_forward: np.ndarray,
    times_backward: np.ndarray,
    zt_np: np.ndarray,
    n_samples_to_plot: int = 5,
) -> plt.Figure:
    """Visualize latent space trajectories using PCA projection.
    
    Args:
        latent_ref: Reference latent marginals, shape (T, N, K)
        traj_forward: Forward generated trajectory, shape (S_f, N, K)
        traj_backward: Backward generated trajectory, shape (S_b, N, K)
        times_forward: Forward trajectory times, shape (S_f,)
        times_backward: Backward trajectory times, shape (S_b,)
        zt_np: Reference marginal times, shape (T,)
        n_samples_to_plot: Number of sample trajectories to plot
    """
    T, N, K = latent_ref.shape
    
    # Flatten all latent points for PCA fitting
    all_points = np.concatenate([
        latent_ref.reshape(-1, K),
        traj_forward.reshape(-1, K),
        traj_backward.reshape(-1, K),
    ], axis=0)
    
    # Fit PCA on all points
    pca = PCA(n_components=2)
    pca.fit(all_points)
    
    # Project reference marginals
    ref_projected = pca.transform(latent_ref.reshape(-1, K)).reshape(T, N, 2)
    
    # Project trajectories
    fwd_projected = pca.transform(traj_forward.reshape(-1, K)).reshape(-1, N, 2)
    bwd_projected = pca.transform(traj_backward.reshape(-1, K)).reshape(-1, N, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: Reference marginals ---
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    for t_idx in range(T):
        ax.scatter(ref_projected[t_idx, :, 0], ref_projected[t_idx, :, 1], 
                   c=[colors[t_idx]], alpha=0.3, s=20, label=f't={zt_np[t_idx]:.3f}')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Reference Latent Marginals', fontsize=14)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Forward trajectories ---
    ax = axes[1]
    # Plot reference marginals as background
    for t_idx in range(T):
        ax.scatter(ref_projected[t_idx, :, 0], ref_projected[t_idx, :, 1], 
                   c=[colors[t_idx]], alpha=0.1, s=10)
    # Plot forward trajectories
    for sample_idx in range(min(n_samples_to_plot, N)):
        traj_sample = fwd_projected[:, sample_idx, :]  # (S, 2)
        ax.plot(traj_sample[:, 0], traj_sample[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax.scatter(traj_sample[0, 0], traj_sample[0, 1], c='green', s=50, marker='o', zorder=5)
        ax.scatter(traj_sample[-1, 0], traj_sample[-1, 1], c='red', s=50, marker='x', zorder=5)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Forward Policy Trajectories\n(green=start, red=end)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: Backward trajectories ---
    ax = axes[2]
    for t_idx in range(T):
        ax.scatter(ref_projected[t_idx, :, 0], ref_projected[t_idx, :, 1], 
                   c=[colors[t_idx]], alpha=0.1, s=10)
    for sample_idx in range(min(n_samples_to_plot, N)):
        traj_sample = bwd_projected[:, sample_idx, :]
        ax.plot(traj_sample[:, 0], traj_sample[:, 1], 'orange', alpha=0.5, linewidth=1)
        # Backward: generated from t=1 (index -1) to t=0 (index 0)
        # So green (start) should be at [-1] and red (end) should be at [0]
        ax.scatter(traj_sample[-1, 0], traj_sample[-1, 1], c='green', s=50, marker='o', zorder=5)
        ax.scatter(traj_sample[0, 0], traj_sample[0, 1], c='red', s=50, marker='x', zorder=5)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Backward Policy Trajectories\n(green=start@t=1, red=end@t=0)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_latent_marginal_comparison(
    latent_ref: np.ndarray,
    traj_forward: np.ndarray,
    traj_backward: np.ndarray,
    times_forward: np.ndarray,
    times_backward: np.ndarray,
    zt_np: np.ndarray,
) -> plt.Figure:
    """Compare generated latent points at marginal times vs reference.
    
    NOTE: With knot sampling, traj_forward/backward are already at marginal times (T, N, K).
    """
    T, N, K = latent_ref.shape
    
    # With knot sampling, trajectories are already at marginal times
    # Check if shapes match expected marginal shape
    if traj_forward.shape[0] == len(zt_np):
        fwd_at_marginals = traj_forward
        bwd_at_marginals = traj_backward
    else:
        # Fallback to sampling (for dense trajectories)
        fwd_at_marginals = sample_trajectory_at_marginals(traj_forward, times_forward, zt_np)
        bwd_at_marginals = sample_trajectory_at_marginals(traj_backward, times_backward, zt_np)
    
    # Compute statistics per dimension (first few dims)
    n_dims_to_show = min(6, K)
    
    fig, axes = plt.subplots(2, n_dims_to_show, figsize=(3 * n_dims_to_show, 8))
    
    for dim in range(n_dims_to_show):
        # Mean comparison
        ax = axes[0, dim]
        ref_mean = latent_ref[:, :, dim].mean(axis=1)
        fwd_mean = fwd_at_marginals[:, :, dim].mean(axis=1)
        bwd_mean = bwd_at_marginals[:, :, dim].mean(axis=1)
        
        ax.plot(zt_np, ref_mean, 'g-o', label='Reference', linewidth=2)
        ax.plot(zt_np, fwd_mean, 'b--s', label='Forward', alpha=0.7)
        ax.plot(zt_np, bwd_mean, 'orange', linestyle='--', marker='^', label='Backward', alpha=0.7)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Mean (dim {dim})')
        ax.set_title(f'Latent Dim {dim}: Mean')
        if dim == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Std comparison
        ax = axes[1, dim]
        ref_std = latent_ref[:, :, dim].std(axis=1)
        fwd_std = fwd_at_marginals[:, :, dim].std(axis=1)
        bwd_std = bwd_at_marginals[:, :, dim].std(axis=1)
        
        ax.plot(zt_np, ref_std, 'g-o', label='Reference', linewidth=2)
        ax.plot(zt_np, fwd_std, 'b--s', label='Forward', alpha=0.7)
        ax.plot(zt_np, bwd_std, 'orange', linestyle='--', marker='^', label='Backward', alpha=0.7)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'Std (dim {dim})')
        ax.set_title(f'Latent Dim {dim}: Std')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Statistics: Reference vs Generated at Marginal Times', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# %%
print("\n" + "="*60)
print("LATENT SPACE TRAJECTORY VISUALIZATION")
print("="*60)

# Visualize latent trajectories
fig = visualize_latent_trajectories(
    latent_ref,
    traj_forward,
    traj_backward,
    times_forward,
    times_backward,
    zt_np,
    n_samples_to_plot=10,
)
save_path = eval_dir / "latent_trajectories_pca.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %%
# Compare latent statistics at marginal times
fig = visualize_latent_marginal_comparison(
    latent_ref,
    traj_forward,
    traj_backward,
    times_forward,
    times_backward,
    zt_np,
)
save_path = eval_dir / "latent_marginal_statistics.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %%
# Diagnostic: Check time parameterization
print("\n" + "="*60)
print("DIAGNOSTIC: TIME PARAMETERIZATION")
print("="*60)

print(f"\n1. Autoencoder times (zt):")
print(f"   zt = {zt_np}")
print(f"   Range: [{zt_np.min():.4f}, {zt_np.max():.4f}]")

print(f"\n2. MSBM training times (t_dists):")
t_dists_np = agent.t_dists.cpu().numpy()
print(f"   t_dists = {t_dists_np}")
print(f"   Range: [{t_dists_np.min():.4f}, {t_dists_np.max():.4f}]")

print(f"\n3. t_scale = {t_scale}")

print(f"\n4. Generated trajectory time ranges:")
print(f"   Forward: [{times_forward.min():.4f}, {times_forward.max():.4f}]")
print(f"   Backward: [{times_backward.min():.4f}, {times_backward.max():.4f}]")

print(f"\n5. Latent statistics at boundaries:")
print(f"   Reference at t=0: mean={latent_ref[0].mean():.4f}, std={latent_ref[0].std():.4f}")
print(f"   Reference at t=1: mean={latent_ref[-1].mean():.4f}, std={latent_ref[-1].std():.4f}")
print(f"   Forward start:    mean={traj_forward[0].mean():.4f}, std={traj_forward[0].std():.4f}")
print(f"   Forward end:      mean={traj_forward[-1].mean():.4f}, std={traj_forward[-1].std():.4f}")
print(f"   Backward start:   mean={traj_backward[0].mean():.4f}, std={traj_backward[0].std():.4f}")
print(f"   Backward end:     mean={traj_backward[-1].mean():.4f}, std={traj_backward[-1].std():.4f}")

# %% [markdown]
# # ## Potential Issues and Fixes
# # 
# # Based on the diagnostics, here are the potential issues causing incorrect field generation:
# # 
# # ### Issue 1: SDE Drift Direction
# # The MSBM SDE uses a learned drift term. If the policy outputs are not correctly scaled
# # or the time integration is wrong, trajectories can diverge.
# # 
# # ### Issue 2: Noise Accumulation
# # The SDE adds noise at each step. Over many steps, this can cause significant drift
# # from the target distribution.
# # 
# # ### Issue 3: Time Mismatch
# # The autoencoder uses `zt` times while MSBM uses `t_dists`. These might not align.
# # 
# # ### Fix Attempt: Use Direct IPF Sampling
# # Instead of full trajectory generation, try the agent's built-in inference methods.

# %%
# FIX ATTEMPT 1: Use agent's built-in infer method (single interval)
print("\n" + "="*60)
print("FIX ATTEMPT 1: Use agent.infer() for single-step generation")
print("="*60)

# The agent has an infer method that samples from one marginal to the next
# Let's try using it directly

@torch.no_grad()
def generate_via_infer(agent, start_marginal_idx: int, direction: str = "forward"):
    """Generate samples using agent's infer method."""
    T = len(agent.zt)
    
    if direction == "forward":
        y_current = agent.latent_test[start_marginal_idx, idx].clone()
        results = [y_current.cpu().numpy()]
        
        for i in range(start_marginal_idx, T - 1):
            # Infer from marginal i to i+1
            y_next = agent.infer(
                y_current, 
                t_idx_start=i, 
                t_idx_end=i + 1, 
                direction="forward",
                n_steps=100,
            )
            results.append(y_next.cpu().numpy())
            y_current = y_next
    else:
        y_current = agent.latent_test[start_marginal_idx, idx].clone()
        results = [y_current.cpu().numpy()]
        
        for i in range(start_marginal_idx, 0, -1):
            y_next = agent.infer(
                y_current,
                t_idx_start=i,
                t_idx_end=i - 1,
                direction="backward",
                n_steps=100,
            )
            results.append(y_next.cpu().numpy())
            y_current = y_next
        results = results[::-1]  # Reverse to go from t=0 to t=1
    
    return np.stack(results, axis=0)  # (T, N, K)

# Check if agent has infer method
if hasattr(agent, 'infer'):
    print("Agent has infer() method - using it for generation")
    try:
        latent_forward_fixed = generate_via_infer(agent, 0, "forward")
        latent_backward_fixed = generate_via_infer(agent, T-1, "backward")
        print(f"✓ Forward infer shape: {latent_forward_fixed.shape}")
        print(f"✓ Backward infer shape: {latent_backward_fixed.shape}")
    except Exception as e:
        print(f"✗ infer() failed: {e}")
        latent_forward_fixed = None
        latent_backward_fixed = None
else:
    print("Agent does not have infer() method")
    latent_forward_fixed = None
    latent_backward_fixed = None

# %%
# FIX ATTEMPT 2: Reduced noise / deterministic generation
print("\n" + "="*60)
print("FIX ATTEMPT 2: Deterministic (ODE) trajectory generation")
print("="*60)

@torch.no_grad()
def generate_deterministic_trajectory(
    agent: LatentMSBMAgent,
    y_init: torch.Tensor,
    t_init: float,
    t_final: float,
    policy: torch.nn.Module,
    n_steps: int = 100,
) -> np.ndarray:
    """Generate deterministic trajectory (no noise, just drift).
    
    Args:
        t_init, t_final: MSBM time labels passed to the policy.
                        For backward, these should be reversed labels.
    """
    dt = 1.0 / n_steps
    y = y_init.clone()
    traj = [y.cpu().numpy()]
    
    for step in range(n_steps):
        s = step / n_steps  # local time in [0, 1]
        t_current = t_init + s * (t_final - t_init)
        
        # Get drift from policy
        t_tensor = torch.full((y.shape[0],), t_current, device=agent.device)
        drift = policy(y, t_tensor)  # (N, K)
        
        # Euler step (no noise)
        y = y + drift * dt * (t_final - t_init)
        traj.append(y.cpu().numpy())
    
    return np.stack(traj, axis=0)  # (n_steps+1, N, K)


def generate_full_deterministic_trajectory(
    agent: LatentMSBMAgent,
    y_start: torch.Tensor,
    direction: str = "forward",
    n_steps_per_interval: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate full deterministic trajectory across all marginals.
    
    Uses correct time reversal for backward policy.
    """
    policy = agent.z_f if direction == "forward" else agent.z_b
    t_dists = agent.t_dists.cpu().numpy()
    zt_arr = np.array(agent.zt)
    n_intervals = len(t_dists) - 1
    
    all_trajs = []
    all_times = []
    y_current = y_start.clone()
    
    if direction == "forward":
        for i in range(n_intervals):
            t_start, t_end = float(t_dists[i]), float(t_dists[i + 1])
            zt_start, zt_end = float(zt_arr[i]), float(zt_arr[i + 1])
            
            traj_interval = generate_deterministic_trajectory(
                agent, y_current, t_start, t_end, policy, n_steps=n_steps_per_interval
            )
            t_interval = np.linspace(zt_start, zt_end, n_steps_per_interval + 1)
            
            if i == 0:
                all_trajs.append(traj_interval)
                all_times.append(t_interval)
            else:
                all_trajs.append(traj_interval[1:])
                all_times.append(t_interval[1:])
            
            y_current = torch.from_numpy(traj_interval[-1]).float().to(agent.device)
    else:
        # Backward: iterate from last marginal to first, using reversed time labels
        for i in range(n_intervals - 1, -1, -1):
            # Actual marginal indices for zt
            marg_start = i + 1  # starting at last marginal
            marg_end = i        # ending at previous marginal
            zt_start = float(zt_arr[marg_start])
            zt_end = float(zt_arr[marg_end])
            
            # Reversed time labels for policy (MSBM flip)
            rev_i = (n_intervals - 1) - i
            t0_rev = float(t_dists[rev_i])
            t1_rev = float(t_dists[rev_i + 1])
            
            traj_interval = generate_deterministic_trajectory(
                agent, y_current, t0_rev, t1_rev, policy, n_steps=n_steps_per_interval
            )
            t_interval = np.linspace(zt_start, zt_end, n_steps_per_interval + 1)
            
            if i == n_intervals - 1:
                all_trajs.append(traj_interval)
                all_times.append(t_interval)
            else:
                all_trajs.append(traj_interval[1:])
                all_times.append(t_interval[1:])
            
            y_current = torch.from_numpy(traj_interval[-1]).float().to(agent.device)
        
        # Reverse to have increasing time
        all_trajs = list(reversed(all_trajs))
        all_times = list(reversed(all_times))
    
    return np.concatenate(all_trajs, axis=0), np.concatenate(all_times, axis=0)


print("Generating deterministic (ODE) trajectories...")

traj_forward_ode, times_forward_ode = generate_full_deterministic_trajectory(
    agent, y_start_forward, direction="forward", n_steps_per_interval=50
)
traj_backward_ode, times_backward_ode = generate_full_deterministic_trajectory(
    agent, y_start_backward, direction="backward", n_steps_per_interval=50
)

print(f"✓ Forward ODE trajectory shape: {traj_forward_ode.shape}")
print(f"✓ Backward ODE trajectory shape: {traj_backward_ode.shape}")

# %%
# Visualize ODE trajectories vs reference
fig = visualize_latent_trajectories(
    latent_ref,
    traj_forward_ode,
    traj_backward_ode,
    times_forward_ode,
    times_backward_ode,
    zt_np,
    n_samples_to_plot=10,
)
plt.suptitle("Deterministic (ODE) Trajectories vs Reference", fontsize=14, y=1.02)
save_path = eval_dir / "latent_trajectories_ode.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %%
# Decode ODE trajectories and compare fields
print("\n" + "="*60)
print("DECODING DETERMINISTIC (ODE) TRAJECTORIES")
print("="*60)

# Sample at marginal times
forward_ode_at_marginals = sample_trajectory_at_marginals(traj_forward_ode, times_forward_ode, zt_np)
backward_ode_at_marginals = sample_trajectory_at_marginals(traj_backward_ode, times_backward_ode, zt_np)

# Decode to ambient
ambient_forward_ode = agent.decode_trajectories(forward_ode_at_marginals, zt_np)
ambient_backward_ode = agent.decode_trajectories(backward_ode_at_marginals, zt_np)

# Lift to fields
forward_fields_ode = lift_coefficients_to_fields(ambient_forward_ode, pca_info, resolution)
backward_fields_ode = lift_coefficients_to_fields(ambient_backward_ode, pca_info, resolution)

print(f"✓ Forward ODE fields shape: {forward_fields_ode.shape}")
print(f"✓ Backward ODE fields shape: {backward_fields_ode.shape}")

# Compute errors
errors_forward_ode = compute_field_errors(reference_fields, forward_fields_ode)
errors_backward_ode = compute_field_errors(reference_fields, backward_fields_ode)

print(f"\n{'='*70}")
print("ODE GENERATION ERROR SUMMARY")
print(f"{'='*70}")
print(f"Forward ODE - Overall RMSE: {errors_forward_ode['overall_rmse']:.4e}, Rel Error: {errors_forward_ode['overall_rel_error']*100:.2f}%")
print(f"Backward ODE - Overall RMSE: {errors_backward_ode['overall_rmse']:.4e}, Rel Error: {errors_backward_ode['overall_rel_error']*100:.2f}%")
print(f"\nCompare with SDE:")
print(f"Forward SDE - Overall RMSE: {errors_forward['overall_rmse']:.4e}, Rel Error: {errors_forward['overall_rel_error']*100:.2f}%")
print(f"Backward SDE - Overall RMSE: {errors_backward['overall_rmse']:.4e}, Rel Error: {errors_backward['overall_rel_error']*100:.2f}%")

# %%
# Compare field distributions: ODE vs SDE vs Reference
fig, axes = plt.subplots(2, 6, figsize=(18, 8))

T_plot = min(6, T)
time_indices = np.linspace(0, T - 1, T_plot, dtype=int)

for col, t_idx in enumerate(time_indices):
    # Top row: SDE comparison
    ax = axes[0, col]
    ref_vals = reference_fields[t_idx].ravel()
    fwd_vals = forward_fields[t_idx].ravel()
    bwd_vals = backward_fields[t_idx].ravel()
    
    vmin = min(ref_vals.min(), fwd_vals.min(), bwd_vals.min())
    vmax = max(ref_vals.max(), fwd_vals.max(), bwd_vals.max())
    bins = np.linspace(vmin, vmax, 50)
    
    ax.hist(ref_vals, bins=bins, alpha=0.5, density=True, label='REF', color='green')
    ax.hist(fwd_vals, bins=bins, alpha=0.4, density=True, label='FWD-SDE', color='blue')
    ax.hist(bwd_vals, bins=bins, alpha=0.4, density=True, label='BWD-SDE', color='orange')
    ax.set_title(f't={zt_np[t_idx]:.3f}', fontsize=10)
    if col == 0:
        ax.set_ylabel('SDE\nDensity', fontsize=10)
    if col == T_plot - 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Bottom row: ODE comparison
    ax = axes[1, col]
    fwd_ode_vals = forward_fields_ode[t_idx].ravel()
    bwd_ode_vals = backward_fields_ode[t_idx].ravel()
    
    vmin = min(ref_vals.min(), fwd_ode_vals.min(), bwd_ode_vals.min())
    vmax = max(ref_vals.max(), fwd_ode_vals.max(), bwd_ode_vals.max())
    bins = np.linspace(vmin, vmax, 50)
    
    ax.hist(ref_vals, bins=bins, alpha=0.5, density=True, label='REF', color='green')
    ax.hist(fwd_ode_vals, bins=bins, alpha=0.4, density=True, label='FWD-ODE', color='blue')
    ax.hist(bwd_ode_vals, bins=bins, alpha=0.4, density=True, label='BWD-ODE', color='orange')
    ax.set_xlabel('Field Value', fontsize=10)
    if col == 0:
        ax.set_ylabel('ODE\nDensity', fontsize=10)
    if col == T_plot - 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Field Distributions: SDE (top) vs ODE (bottom) Generation', fontsize=14, y=1.02)
plt.tight_layout()
save_path = eval_dir / "field_distributions_sde_vs_ode.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %%


# %%
# Visual comparison: AE reconstruction vs reference
fig, axes = plt.subplots(3, T, figsize=(3*T, 9))

sample_idx = 0
vmin = min(reference_fields[:, sample_idx].min(), reconstructed_fields[:, sample_idx].min())
vmax = max(reference_fields[:, sample_idx].max(), reconstructed_fields[:, sample_idx].max())

for t_idx in range(T):
    # Reference
    ax = axes[0, t_idx]
    ax.imshow(reference_fields[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(f't={zt_np[t_idx]:.3f}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('Reference', fontsize=11, fontweight='bold')
    
    # Reconstructed
    ax = axes[1, t_idx]
    ax.imshow(reconstructed_fields[t_idx, sample_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('AE Recon', fontsize=11)
    
    # Difference
    diff = reference_fields[t_idx, sample_idx] - reconstructed_fields[t_idx, sample_idx]
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    ax = axes[2, t_idx]
    ax.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    if t_idx == 0:
        ax.set_ylabel('Difference', fontsize=11)
    rmse = np.sqrt(np.mean(diff ** 2))
    ax.text(0.5, -0.15, f'RMSE: {rmse:.2e}', transform=ax.transAxes, ha='center', fontsize=8)

plt.suptitle('Autoencoder Reconstruction: Reference vs Encode-Decode Round-Trip', fontsize=14, y=0.98)
plt.tight_layout()
save_path = eval_dir / "ae_reconstruction_baseline.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {save_path.name}")
plt.show()

# %% [markdown]
# # ## Diagnosis Summary and Recommended Fixes
# # 
# # Based on the visualizations above, identify the source of error:
# # 
# # ### If AE reconstruction is good but MSBM generation is bad:
# # **Root cause**: The MSBM policies are not correctly learning the transport.
# # 
# # **Fixes to try**:
# # 1. **Train longer** - increase epochs
# # 2. **Lower noise variance** - reduce `var` parameter (e.g., 0.1 instead of 0.5)
# # 3. **Use exponential contracting noise schedule** - helps with boundary matching
# # 4. **Check policy architecture** - ensure sufficient capacity
# # 5. **Verify time alignment** - ensure `zt` and `t_dists` are consistent
# # 
# # ### If AE reconstruction is also bad:
# # **Root cause**: The autoencoder is not accurately encoding/decoding.
# # 
# # **Fixes to try**:
# # 1. **Retrain autoencoder** with lower reconstruction loss
# # 2. **Increase latent dimension** if too compressed
# # 3. **Check normalization** - ensure data preprocessing is consistent
# # 
# # ### If trajectories diverge from marginal support:
# # **Root cause**: The SDE drift is too strong or noise accumulates.
# # 
# # **Fixes to try**:
# # 1. **Use ODE (deterministic) generation** instead of SDE
# # 2. **Reduce integration steps** but increase accuracy
# # 3. **Use higher-order integrators** (RK4 instead of Euler)


print(f"\n{'='*70}")
print("AUTOENCODER ROUND-TRIP ERROR (encode → decode)")
print(f"{'='*70}")
print(f"Overall RMSE: {errors_ae_recon['overall_rmse']:.4e}")
print(f"Overall Relative Error: {errors_ae_recon['overall_rel_error']*100:.2f}%")

print(f"\nComparison:")
print(f"  AE Round-trip: {errors_ae_recon['overall_rel_error']*100:.2f}%")
print(f"  Forward SDE:   {errors_forward['overall_rel_error']*100:.2f}%")
print(f"  Backward SDE:  {errors_backward['overall_rel_error']*100:.2f}%")
print(f"  Forward ODE:   {errors_forward_ode['overall_rel_error']*100:.2f}%")
print(f"  Backward ODE:  {errors_backward_ode['overall_rel_error']*100:.2f}%")

# %%
# Summary statistics for diagnosis
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("\n📊 ERROR COMPARISON (Relative Error %):")
print("-" * 50)
print(f"  AE Round-trip:  {errors_ae_recon['overall_rel_error']*100:>8.2f}%  ← Baseline")
print(f"  Forward SDE:    {errors_forward['overall_rel_error']*100:>8.2f}%")
print(f"  Backward SDE:   {errors_backward['overall_rel_error']*100:>8.2f}%")
print(f"  Forward ODE:    {errors_forward_ode['overall_rel_error']*100:>8.2f}%")
print(f"  Backward ODE:   {errors_backward_ode['overall_rel_error']*100:>8.2f}%")

# Determine the issue
ae_error = errors_ae_recon['overall_rel_error']
sde_error = min(errors_forward['overall_rel_error'], errors_backward['overall_rel_error'])
ode_error = min(errors_forward_ode['overall_rel_error'], errors_backward_ode['overall_rel_error'])

print("\n🔍 DIAGNOSIS:")
if ae_error > 0.20:
    print("  ⚠️  Autoencoder has HIGH reconstruction error (>20%)")
    print("  → Consider retraining the autoencoder or increasing latent dimension")
elif ode_error < sde_error * 0.8:
    print("  ⚠️  ODE is significantly better than SDE")
    print("  → Noise is causing drift. Reduce variance or use ODE mode for generation")
elif sde_error > ae_error * 3:
    print("  ⚠️  MSBM generation error is much higher than AE baseline")
    print("  → MSBM policies need retraining with different hyperparameters")
else:
    print("  ✓  Errors are within expected range")

print("\n💡 RECOMMENDED NEXT STEPS:")
if sde_error > 0.50:
    print("  1. Try training MSBM with lower noise: var=0.1 or var=0.05")
    print("  2. Use exponential contracting schedule for better boundary matching")
    print("  3. Increase training epochs")
    print("  4. Consider using ODE generation for inference")
else:
    print("  1. Current model quality appears acceptable")
    print("  2. Fine-tune with more training data if available")

print("="*70)

