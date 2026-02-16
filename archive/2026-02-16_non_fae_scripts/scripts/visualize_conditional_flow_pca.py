
"""
Visualization script for Latent Flow Matching with PCA Projection.
Replicates the "Training Interpolation" visualization from the notebook but adds
PCA projection to handle high-dimensional latent spaces (308D) effectively.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.latent_flow_main import (
    ExponentialContractingSchedule,
    load_autoencoder,
    LatentFlowMatcher
)
from scripts.pca_precomputed_utils import load_pca_data
from scripts.utils import build_zt, get_device

def main():
    # ============================================================================
    # Configuration
    # ============================================================================
    AE_TYPE = "diffeo"
    DATA_PATH = REPO_ROOT / "data/tran_inclusions.npz"
    # Using the checkpoint paths from the notebook
    AE_CHECKPOINT = REPO_ROOT / "results/2026-01-20T09-25-07-79/geodesic_autoencoder.pth"
    FLOW_CHECKPOINT = REPO_ROOT / "results/2026-01-20T16-14-54-12/latent_flow_model.pth"
    SCORE_CHECKPOINT = REPO_ROOT / "results/2026-01-20T16-14-54-12/score_model.pth"
    OUTPUT_DIR = REPO_ROOT / "results/flow_visualization"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model config
    HIDDEN_DIMS = [512, 512, 512]
    TIME_DIM = 32
    SIGMA_0 = 0.05
    DECAY_RATE = 2.0
    
    # Viz config
    N_INTERP_SAMPLES = 200  # More samples for better density
    INTERP_MODE = "pairwise"
    SPLINE = "pchip"
    
    SEED = 42
    USE_GPU = True
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device_str = get_device(not USE_GPU)
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # ============================================================================
    # Load Data
    # ============================================================================
    print("Loading data...")
    try:
        data_tuple = load_pca_data(
            str(DATA_PATH),
            0.2, # test_size
            SEED,
            return_indices=True,
            return_full=True,
            return_times=True,
        )
        data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple
        
        # Drop first marginal as in notebook
        if len(full_marginals) > 0:
            full_marginals = full_marginals[1:]
            if marginal_times is not None:
                marginal_times = marginal_times[1:]
                
        # Build time array
        marginals = list(range(len(full_marginals)))
        zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
        
        # Stack frames
        frames = np.stack(full_marginals, axis=0).astype(np.float32)
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
        
        print(f"Data Loaded: x_train={x_train.shape}, T={len(zt)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ============================================================================
    # Load Models
    # ============================================================================
    print(f"Loading {AE_TYPE} autoencoder...")
    encoder, decoder, ae_config = load_autoencoder(
        AE_CHECKPOINT,
        device_str,
        ae_type=AE_TYPE,
    )
    latent_dim = ae_config["latent_dim"]
    print(f"Latent dim: {latent_dim}")

    # ============================================================================
    # Prepare Flow Matcher
    # ============================================================================
    schedule = ExponentialContractingSchedule(sigma_0=SIGMA_0, decay_rate=DECAY_RATE)
    
    flow_matcher = LatentFlowMatcher(
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        zt=zt,
        interp_mode=INTERP_MODE,
        spline=SPLINE,
        device=device_str,
    )
    
    print("Encoding marginals...")
    flow_matcher.encode_marginals(x_train, x_test)
    latent_train = flow_matcher.latent_train # (T, N_train, K)
    
    print(f"Sampling {N_INTERP_SAMPLES} training interpolation points...")
    t_batch, y_t_batch, u_t_batch, eps_batch = flow_matcher.sample_location_and_conditional_flow(
        N_INTERP_SAMPLES, return_noise=True
    )

    # ============================================================================
    # PCA Visualization
    # ============================================================================
    print("Fitting PCA to latent trajectories...")
    all_data = latent_train.detach().cpu().numpy().reshape(-1, latent_dim)
    pca = PCA(n_components=2)
    pca.fit(all_data)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Project data
    y_t_np = y_t_batch.detach().cpu().numpy()
    u_t_np = u_t_batch.detach().cpu().numpy()
    t_np = t_batch.detach().cpu().numpy()
    
    # y_t projection
    y_pca = pca.transform(y_t_np)
    
    # u_t projection (vector projection)
    # v_pca = v @ components.T
    u_pca = u_t_np @ pca.components_.T
    
    # Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Trajectories / Positions (PCA)
    ax = axes[0]
    sc1 = ax.scatter(y_pca[:, 0], y_pca[:, 1], c=t_np, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(sc1, ax=ax, label='Time t')
    ax.set_title("Sampled Positions y_t (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Velocity Field Targets (PCA)
    ax = axes[1]
    # Background points for context
    ax.scatter(y_pca[:, 0], y_pca[:, 1], c='gray', s=5, alpha=0.1)
    
    # Quiver plot
    # Normalize magnitudes for visualization if needed, but keeping raw is more accurate for "targets"
    # To see direction clearly, we might want to normalize arrow lengths or scale them
    vel_scale = 1.0 
    Q = ax.quiver(
        y_pca[:, 0], y_pca[:, 1], 
        u_pca[:, 0], u_pca[:, 1], 
        t_np, cmap='viridis', 
        angles='xy', scale_units='xy', scale=20, width=0.003, headwidth=4
    )
    plt.colorbar(Q, ax=ax, label='Time t')
    ax.set_title("Target Velocities u_t (PCA)\n(Arrows colored by Time)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Marginals (Projected)
    ax = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(zt)))
    for i, t_val in enumerate(zt):
        lat_i = latent_train[i].detach().cpu().numpy()
        lat_proj = pca.transform(lat_i)
        ax.scatter(
            lat_proj[:, 0], lat_proj[:, 1], 
            color=colors[i], s=10, alpha=0.2, 
            label=f"t={t_val:.2f}"
        )
        
    ax.set_title("Marginal Distributions (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Legend might be big
    ax.legend(fontsize='x-small', markerscale=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Interpolation (PCA Projected) | Mode: {INTERP_MODE}", fontsize=14)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "training_interpolation_samples_pca.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
