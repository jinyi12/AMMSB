"""
Visualization script for downstream analysis of PCA MMSFM results.
Specifically focuses on visualizing the lifting neighbors in latent and PCA space.
"""

import argparse
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from scripts.pca_precomputed_main import _load_cached_result, _meta_hash, load_pca_data
from diffmap.diffusion_maps import ConvexHullInterpolator

def plot_lifting_neighbors(
    lifter: ConvexHullInterpolator,
    query_point: np.ndarray,
    macro_neighbors: np.ndarray,
    micro_neighbors: np.ndarray,
    weights: np.ndarray,
    lifted_point: np.ndarray,
    out_path: str,
    pca_info: dict = None,
    marginal_idx: int = None,
):
    """
    Visualize the query point and its neighbors in both latent (macro) and 
    PCA/Image (micro) space.
    """
    
    # 1. Latent Space Plot (Scatter)
    # We project to 2D for visualization if latent dim > 2 using first 2 coords
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Macro (Latent) Space
    ax = axes[0]
    ax.set_title("Latent Space (Macro)")
    
    # Plot all macro states for context (subsampled)
    # Using the lifter's stored macro states
    all_macros = lifter.macro_states
    indices = np.random.choice(len(all_macros), size=min(2000, len(all_macros)), replace=False)
    ax.scatter(all_macros[indices, 0], all_macros[indices, 1], c='lightgray', alpha=0.3, label='Background')
    
    # Plot neighbors
    ax.scatter(macro_neighbors[:, 0], macro_neighbors[:, 1], c='blue', s=50, label='Neighbors')
    
    # Plot query point
    ax.scatter(query_point[0], query_point[1], c='red', s=100, marker='*', label='Query')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Micro (PCA) Space - Projection to first 2 PCs
    ax = axes[1]
    ax.set_title("PCA Coefficient Space (Micro, PC1 vs PC2)")
    
    all_micros = lifter.micro_states
    ax.scatter(all_micros[indices, 0], all_micros[indices, 1], c='lightgray', alpha=0.3, label='Background')
    
    ax.scatter(micro_neighbors[:, 0], micro_neighbors[:, 1], c='blue', s=50, label='Neighbors')
    ax.scatter(lifted_point[0], lifted_point[1], c='red', s=100, marker='*', label='Lifted Result')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_path}_scatter.png")
    plt.close()
    
    # 2. Image Reconstruction Grid (if pca_info is provided)
    if pca_info is not None:
        plot_image_grid(
            micro_neighbors, 
            weights, 
            lifted_point, 
            pca_info, 
            f"{out_path}_recon.png"
        )

def reconstruct_image(coeffs, pca_info):
    """Reconstruct image from PCA coefficients."""
    coeffs = np.atleast_2d(coeffs)
    mean = pca_info['mean']
    components = pca_info['components']
    
    if pca_info.get('is_whitened', True):
        explained_variance = pca_info['explained_variance']
        # Whiten: scale coefficients by sqrt(eigenvalues)
        sqrt_eig = np.sqrt(np.maximum(explained_variance, 1e-12))
        coeffs = coeffs * sqrt_eig
        
    reconstructed = coeffs @ components + mean
    return reconstructed

def plot_image_grid(neighbors, weights, lifted, pca_info, out_path):
    """
    Plot a grid showing the neighbors (weighted) and the result.
    """
    n_neighbors = len(neighbors)
    
    # Reconstruct images
    neighbor_imgs = reconstruct_image(neighbors, pca_info)
    lifted_img = reconstruct_image(lifted, pca_info)
    
    # Assume square images
    img_dim = int(np.sqrt(neighbor_imgs.shape[1]))
    if img_dim * img_dim != neighbor_imgs.shape[1]:
        # Try 3 channels?
        if neighbor_imgs.shape[1] % 3 == 0:
             img_dim = int(np.sqrt(neighbor_imgs.shape[1] // 3))
    
    # Setup grid
    cols = min(5, n_neighbors + 1)
    rows = (n_neighbors + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_1d(axes).flatten()
    
    # Plot neighbors
    for i in range(n_neighbors):
        if i < len(axes):
            ax = axes[i]
            img = neighbor_imgs[i].reshape(img_dim, img_dim) # Assuming grayscale for now
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Neighbor {i}\nx{weights[i]:.3f}")
            ax.axis('off')
            
    # Plot result
    if n_neighbors < len(axes):
        ax = axes[n_neighbors]
        img = lifted_img[0].reshape(img_dim, img_dim)
        ax.imshow(img, cmap='gray')
        ax.set_title("Lifted Result")
        ax.axis('off')
        
    # Turn off remaining axes
    for i in range(n_neighbors + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to PCA data npz')
    parser.add_argument('--outdir', type=str, required=True, help='Experiment output directory')
    parser.add_argument('--traj_file', type=str, required=True, help='Path to latent trajectory npy file')
    parser.add_argument('--tc_cache', type=str, default=None, help='Path to tc_embeddings.pkl cache file')
    parser.add_argument('--marginal_idxs', type=int, nargs='+', default=[0, 10, 20], help='Time steps to visualize')
    parser.add_argument('--neighbor_k', type=int, default=16, help='Number of neighbors to use for lifting')
    
    args = parser.parse_args()
    
    # 1. Load Resources
    # Try to find cache file if not provided
    if args.tc_cache is None:
        # Infer from data path similar to precomputed main
        repo_root = Path(__file__).resolve().parent.parent
        data_stem = Path(args.data_path).stem
        cache_base = repo_root / "data" / "cache_pca_precomputed" / data_stem
        args.tc_cache = cache_base / "tc_embeddings.pkl"
    
    tc_cache_path = Path(args.tc_cache)
    if not tc_cache_path.exists():
        print(f"Error: Cache file not found at {tc_cache_path}")
        # Try to look deeply for it
        # Assuming it might be relative to outdir parent?
        # Actually, let's just fail if we can't find it, user should know where it is or use standard paths
        return

    print(f"Loading cache from {tc_cache_path}...")
    # We need to construct expected meta to verify? 
    # Actually, let's bypass verification for this downstream script and just load the data
    with open(tc_cache_path, "rb") as f:
        payload = pickle.load(f)
        tc_data = payload['data']
        
    lifter = tc_data['lifter']
    
    # Load PCA data for reconstruction
    print(f"Loading PCA data from {args.data_path}...")
    npz_data = np.load(args.data_path)
    pca_info = {
        'components': npz_data['pca_components'],
        'mean': npz_data['pca_mean'],
        'explained_variance': npz_data['pca_explained_variance'],
        'is_whitened': bool(npz_data.get('is_whitened', True))
    }
    
    # Load Trajectory
    print(f"Loading trajectory from {args.traj_file}...")
    traj = np.load(args.traj_file) # (T, latent_dim) or (T, N, latent_dim)
    
    # Handle both single trajectory and batch of trajectories
    if traj.ndim == 2:
        # (T, latent_dim) - single trajectory
        traj = traj[:, np.newaxis, :]
    
    # 2. Visualize
    vis_dir = Path(args.outdir) / "vis_downstream"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    nb_k = args.neighbor_k
    
    for t_idx in args.marginal_idxs:
        if t_idx >= len(traj):
            print(f"Warning: Time index {t_idx} is out of bounds for trajectory of length {len(traj)}")
            continue
            
        # Visualize the first sample in the batch for now
        sample_idx = 0 
        query_point = traj[t_idx, sample_idx].astype(np.float64)
        
        # Lift Manually to get neighbors
        # lifter.lift returns (lifted_point, metadata)
        # metadata contains 'indices', 'weights', 'distances'
        
        lifted_point, metadata = lifter.lift(query_point, k=nb_k)
        
        neighbor_indices = metadata['indices']
        weights = metadata['weights']
        
        macro_neighbors = lifter.macro_states[neighbor_indices]
        micro_neighbors = lifter.micro_states[neighbor_indices]
        
        out_base = str(vis_dir / f"time_{t_idx:03d}_sample_{sample_idx}")
        plot_lifting_neighbors(
            lifter,
            query_point,
            macro_neighbors,
            micro_neighbors,
            weights,
            lifted_point,
            out_base,
            pca_info,
            marginal_idx=t_idx
        )
        print(f"Saved visualization for time {t_idx} to {vis_dir}")

if __name__ == '__main__':
    main()
