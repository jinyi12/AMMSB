#!/usr/bin/env python3
"""Compare landmark TCDM approximation against full dataset results.

This script aligns the landmark dataset with the full dataset (based on frame indices)
and evaluates:
1. Spectral consistency (eigenvalue scaling)
2. Subspace alignment (Principal Angles / CCA)
3. Embedding quality on shared points

Example:
    python scripts/pca/compare_landmark_vs_full.py \
        --landmark_path data/cache_landmark_dataset/tran_inclusions/tc_landmark_dataset.pkl \
        --full_path data/cache_pca_precomputed/tran_inclusions/tc_raw_embeddings.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        p = pickle.load(f)
    return p.get("data", p)


def get_principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces spanned by columns of U and V.
    
    Args:
        U: (N, k1) orthogonal matrix
        V: (N, k2) orthogonal matrix
        
    Returns:
        angles: (min(k1, k2),) array of angles in degrees
    """
    # Ensure matrices are orthogonal
    Q_u, _ = linalg.qr(U, mode='economic')
    Q_v, _ = linalg.qr(V, mode='economic')
    
    # Compute SVD of cross-correlation
    # Singular values are cosines of principal angles
    _, S, _ = linalg.svd(Q_u.T @ Q_v)
    
    # Clip for numerical stability
    S = np.clip(S, 0.0, 1.0)
    
    # Convert to angles in degrees
    angles = np.arccos(S) * 180 / np.pi
    return angles


def align_datasets(land_data: dict, full_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Find common points using landmark_idx.
    
    Returns:
        indices_in_land: Indices in landmark dataset (0..1023)
        indices_in_full: Indices in full dataset (from landmark_idx)
    """
    landmark_idx = land_data.get("landmark_idx")
    
    if landmark_idx is None:
        # Fallback to frame matching if landmark_idx missing (unlikely based on check)
        print("Warning: landmark_idx not found, attempting frame matching...")
        frames_land = land_data.get("frames_landmarks") # This was data, not frames?
        # Check if frames_landmarks is actually frames or data
        # Based on check, it was data. So we can't use it.
        raise ValueError("landmark_idx missing and frames_landmarks appears to be data, cannot align.")
        
    print(f"Using landmark_idx for alignment (N={len(landmark_idx)})")
    
    # Indices in landmark dataset are just 0..N-1 because the dataset IS the subset
    indices_land = np.arange(len(landmark_idx))
    
    # Indices in full dataset are the values in landmark_idx
    indices_full = landmark_idx
    
    return indices_land, indices_full


def plot_spectral_comparison(
    land_sv: list[np.ndarray],
    full_sv: list[np.ndarray],
    times: np.ndarray,
    output_dir: Path
):
    """Compare eigenvalue spectra."""
    n_times = len(times)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Determine dimension limit (min of both)
    max_k = min(len(land_sv[0]), len(full_sv[0]))
    
    # N scaling factor: eigenvalues scale with N
    # We normalized by dividing by N or sqrt(N) usually? 
    # Diffusion map eigenvalues are usually normalized to 1 (top one).
    # But singular values from TCDM might not be.
    # Let's normalize top SV to 1 for shape comparison.
    
    correlations = []
    
    for i in range(n_times):
        ax = axes[i]
        
        s_land = land_sv[i][:max_k]
        s_full = full_sv[i][:max_k]
        
        # Normalize
        s_land_norm = s_land / s_land[0]
        s_full_norm = s_full / s_full[0]
        
        ax.plot(s_land_norm, label="Landmark (norm)", alpha=0.7)
        ax.plot(s_full_norm, label="Full (norm)", alpha=0.7, linestyle="--")
        
        ax.set_title(f"t={times[i]:.2f}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
            
        # Pearson correlation of spectra
        corr = np.corrcoef(s_land, s_full)[0, 1]
        correlations.append(corr)
        ax.text(0.5, 0.9, f"Corr: {corr:.4f}", transform=ax.transAxes, ha='center')

    # Summary plot in last axis
    ax = axes[-1]
    ax.plot(times, correlations, "o-")
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Spectral Shape Correlation")
    ax.set_xlabel("Time")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_comparison.png")
    plt.close()


def plot_subspace_alignment(
    land_emb: np.ndarray,
    full_emb: np.ndarray,
    times: np.ndarray,
    output_dir: Path
):
    """Compare subspace alignment pairwise."""
    # land_emb: (T, N_sub, K)
    # full_emb: (T, N_sub, K)
    
    n_times = len(times)
    K = min(land_emb.shape[2], full_emb.shape[2])
    # Compare top 10 dimensions or K
    K_comp = min(K, 20)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    min_angles = []
    max_angles = []
    mean_angles = []
    
    for i in range(n_times):
        ax = axes[i]
        
        U = land_emb[i, :, :K_comp]
        V = full_emb[i, :, :K_comp]
        
        angles = get_principal_angles(U, V)
        
        ax.plot(angles, "o-", markersize=4)
        ax.set_title(f"t={times[i]:.2f} (Top {K_comp} dims)")
        ax.set_xlabel("Principal Angle Index")
        ax.set_ylabel("Angle (Degrees)")
        ax.set_ylim(0, 90)
        ax.grid(True, alpha=0.3)
        
        # Reference lines
        ax.axhline(10, color="green", linestyle="--", alpha=0.5, label="Excellent (<10°)")
        ax.axhline(30, color="orange", linestyle="--", alpha=0.5, label="Fair (<30°)")
        if i == 0:
            ax.legend()
            
        min_angles.append(np.min(angles))
        max_angles.append(np.max(angles))
        mean_angles.append(np.mean(angles))

    # Summary
    ax = axes[-1]
    ax.plot(times, mean_angles, "o-", label="Mean Angle")
    ax.plot(times, min_angles, "s--", label="Min Angle", alpha=0.5)
    ax.set_title("Principal Angles Summary")
    ax.set_ylim(0, 90)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "subspace_alignment.png")
    plt.close()
    
    return np.mean(mean_angles)


def plot_embedding_scatter(
    land_emb: np.ndarray,
    full_emb: np.ndarray,
    indices_land: np.ndarray,
    indices_full: np.ndarray,
    times: np.ndarray,
    output_dir: Path
):
    """Scatter comparison of first 2 (aligned) dimensions."""
    # We need to align signs because PCA has sign indeterminacy
    # Simple heuristic: flip sign if correlation is negative
    
    t_mid_idx = len(times) // 2
    
    # Extract data for middle time
    X = land_emb[t_mid_idx, :, 0]
    Y = full_emb[t_mid_idx, :, 0]
    
    # Check sign
    if np.corrcoef(X, Y)[0, 1] < 0:
        Y = -Y
        
    X2 = land_emb[t_mid_idx, :, 1]
    Y2 = full_emb[t_mid_idx, :, 1]
    if np.corrcoef(X2, Y2)[0, 1] < 0:
        Y2 = -Y2
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dim 1 correlation
    ax = axes[0]
    ax.scatter(X, Y, alpha=0.5, s=10)
    ax.set_xlabel("Landmark Dim 1")
    ax.set_ylabel("Full Dim 1")
    ax.set_title(f"Dimension 1 Correlation (t={times[t_mid_idx]:.2f})\nR={np.corrcoef(X, Y)[0,1]:.4f}")
    ax.grid(True, alpha=0.3)
    
    # Dim 2 correlation
    ax = axes[1]
    ax.scatter(X2, Y2, alpha=0.5, s=10)
    ax.set_xlabel("Landmark Dim 2")
    ax.set_ylabel("Full Dim 2")
    ax.set_title(f"Dimension 2 Correlation (t={times[t_mid_idx]:.2f})\nR={np.corrcoef(X2, Y2)[0,1]:.4f}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "embedding_correlation.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_path", type=str, required=True)
    parser.add_argument("--full_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    land_path = Path(args.landmark_path).resolve()
    full_path = Path(args.full_path).resolve()
    
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = land_path.parent / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LANDMARK vs FULL DATASET COMPARISON")
    print("="*80)
    
    # Load data
    land_data = load_pickle(land_path)
    full_data = load_pickle(full_path)
    
    # Handle object structures
    if "raw_result_landmarks" in land_data:
        land_res = land_data["raw_result_landmarks"]
        times = land_data["times"]
    else:
        raise ValueError("Unknown landmark dataset structure")
        
    if "raw_result" in full_data:
        full_res = full_data["raw_result"]
        # times usually match, but verify
    elif hasattr(full_data, "embeddings_time"):
        full_res = full_data  # It is the result object
    else:
        # Fallback if full_data is the TCDMRawResult itself (pickled directly)
        full_res = full_data
        
    # Align samples
    idxs_land, idxs_full = align_datasets(land_data, full_data)
    
    # Extract subsets
    # Shape: (T, N, K)
    land_emb_sub = land_res.embeddings_time[:, idxs_land, :]
    full_emb_sub = full_res.embeddings_time[:, idxs_full, :]
    
    print(f"\nAligned shapes:")
    print(f"  Landmark subset: {land_emb_sub.shape}")
    print(f"  Full subset:     {full_emb_sub.shape}")
    
    # 1. Spectral Comparison
    print("\nRunning Spectral Comparison...")
    plot_spectral_comparison(land_res.singular_values, full_res.singular_values, times, output_dir)
    
    # 2. Subspace Alignment
    print("Running Subspace Alignment...")
    mean_angle = plot_subspace_alignment(land_emb_sub, full_emb_sub, times, output_dir)
    print(f"  Mean Principal Angle (across time & top dims): {mean_angle:.2f} degrees")
    
    # 3. Direct Embedding Correlation
    print("Running Embedding Correlation...")
    plot_embedding_scatter(land_emb_sub, full_emb_sub, idxs_land, idxs_full, times, output_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
