"""
Analyze whether an autoencoder embedding is monotonically contracting over time.

Notebook-cell formatting (`# %%`) is included so this file can be opened and run
as a "Python notebook" in editors like VSCode.

This script:
1) Loads PCA coefficient marginals from `data/tran_inclusions.npz` (same pattern as
   `scripts/latent_flow_main.py`: drops the first marginal).
2) Loads a pretrained autoencoder via `scripts.latent_flow_main.load_autoencoder`.
3) Encodes each marginal at its physical time t into latent space.
4) Computes spread metrics in latent space vs time and checks monotonicity +
   approximate exponential decay (linear fit in log-space).
"""

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pickle
from scripts.images.field_visualization import format_for_paper

# %%
# Add repo root to path (match other scripts)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.latent_flow_main import load_autoencoder
from scripts.pca_precomputed_utils import load_pca_data
from scripts.utils import build_zt, get_device


# %%
def _sample_pairwise_distances_mean(x: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    """Estimate mean pairwise Euclidean distance via random pairs."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    n = int(x.shape[0])
    k = int(min(n_pairs, n * (n - 1) // 2))
    if k <= 0:
        return float("nan")
    i = rng.integers(0, n, size=k, dtype=np.int64)
    j = rng.integers(0, n, size=k, dtype=np.int64)
    same = (i == j)
    if np.any(same):
        # Ensure i != j without resampling-heavy logic.
        j[same] = (j[same] + 1) % n
    d = np.linalg.norm(x[i] - x[j], axis=1)
    return float(np.mean(d))


def _spread_about_mean(x: np.ndarray) -> float:
    """Mean distance to the marginal mean, i.e. E||x - E[x]||."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] == 0:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(xc, axis=1)))


def _fit_log_linear(t: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Fit log(y) ~ a + b t; returns slope b, intercept a, R^2, and decay=-b."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "decay": float("nan")}
    t_m = t[mask]
    logy = np.log(np.maximum(y[mask], eps))
    slope, intercept = np.polyfit(t_m, logy, 1)
    pred = intercept + slope * t_m
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "decay": float(-slope)}


@torch.no_grad()
def _encode_marginals(
    *,
    encoder,
    frames: np.ndarray,  # (T, N, D)
    zt: np.ndarray,  # (T,)
    device: str,
    max_samples_per_time: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Encode each time marginal; returns (latents_by_t, indices_by_t)."""
    rng = np.random.default_rng(seed)
    latents: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    T = int(frames.shape[0])
    for t_idx in range(T):
        x = frames[t_idx]
        n = int(x.shape[0])
        take = int(min(max_samples_per_time, n)) if max_samples_per_time > 0 else n
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n, dtype=int)

        x_t = torch.from_numpy(x[idx]).to(device=device, dtype=torch.float32)
        t_val = float(zt[t_idx])
        t_t = torch.full((x_t.shape[0],), t_val, device=device, dtype=torch.float32)
        z_t = encoder(x_t, t_t)
        latents.append(z_t.detach().cpu().numpy().astype(np.float32))
        indices.append(idx.astype(int))
    return latents, indices


# %%
def run(
    *,
    data_path: Path,
    ae_checkpoint: Path,
    ae_type: str,
    latent_dim_override: Optional[int],
    max_samples_per_time: int,
    n_pairs: int,
    seed: int,
    nogpu: bool,
    outdir: Path,
    comp_tcdm_path: Optional[Path] = None,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    device_str = get_device(nogpu)
    print(f"Device: {device_str}")

    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load PCA coefficient marginals (match latent_flow_main.py behavior)
    # -------------------------------------------------------------------------
    print(f"Loading PCA data: {data_path}")
    data_tuple = load_pca_data(
        str(data_path),
        0.2,  # test_size unused here, but required by loader
        seed,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    _, _, pca_info, (_, _), full_marginals, marginal_times = data_tuple

    tcdm_latents_by_t = None
    if comp_tcdm_path is not None:
        print(f"Loading TCDM data: {comp_tcdm_path}")
        with open(comp_tcdm_path, "rb") as f:
            tcdm_data = pickle.load(f)
        # Check structure: might be dict with 'data' key or direct dict
        if 'data' in tcdm_data and isinstance(tcdm_data['data'], dict):
             tcdm_inner = tcdm_data['data']
        else:
             tcdm_inner = tcdm_data
        
        # We expect 'frames' or 'latent_train'/'latent_test'
        # based on inspection: 'latent_train' shape=(7, 4000, 376), 'latent_test' shape=(7, 1000, 376)
        # We should combine them if possible, or just use latent_train if that's what we want to compare.
        # Let's try to combine train and test to get full distribution if available.
        t_list_tcdm = []
        
        l_train = tcdm_inner.get('latent_train')
        l_test = tcdm_inner.get('latent_test')
        
        if l_train is not None:
            # shape (T, N_train, D)
            tcdm_latents_by_t = []
            T_tcdm = l_train.shape[0]
            for t_i in range(T_tcdm):
                # combine train and test for this time step if available
                parts = [l_train[t_i]]
                if l_test is not None and l_test.shape[0] > t_i:
                    parts.append(l_test[t_i])
                combined = np.concatenate(parts, axis=0)
                tcdm_latents_by_t.append(combined)
            print(f"Loaded TCDM latents for {len(tcdm_latents_by_t)} time steps.")
        else:
             print("WARNING: Could not find 'latent_train' in TCDM pickle. Skipping comparison.")


    if len(full_marginals) > 0:
        full_marginals = full_marginals[1:]
        if marginal_times is not None:
            marginal_times = marginal_times[1:]

    frames = np.stack(full_marginals, axis=0).astype(np.float32)  # (T, N, D)
    marginals = list(range(frames.shape[0]))
    zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
    zt = np.asarray(zt, dtype=np.float32)

    print(f"Frames: {frames.shape} (T, N, D), D={pca_info.get('data_dim', '?')} (original), PCA_dim={frames.shape[-1]}")
    print(f"Times: T={len(zt)}, zt[0]={zt[0]:.4f}, zt[-1]={zt[-1]:.4f}")

    # -------------------------------------------------------------------------
    # Load autoencoder + encode marginals
    # -------------------------------------------------------------------------
    print(f"Loading autoencoder: {ae_checkpoint} (ae_type={ae_type})")
    encoder, _decoder, ae_config = load_autoencoder(
        ae_checkpoint,
        device_str,
        ae_type=ae_type,
        latent_dim_override=latent_dim_override,
    )
    print(f"AE config: {ae_config}")

    print(f"Encoding marginals (max_samples_per_time={max_samples_per_time})...")
    latents_by_t, idx_by_t = _encode_marginals(
        encoder=encoder,
        frames=frames,
        zt=zt,
        device=device_str,
        max_samples_per_time=max_samples_per_time,
        seed=seed,
    )

    # -------------------------------------------------------------------------
    # Compute TCDM metrics if available
    # -------------------------------------------------------------------------
    tcdm_metrics = {}
    if tcdm_latents_by_t is not None:
        if len(tcdm_latents_by_t) != len(zt):
             print(f"WARNING: TCDM steps {len(tcdm_latents_by_t)} != AE steps {len(zt)}. Truncating/Aligning to min.")
        
        n_steps = min(len(tcdm_latents_by_t), len(zt))
        
        tcdm_pairwise = []
        tcdm_spread = []
        tcdm_norm_mean = []
        
        for t_idx in range(n_steps):
            z = tcdm_latents_by_t[t_idx]
            tcdm_pairwise.append(_sample_pairwise_distances_mean(z, n_pairs, rng))
            tcdm_spread.append(_spread_about_mean(z))
            tcdm_norm_mean.append(float(np.mean(np.linalg.norm(z, axis=1))))
            
        tcdm_metrics['pairwise'] = np.asarray(tcdm_pairwise, dtype=float)
        tcdm_metrics['spread'] = np.asarray(tcdm_spread, dtype=float)
        tcdm_metrics['norm_mean'] = np.asarray(tcdm_norm_mean, dtype=float)

    # -------------------------------------------------------------------------
    # Compute contraction metrics vs time
    # -------------------------------------------------------------------------
    ambient_pairwise: list[float] = []
    latent_pairwise: list[float] = []
    latent_spread: list[float] = []
    latent_norm_mean: list[float] = []

    for t_idx in range(len(zt)):
        x = frames[t_idx][idx_by_t[t_idx]]
        z = latents_by_t[t_idx]

        ambient_pairwise.append(_sample_pairwise_distances_mean(x, n_pairs, rng))
        latent_pairwise.append(_sample_pairwise_distances_mean(z, n_pairs, rng))
        latent_spread.append(_spread_about_mean(z))
        latent_norm_mean.append(float(np.mean(np.linalg.norm(z, axis=1))))

    ambient_pairwise_np = np.asarray(ambient_pairwise, dtype=float)
    latent_pairwise_np = np.asarray(latent_pairwise, dtype=float)
    latent_spread_np = np.asarray(latent_spread, dtype=float)
    latent_norm_mean_np = np.asarray(latent_norm_mean, dtype=float)

    ratio_pairwise = latent_pairwise_np / (ambient_pairwise_np + 1e-12)

    def _monotone_report(y: np.ndarray, *, name: str, tol: float = 0.0) -> None:
        dy = np.diff(y)
        violations = np.where(dy > tol)[0]
        if violations.size == 0:
            print(f"[monotone] {name}: OK (non-increasing)")
            return
        idxs = violations[:10].tolist()
        print(f"[monotone] {name}: {violations.size} violations; first indices={idxs}")

    _monotone_report(latent_pairwise_np, name="latent mean pairwise dist")
    _monotone_report(latent_spread_np, name="latent spread about mean")
    _monotone_report(latent_norm_mean_np, name="latent mean norm")
    
    if tcdm_metrics:
        print("\n--- TCDM Metrics ---")
        _monotone_report(tcdm_metrics['pairwise'], name="TCDM mean pairwise dist")
        _monotone_report(tcdm_metrics['spread'], name="TCDM spread about mean")
        _monotone_report(tcdm_metrics['norm_mean'], name="TCDM mean norm")
    

    fit_pairwise = _fit_log_linear(zt, latent_pairwise_np)
    fit_spread = _fit_log_linear(zt, latent_spread_np)
    print(f"[exp-fit] latent pairwise: decay={fit_pairwise['decay']:.4f}, R2={fit_pairwise['r2']:.4f}")
    print(f"[exp-fit] latent spread:   decay={fit_spread['decay']:.4f}, R2={fit_spread['r2']:.4f}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    format_for_paper()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    if tcdm_metrics:
         ax.plot(zt[:len(tcdm_metrics['pairwise'])], tcdm_metrics['pairwise'], marker="x", linestyle="--", label="TCDM", color='orange')
    ax.plot(zt, latent_pairwise_np, marker="o", label="latent")
    ax.plot(zt, ambient_pairwise_np, marker="o", label="ambient (PCA coeffs)", alpha=0.6)
    ax.set_title("Mean Pairwise Distance vs Time")
    ax.set_xlabel("t")
    ax.set_ylabel("mean ||x_i - x_j||")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(zt, ratio_pairwise, marker="o")
    ax.set_title("Latent/Ambient Pairwise Distance Ratio")
    ax.set_xlabel("t")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(zt, latent_spread_np, marker="o", label="E||z - E[z]||")
    ax.plot(zt, latent_norm_mean_np, marker="o", label="E||z||", alpha=0.8)
    ax.set_title("Latent Scale Metrics vs Time")
    ax.set_xlabel("t")
    ax.set_ylabel("scale")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    # Add TCDM to scale plot (axes[1,0]) which comes before axes[1,1] in code block
    # But wait, we passed it. Let's fix the targeting.
    safe = np.maximum(latent_pairwise_np, 1e-12)
    ax.plot(zt, np.log(safe), marker="o", label="log(mean pairwise)")
    if np.isfinite(fit_pairwise["slope"]):
        ax.plot(
            zt,
            fit_pairwise["intercept"] + fit_pairwise["slope"] * zt,
            linestyle="--",
            label=f"fit (decay={fit_pairwise['decay']:.3f}, R2={fit_pairwise['r2']:.3f})",
        )
    ax.set_title("Exponential Check (Log-Space)")
    ax.set_xlabel("t")
    ax.set_ylabel("log(metric)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    metrics_path = outdir / "contraction_metrics.png"
    fig.savefig(metrics_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {metrics_path}")

    # Optional: quick 2D PCA of latent samples (for qualitative contraction)
    try:
        from sklearn.decomposition import PCA

        all_lat = np.concatenate(latents_by_t, axis=0)
        all_t = np.concatenate([np.full((lat.shape[0],), float(zt[i])) for i, lat in enumerate(latents_by_t)], axis=0)
        pca2 = PCA(n_components=2)
        proj = pca2.fit_transform(all_lat)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        sc = ax2.scatter(proj[:, 0], proj[:, 1], c=all_t, cmap="viridis", s=8, alpha=0.6)
        fig2.colorbar(sc, ax=ax2, label="t")
        ax2.set_title("Latent Samples (2D PCA) Colored by Time")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid(True, alpha=0.25)
        pca_path = outdir / "latent_pca2.png"
        fig2.savefig(pca_path, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {pca_path}")
    except Exception as exc:
        print(f"Skipping PCA plot (missing sklearn or failed): {exc}")


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze monotonic contraction of AE latent embedding over time.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(REPO_ROOT / "data" / "tran_inclusions.npz"),
        help="Path to PCA coeff dataset (npz).",
    )
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default=str(REPO_ROOT / "results" / "2026-01-23T17-16-56-39" / "geodesic_autoencoder_best.pth"),
        help="Path to pretrained autoencoder checkpoint.",
    )
    parser.add_argument(
        "--ae_type",
        type=str,
        default="diffeo",
        choices=["geodesic", "diffeo"],
        help="Type of autoencoder to load.",
    )
    parser.add_argument(
        "--latent_dim_override",
        type=int,
        default=None,
        help="Override latent_dim when diffeo checkpoint doesn't store it.",
    )
    parser.add_argument(
        "--max_samples_per_time",
        type=int,
        default=256,
        help="Max samples per marginal time to encode (controls runtime). Use 0 for all.",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=20000,
        help="Number of random pairs per marginal used to estimate mean pairwise distances.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "results" / "embedding_contraction_analysis" / "2026-01-23T17-16-56-39"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--comp_tcdm_path",
        type=str,
        default=None,
        help="Optional path to TCDM pickle file for comparison (e.g. data/cache_pca_precomputed/tran_inclusions/tc_selected_embeddings_full.pkl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=Path(args.data_path),
        ae_checkpoint=Path(args.ae_checkpoint),
        ae_type=str(args.ae_type),
        latent_dim_override=args.latent_dim_override,
        max_samples_per_time=int(args.max_samples_per_time),
        n_pairs=int(args.n_pairs),
        seed=int(args.seed),
        nogpu=bool(args.nogpu),
        outdir=Path(args.outdir),
        comp_tcdm_path=Path(args.comp_tcdm_path) if args.comp_tcdm_path else None,
    )

