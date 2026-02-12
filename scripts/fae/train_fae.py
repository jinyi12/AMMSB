"""Train a time-conditioned Functional Autoencoder on multiscale 2D fields.

The decoder is a coordinate-MLP ``g(z)(x,y,t) = f(z, gamma(x,y,t))`` where
``gamma`` is a ``RandomFourierEncoding`` with a 3D random matrix
``B ~ N(0, sigma^2 I)`` of shape ``[n_freqs, 3]``.  Because time is a
continuous coordinate, the decoder generalises to intermediate times not
seen during training.

Features:
- Real-time wandb logging of training loss and evaluation metrics
- Periodic visualization logging (--vis-interval) showing sample reconstructions
- Relative error metrics (Rel-MSE) for more interpretable error measures
- Automatic exclusion of t=0 (microscale) for tran_inclusion datasets
- Support for log-standardized data (recommended for positive fields)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
import warnings

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.datasets import NumpyLoader
from functional_autoencoders.decoders.nonlinear_decoder import NonlinearDecoder
from functional_autoencoders.domains.off_grid import RandomlySampledEuclidean
from functional_autoencoders.encoders.pooling_encoder import PoolingEncoder
from functional_autoencoders.losses.fae import get_loss_fae_fn
from functional_autoencoders.positional_encodings import RandomFourierEncoding
from functional_autoencoders.train.metrics import MSEMetric

from scripts.fae.multiscale_dataset import (
    MultiscaleFieldDataset,
    load_held_out_data,
)
from scripts.fae.wandb_trainer import WandbAutoencoderTrainer


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def build_positional_encoding(
    key: jax.Array,
    n_freqs: int,
    in_dim: int = 3,
    sigma: float = 1.0,
) -> RandomFourierEncoding:
    """Sample a random Fourier matrix and wrap it in a ``RandomFourierEncoding``."""
    B = jax.random.normal(key, (n_freqs, in_dim)) * sigma
    return RandomFourierEncoding(B=B)


def build_autoencoder(
    key: jax.Array,
    latent_dim: int,
    n_freqs: int,
    fourier_sigma: float,
    decoder_features: tuple[int, ...],
    encoder_mlp_dim: int = 128,
    encoder_mlp_layers: int = 2,
) -> Autoencoder:
    """Build a ``PoolingEncoder + NonlinearDecoder`` autoencoder.

    Both encoder and decoder share the same ``RandomFourierEncoding``.
    """
    key, k1 = jax.random.split(key)
    pos_enc = build_positional_encoding(k1, n_freqs, in_dim=3, sigma=fourier_sigma)

    from functional_autoencoders.util.networks.pooling import DeepSetPooling

    encoder = PoolingEncoder(
        latent_dim=latent_dim,
        positional_encoding=pos_enc,
        pooling_fn=DeepSetPooling(
            mlp_dim=encoder_mlp_dim,
            mlp_n_hidden_layers=encoder_mlp_layers,
        ),
        is_variational=False,
    )

    decoder = NonlinearDecoder(
        out_dim=1,
        features=decoder_features,
        positional_encoding=pos_enc,
    )

    return Autoencoder(encoder=encoder, decoder=decoder)


def load_checkpoint(checkpoint_path: str, autoencoder: Autoencoder) -> object:
    """Load a saved checkpoint and restore model state.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint (.pkl file).
    autoencoder : Autoencoder
        Autoencoder instance to load parameters into.

    Returns
    -------
    state : object
        Restored model state that can be used for inference.

    Example
    -------
    >>> # Build model with same architecture as training
    >>> autoencoder = build_autoencoder(key, latent_dim=32, n_freqs=64, ...)
    >>> state = load_checkpoint("results/fae_tran/best_state.pkl", autoencoder)
    >>> # Use for inference
    >>> z = autoencoder.encode(state, u_batch, x_batch, train=False)
    >>> u_hat = autoencoder.decode(state, z, x_batch, train=False)
    """
    import pickle
    from functional_autoencoders.train.state import AutoencoderState

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Create state with loaded parameters
    state = AutoencoderState(
        params=checkpoint["params"],
        batch_stats=checkpoint["batch_stats"],
    )

    if "best_metric_value" in checkpoint:
        print(f"Loaded checkpoint with MSE={checkpoint['best_metric_value']:.6f}")

    return state


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------


def inverse_log_standardize(
    standardized_data: jnp.ndarray,
    log_mean: float,
    log_std: float,
    log_epsilon: float,
) -> jnp.ndarray:
    """Invert log-standardization to recover original positive values.

    This function reverses the log_standardize transformation applied during
    data preprocessing. The forward transform is:
        1. y = log(x + eps)
        2. z = (y - mean) / std

    The inverse transform is:
        1. y = z * std + mean
        2. x = exp(y) - eps

    Parameters
    ----------
    standardized_data : jnp.ndarray
        Data in standardized log-space (mean=0, std=1).
    log_mean : float
        Mean of log-transformed data (before standardization).
    log_std : float
        Std of log-transformed data (before standardization).
    log_epsilon : float
        Epsilon added before log transform for numerical stability.

    Returns
    -------
    jnp.ndarray
        Original positive values.

    Notes
    -----
    For training with log_standardized data:
    - The decoder learns to predict in standardized log-space
    - No special output activation is needed (MLP outputs unbounded reals)
    - Use this function to convert predictions back to original space
      for visualization or evaluation in physical units
    """
    log_space = standardized_data * log_std + log_mean
    return jnp.exp(log_space) - log_epsilon


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_interpolation(
    autoencoder: Autoencoder,
    state,
    held_out_data: list[dict],
    batch_size: int = 64,
) -> dict:
    """Compute per-time MSE and relative error on held-out (unseen) times.

    Returns a dict mapping ``t_norm`` -> {"mse": float, "rel_mse": float}.
    """
    results = {}
    for ho in held_out_data:
        u_all = ho["u"]  # [N, res^2, 1]
        x = ho["x"]      # [res^2, 3]
        n = u_all.shape[0]

        se_sum = 0.0
        u_norm_sq_sum = 0.0
        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            u_batch = jnp.array(u_all[start:end])           # [B, res^2, 1]
            x_batch = jnp.broadcast_to(
                jnp.array(x)[None], (u_batch.shape[0], *x.shape)
            )                                                 # [B, res^2, 3]

            z = autoencoder.encode(state, u_batch, x_batch, train=False)
            u_hat = autoencoder.decode(state, z, x_batch, train=False)

            se = jnp.sum((u_batch - u_hat) ** 2)
            u_norm_sq = jnp.sum(u_batch ** 2)

            se_sum += float(se)
            u_norm_sq_sum += float(u_norm_sq)
            count += (end - start) * u_all.shape[1]

        mse = se_sum / count
        rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
        results[ho["t_norm"]] = {"mse": mse, "rel_mse": rel_mse}
        print(
            f"  Held-out t={ho['t']:.4f} (t_norm={ho['t_norm']:.4f}): "
            f"MSE={mse:.6f}, Rel-MSE={rel_mse:.6f}"
        )
    return results


def evaluate_train_reconstruction(
    autoencoder: Autoencoder,
    state,
    test_dataloader,
    n_batches: int = 10,
) -> dict:
    """Quick MSE and relative error estimate on the test split of training-time data.

    Returns
    -------
    dict with keys "mse" and "rel_mse"
    """
    se_sum = 0.0
    u_norm_sq_sum = 0.0
    count = 0
    for i, batch in enumerate(test_dataloader):
        if i >= n_batches:
            break
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        z = autoencoder.encode(state, u_enc, x_enc, train=False)
        u_hat = autoencoder.decode(state, z, x_dec, train=False)

        se_sum += float(jnp.sum((u_dec - u_hat) ** 2))
        u_norm_sq_sum += float(jnp.sum(u_dec ** 2))
        count += u_dec.shape[0] * u_dec.shape[1]

    mse = se_sum / max(count, 1)
    rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
    return {"mse": mse, "rel_mse": rel_mse}


def _sorted_marginal_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(
        [k for k in npz.files if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )


def visualize_sample_reconstructions(
    autoencoder: Autoencoder,
    state,
    test_dataloader,
    n_samples: int = 4,
    n_batches: int = 1,
) -> plt.Figure:
    """Create a figure showing sample reconstructions from test set.

    Returns a matplotlib figure with original and reconstructed fields side-by-side.

    Notes
    -----
    The training dataset applies complement masking, so decoder points generally
    do not cover a full regular grid. We therefore visualise the reconstruction
    on the (potentially irregular) decoder point set using triangulation.
    """
    import matplotlib.tri as mtri

    samples_collected = []
    for i, batch in enumerate(test_dataloader):
        if i >= n_batches:
            break
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        z = autoencoder.encode(state, u_enc, x_enc, train=False)
        u_hat = autoencoder.decode(state, z, x_dec, train=False)

        # Store samples
        for j in range(min(n_samples - len(samples_collected), u_dec.shape[0])):
            coords = np.array(x_dec[j])
            if coords.ndim != 2 or coords.shape[-1] < 2:
                continue
            samples_collected.append({
                "coords": coords[:, :2],
                "original": np.array(u_dec[j, :, 0]),
                "reconstructed": np.array(u_hat[j, :, 0]),
            })
            if len(samples_collected) >= n_samples:
                break
        if len(samples_collected) >= n_samples:
            break

    if not samples_collected:
        return None

    n_show = len(samples_collected)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes[:, None]

    for j in range(n_show):
        coords = samples_collected[j]["coords"]
        orig = samples_collected[j]["original"]
        recon = samples_collected[j]["reconstructed"]
        vmin = float(min(orig.min(), recon.min()))
        vmax = float(max(orig.max(), recon.max()))

        x = coords[:, 0]
        y = coords[:, 1]
        try:
            tri = mtri.Triangulation(x, y)
        except Exception:
            tri = None

        # Original
        if tri is not None:
            axes[0, j].tripcolor(tri, orig, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
        else:
            axes[0, j].scatter(x, y, c=orig, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[0, j].set_title(f"Original {j+1}")
        axes[0, j].axis("off")
        axes[0, j].set_aspect("equal")

        # Reconstructed
        if tri is not None:
            axes[1, j].tripcolor(tri, recon, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
        else:
            axes[1, j].scatter(x, y, c=recon, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1, j].set_title(f"Reconstructed {j+1}")
        axes[1, j].axis("off")
        axes[1, j].set_aspect("equal")

        # Compute relative error for this sample
        rel_error = np.linalg.norm(orig - recon) / max(np.linalg.norm(orig), 1e-10)
        axes[1, j].text(
            0.5, -0.05, f"Rel-Err: {rel_error:.3f}",
            transform=axes[1, j].transAxes,
            ha="center", fontsize=8
        )

    fig.tight_layout()
    return fig


def load_dataset_metadata(npz_path: str) -> dict:
    """Load lightweight metadata from the FAE npz without reading field arrays."""
    with np.load(npz_path, allow_pickle=True) as data:
        marginal_keys = _sorted_marginal_keys(data)
        n_samples = None
        n_times = None
        if marginal_keys:
            # Accessing .shape reads only the .npy header inside the npz.
            n_samples = int(data[marginal_keys[0]].shape[0])
            n_times = int(len(marginal_keys))

        meta = {
            "data_generator": str(data.get("data_generator", "")),
            "scale_mode": str(data.get("scale_mode", "")),
            "resolution": int(data["resolution"]) if "resolution" in data else None,
            "data_dim": int(data["data_dim"]) if "data_dim" in data else None,
            "times": np.array(data["times"]).astype(np.float32) if "times" in data else None,
            "times_normalized": (
                np.array(data["times_normalized"]).astype(np.float32)
                if "times_normalized" in data
                else None
            ),
            "held_out_indices": (
                [int(i) for i in np.array(data["held_out_indices"]).tolist()]
                if "held_out_indices" in data
                else []
            ),
            "held_out_times": (
                [float(t) for t in np.array(data["held_out_times"]).tolist()]
                if "held_out_times" in data
                else []
            ),
            "n_samples": n_samples,
            "n_times": n_times,
            "has_log_stats": all(
                k in data for k in ("log_epsilon", "log_mean", "log_std")
            ),
        }

        if meta["has_log_stats"]:
            meta["log_epsilon"] = float(data["log_epsilon"])
            meta["log_mean"] = float(data["log_mean"])
            meta["log_std"] = float(data["log_std"])

        return meta


def parse_held_out_indices_arg(raw: str) -> list[int]:
    if not raw:
        return []
    if raw.strip().lower() in {"none", "null", "no", "false"}:
        return []
    indices: list[int] = []
    seen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def parse_held_out_times_arg(raw: str, times_normalized: np.ndarray) -> list[int]:
    if not raw:
        return []
    if raw.strip().lower() in {"none", "null", "no", "false"}:
        return []
    targets: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        targets.append(float(token))

    indices: list[int] = []
    for t in targets:
        diffs = np.abs(times_normalized - t)
        idx = int(diffs.argmin())
        if diffs[idx] > 1e-6:
            raise ValueError(
                f"Could not match held-out time {t} to dataset times_normalized. "
                f"Closest is {float(times_normalized[idx])} at index {idx}."
            )
        if idx not in indices:
            indices.append(idx)
    return indices


def visualize_reconstructions(
    autoencoder: Autoencoder,
    state,
    npz_path: str,
    output_dir: str,
    n_samples: int = 4,
    held_out_indices: list[int] | None = None,
) -> None:
    """Save side-by-side original vs. reconstructed field images.

    Uses exact marginal keys from npz to avoid floating-point precision issues.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)
    times_norm = data["times_normalized"].astype(np.float32)
    resolution = int(data["resolution"])

    # Get actual marginal keys from npz (handles float precision correctly)
    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", ""))
    )

    if held_out_indices is None:
        held_out_indices = [int(i) for i in data["held_out_indices"]]
    ho_set = set(held_out_indices)

    os.makedirs(output_dir, exist_ok=True)

    for tidx, key in enumerate(marginal_keys):
        t = float(key.replace("raw_marginal_", ""))
        t_n = float(times_norm[tidx])
        tag = "held_out" if tidx in ho_set else "train"

        fields = data[key].astype(np.float32)  # [N, res^2]

        n_show = min(n_samples, fields.shape[0])
        u_batch = jnp.array(fields[:n_show, :, None])  # [n_show, res^2, 1]

        t_col = np.full((grid_coords.shape[0], 1), t_n, dtype=np.float32)
        x = np.concatenate([grid_coords, t_col], axis=1)  # [res^2, 3]
        x_batch = jnp.broadcast_to(jnp.array(x)[None], (n_show, *x.shape))

        z = autoencoder.encode(state, u_batch, x_batch, train=False)
        u_hat = autoencoder.decode(state, z, x_batch, train=False)

        u_hat_np = np.array(u_hat[:, :, 0])  # [n_show, res^2]

        fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        if n_show == 1:
            axes = axes[:, None]
        for j in range(n_show):
            orig = fields[j].reshape(resolution, resolution)
            recon = u_hat_np[j].reshape(resolution, resolution)
            vmin = min(orig.min(), recon.min())
            vmax = max(orig.max(), recon.max())

            axes[0, j].imshow(orig, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[0, j].set_title("Original")
            axes[0, j].axis("off")

            axes[1, j].imshow(recon, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[1, j].set_title("Reconstructed")
            axes[1, j].axis("off")

        fig.suptitle(f"t={t:.4f} ({tag})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"recon_t{tidx}_{tag}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved reconstruction visualisations to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a time-conditioned FAE on multiscale 2D fields."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    # Architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument("--fourier-sigma", type=float, default=1.0)
    parser.add_argument(
        "--decoder-features",
        type=str,
        default="128,128,128,128",
        help="Comma-separated hidden layer sizes for the decoder MLP.",
    )
    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)

    # Masking / loss
    parser.add_argument("--encoder-point-ratio", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1e-4)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-step", type=int, default=10000)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    # Held-out times (for interpolation evaluation)
    parser.add_argument(
        "--held-out-times",
        type=str,
        default="",
        help=(
            "Comma-separated normalised times to treat as held-out. "
            "If set, overrides any held-out indices stored in the npz."
        ),
    )
    parser.add_argument(
        "--held-out-indices",
        type=str,
        default="",
        help=(
            "Comma-separated 0-based time indices to treat as held-out. "
            "Takes precedence over --held-out-times and overrides npz-held-out indices."
        ),
    )

    # Data generation metadata (for documentation/logging)
    parser.add_argument("--data-generator", type=str, default="", help="Data generator type (e.g., grf, tran_inclusion)")
    parser.add_argument("--resolution", type=int, default=None, help="Spatial resolution of data")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples in dataset")
    parser.add_argument("--L-domain", type=float, default=None, help="Domain size")
    parser.add_argument("--scale-mode", type=str, default="", help="Scaling mode (e.g., log_standardize, minmax)")

    # Tran-inclusion specific metadata
    parser.add_argument("--tran-D-large", type=float, default=None, help="Large inclusion diameter")
    parser.add_argument("--tran-H-meso-list", type=str, default="", help="Mesoscale filter sizes (e.g., 1D,1.25D,1.5D,2D,2.5D,3D)")
    parser.add_argument("--tran-inclusion-value", type=float, default=None, help="Inclusion conductivity value")
    parser.add_argument("--tran-vol-frac-large", type=float, default=None, help="Volume fraction of large inclusions")
    parser.add_argument("--tran-vol-frac-small", type=float, default=None, help="Volume fraction of small inclusions")

    # Visualization & Evaluation
    parser.add_argument("--n-vis-samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=50,
        help=(
            "Log reconstruction visualizations to wandb every N epochs (0 to disable). "
            "Visualizations are emitted on evaluation epochs only (see --eval-interval)."
        ),
    )
    parser.add_argument(
        "--eval-n-batches",
        type=int,
        default=10,
        help="Number of batches to use for final evaluation metrics",
    )

    # Checkpointing
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save best model based on validation MSE (in addition to final model)",
    )

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fae-multiscale")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    decoder_features = tuple(int(x) for x in args.decoder_features.split(","))

    # Initialize wandb
    if not args.wandb_disabled:
        wandb_name = args.wandb_name or f"fae_{os.path.basename(args.output_dir)}"

        # Prepare config with proper types
        config = vars(args).copy()
        config["decoder_features_tuple"] = list(decoder_features)

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            dir=args.output_dir,
            tags=[args.data_generator] if args.data_generator else [],
        )

    key = jax.random.PRNGKey(args.seed)

    # ------------------------------------------------------------------
    # Print data generation metadata (if provided)
    # ------------------------------------------------------------------
    dataset_meta = load_dataset_metadata(args.data_path)

    print("\n" + "=" * 70)
    print("DATASET METADATA (from npz)")
    print("=" * 70)
    if dataset_meta.get("data_generator"):
        print(f"  Data generator: {dataset_meta['data_generator']}")
    if dataset_meta.get("resolution") is not None:
        print(f"  Resolution: {dataset_meta['resolution']}")
    if dataset_meta.get("n_samples") is not None:
        print(f"  Number of samples: {dataset_meta['n_samples']}")
    if dataset_meta.get("scale_mode"):
        print(f"  Scaling mode: {dataset_meta['scale_mode']}")
    if dataset_meta.get("n_times") is not None:
        print(f"  Number of times: {dataset_meta['n_times']}")
    if dataset_meta.get("held_out_indices"):
        print(f"  Held-out indices (npz): {dataset_meta['held_out_indices']}")
    print("=" * 70 + "\n")

    if args.data_generator:
        print("\n" + "="*70)
        print("DATA GENERATION METADATA")
        print("="*70)
        print(f"  Data generator: {args.data_generator}")
        if args.resolution:
            print(f"  Resolution: {args.resolution}")
        if args.n_samples:
            print(f"  Number of samples: {args.n_samples}")
        if args.L_domain:
            print(f"  Domain size: {args.L_domain}")
        if args.scale_mode:
            print(f"  Scaling mode: {args.scale_mode}")

        if args.data_generator == "tran_inclusion":
            if args.tran_D_large:
                print(f"  Large inclusion diameter (D): {args.tran_D_large}")
            if args.tran_H_meso_list:
                print(f"  Mesoscale filter sizes: {args.tran_H_meso_list}")
            if args.tran_inclusion_value:
                print(f"  Inclusion conductivity: {args.tran_inclusion_value}")
            if args.tran_vol_frac_large:
                print(f"  Volume fraction (large): {args.tran_vol_frac_large}")
            if args.tran_vol_frac_small:
                print(f"  Volume fraction (small): {args.tran_vol_frac_small}")
        print("="*70 + "\n")

    # ------------------------------------------------------------------
    # Resolve held-out indices for training/evaluation
    # ------------------------------------------------------------------
    held_out_indices: list[int] | None
    if args.held_out_indices:
        held_out_indices = parse_held_out_indices_arg(args.held_out_indices)
    elif args.held_out_times:
        if dataset_meta.get("times_normalized") is None:
            raise ValueError(
                "--held-out-times was provided but dataset has no times_normalized."
            )
        held_out_indices = parse_held_out_times_arg(
            args.held_out_times, dataset_meta["times_normalized"]
        )
    else:
        held_out_indices = None  # Use indices stored in the npz

    # ------------------------------------------------------------------
    # Consistency checks (warn-only; training uses pre-scaled npz fields)
    # ------------------------------------------------------------------
    if args.scale_mode and dataset_meta.get("scale_mode") and args.scale_mode != dataset_meta["scale_mode"]:
        warnings.warn(
            f"CLI --scale-mode={args.scale_mode!r} does not match npz scale_mode={dataset_meta['scale_mode']!r}. "
            "Training uses the pre-scaled values stored in the npz; consider updating your CLI metadata."
        )
    if args.data_generator and dataset_meta.get("data_generator") and args.data_generator != dataset_meta["data_generator"]:
        warnings.warn(
            f"CLI --data-generator={args.data_generator!r} does not match npz data_generator={dataset_meta['data_generator']!r}. "
            "This only affects logging/tags, not training."
        )
    if args.resolution and dataset_meta.get("resolution") and int(args.resolution) != int(dataset_meta["resolution"]):
        warnings.warn(
            f"CLI --resolution={args.resolution} does not match npz resolution={dataset_meta['resolution']}. "
            "Training uses the npz resolution."
        )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading dataset ...")

    # Note: Data is expected to be pre-scaled (e.g., log_standardize or minmax).
    # For log_standardized data (recommended for positive fields):
    #   - Training data is in standardized log-space: (log(x + eps) - mean) / std
    #   - Decoder learns to predict in this space (no special output activation needed)
    #   - Use inverse_log_standardize() to convert predictions back to original space
    #   - For tran_inclusion data, t=0 (microscale) is automatically excluded from training
    train_dataset = MultiscaleFieldDataset(
        npz_path=args.data_path,
        train=True,
        train_ratio=args.train_ratio,
        encoder_point_ratio=args.encoder_point_ratio,
        held_out_indices=held_out_indices,
    )
    test_dataset = MultiscaleFieldDataset(
        npz_path=args.data_path,
        train=False,
        train_ratio=args.train_ratio,
        encoder_point_ratio=args.encoder_point_ratio,
        held_out_indices=held_out_indices,
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    print(
        f"  Train samples: {len(train_dataset)}  |  "
        f"Test samples: {len(test_dataset)}"
    )

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    autoencoder = build_autoencoder(
        key=subkey,
        latent_dim=args.latent_dim,
        n_freqs=args.n_freqs,
        fourier_sigma=args.fourier_sigma,
        decoder_features=decoder_features,
        encoder_mlp_dim=args.encoder_mlp_dim,
        encoder_mlp_layers=args.encoder_mlp_layers,
    )

    # ------------------------------------------------------------------
    # 3. Loss + Trainer
    # ------------------------------------------------------------------
    domain = RandomlySampledEuclidean(s=0.0)
    loss_fn = get_loss_fae_fn(autoencoder, domain, beta=args.beta)

    metrics = [MSEMetric(autoencoder, domain)]

    # Use wandb-enabled trainer if wandb is active
    wandb_run = None if args.wandb_disabled else wandb.run

    # Create visualization callback for periodic logging
    vis_callback = None
    if wandb_run is not None and args.vis_interval > 0:
        def vis_callback(state, epoch):
            """Generate reconstruction visualization for wandb logging."""
            return visualize_sample_reconstructions(
                autoencoder,
                state,
                test_loader,
                n_samples=args.n_vis_samples,
                n_batches=1,
            )

    # Setup best model path if enabled
    best_model_path = None
    if args.save_best_model:
        best_model_path = os.path.join(args.output_dir, "best_state.pkl")

    trainer = WandbAutoencoderTrainer(
        autoencoder=autoencoder,
        loss_fn=loss_fn,
        metrics=metrics,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        wandb_run=wandb_run,
        vis_callback=vis_callback,
        vis_interval=args.vis_interval,
        save_best_model=args.save_best_model,
        best_model_path=best_model_path,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("Starting training ...")
    key, subkey = jax.random.split(key)
    result = trainer.fit(
        key=subkey,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_factor=args.lr_decay_factor,
        max_step=args.max_steps,
        eval_interval=args.eval_interval,
        verbose="full",
    )

    state = result["state"]
    training_loss = result["training_loss_history"]

    # Save training loss
    np.save(
        os.path.join(args.output_dir, "training_loss.npy"),
        np.array(training_loss, dtype=np.float32),
    )

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("FAE Training Loss")
    fig.tight_layout()
    loss_plot_path = os.path.join(args.output_dir, "training_loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    if not args.wandb_disabled:
        wandb.log({"plots/training_loss": wandb.Image(loss_plot_path)})

    # ------------------------------------------------------------------
    # 5. Evaluate on training-time data
    # ------------------------------------------------------------------
    print("\nEvaluating reconstruction on test split ...")
    test_metrics = evaluate_train_reconstruction(
        autoencoder, state, test_loader, n_batches=args.eval_n_batches
    )
    print(f"  Test-split reconstruction MSE: {test_metrics['mse']:.6f}")
    print(f"  Test-split reconstruction Rel-MSE: {test_metrics['rel_mse']:.6f}")

    if not args.wandb_disabled:
        wandb.log({
            "final/test_mse": test_metrics["mse"],
            "final/test_rel_mse": test_metrics["rel_mse"],
        })

    # ------------------------------------------------------------------
    # 6. Evaluate on held-out times
    # ------------------------------------------------------------------
    held_out_data = load_held_out_data(
        args.data_path, held_out_indices=held_out_indices
    )
    if held_out_data:
        print("\nEvaluating on held-out times ...")
        ho_results = evaluate_interpolation(
            autoencoder, state, held_out_data, batch_size=args.batch_size
        )

        # Log held-out metrics
        if not args.wandb_disabled:
            for t_norm, metrics in ho_results.items():
                wandb.log({
                    f"final/held_out_mse_t{t_norm:.3f}": metrics["mse"],
                    f"final/held_out_rel_mse_t{t_norm:.3f}": metrics["rel_mse"],
                })
    else:
        ho_results = {}
        print("\nNo held-out times specified; skipping interpolation evaluation.")

    # Save evaluation results
    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "held_out_results": {
            str(k): {
                "mse": float(v["mse"]),
                "rel_mse": float(v["rel_mse"]),
            }
            for k, v in ho_results.items()
        },
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_dict, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Save checkpoints (BEFORE visualization to ensure they're saved)
    # ------------------------------------------------------------------
    import pickle

    print("\nSaving checkpoints ...")

    # Save final model
    ckpt_file = os.path.join(args.output_dir, "state.pkl")
    with open(ckpt_file, "wb") as f:
        pickle.dump(
            {
                "params": jax.tree.map(np.array, state.params),
                "batch_stats": (
                    jax.tree.map(np.array, state.batch_stats)
                    if state.batch_stats
                    else None
                ),
            },
            f,
        )
    print(f"Final model checkpoint saved to {ckpt_file}")

    # Save best model if enabled
    if args.save_best_model:
        trainer.save_best_model_checkpoint()

    # ------------------------------------------------------------------
    # 8. Visualise (with error handling to prevent crash)
    # ------------------------------------------------------------------
    print("\nGenerating visualisations ...")
    vis_dir = os.path.join(args.output_dir, "figures")

    try:
        visualize_reconstructions(
            autoencoder,
            state,
            args.data_path,
            vis_dir,
            n_samples=args.n_vis_samples,
            held_out_indices=held_out_indices,
        )

        # Upload visualizations to wandb
        if not args.wandb_disabled:
            import glob
            vis_files = glob.glob(os.path.join(vis_dir, "*.png"))
            for vis_file in vis_files:
                img_name = os.path.basename(vis_file).replace(".png", "")
                wandb.log({f"reconstructions/{img_name}": wandb.Image(vis_file)})
    except Exception as e:
        print(f"Warning: Visualization failed with error: {e}")
        print("Training completed successfully. Checkpoint was saved.")
        import traceback
        traceback.print_exc()

    # Finish wandb run
    if not args.wandb_disabled:
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
