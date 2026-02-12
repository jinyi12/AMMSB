"""Train a time-conditioned Encoder with Time-Invariant Decoder FAE.

Architecture:
- Encoder: Time-conditioned with 3D positional encoding (x, y, t)
  - Learns smooth interpolation in time
  - Can encode fields at arbitrary time points

- Decoder: Time-invariant with 2D positional encoding (x, y)
  - Decodes purely based on spatial coordinates
  - Latent z must encode all temporal information
  - Produces distinguishable latents across time

The decoder being time-invariant forces the latent space to capture temporal
variations, making latents more interpretable and distinguishable across time.

Features:
- Real-time wandb logging of training loss and evaluation metrics
- Reconstruction visualizations logged to wandb at every eval interval
- Relative error metrics (Rel-MSE) for interpretable error measures
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
from functional_autoencoders.train.metrics import Metric

from scripts.fae.multiscale_dataset_time_invariant import (
    MultiscaleFieldDatasetTimeInvariant,
    load_held_out_data_time_invariant,
)
from scripts.fae.wandb_trainer import WandbAutoencoderTrainer


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def build_positional_encoding(
    key: jax.Array,
    n_freqs: int,
    in_dim: int,
    sigma: float = 1.0,
) -> RandomFourierEncoding:
    """Sample a random Fourier matrix and wrap it in a ``RandomFourierEncoding``.

    Parameters
    ----------
    key : jax.Array
        Random key for sampling.
    n_freqs : int
        Number of frequency components.
    in_dim : int
        Input dimension (3 for encoder with time, 2 for decoder without time).
    sigma : float
        Standard deviation for the random Fourier matrix.
    """
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
    encoder_pooling: str = "deepset",
    encoder_attn_heads: int = 4,
) -> Autoencoder:
    """Build a FAE with time-conditioned encoder and time-invariant decoder.

    The encoder uses 3D positional encoding (x, y, t) to allow smooth
    interpolation in time. The decoder uses 2D positional encoding (x, y)
    only, forcing the latent z to capture all temporal information.

    Parameters
    ----------
    key : jax.Array
        Random key.
    latent_dim : int
        Dimension of the latent space.
    n_freqs : int
        Number of Fourier frequencies for positional encoding.
    fourier_sigma : float
        Standard deviation for random Fourier features.
    decoder_features : tuple[int, ...]
        Hidden layer sizes for decoder MLP.
    encoder_mlp_dim : int
        Hidden dimension for encoder MLP.
    encoder_mlp_layers : int
        Number of hidden layers in encoder MLP.
    encoder_pooling : str
        Pooling operator used to aggregate point-wise features into a fixed
        latent representation. Supported: {"deepset", "attention"}.
    encoder_attn_heads : int
        Number of attention heads for attention pooling (ignored for deepset).
    """
    key, k1, k2 = jax.random.split(key, 3)

    # Encoder: 3D positional encoding (x, y, t) - TIME CONDITIONED
    encoder_pos_enc = build_positional_encoding(
        k1, n_freqs, in_dim=3, sigma=fourier_sigma
    )

    # Decoder: 2D positional encoding (x, y) - TIME INVARIANT
    decoder_pos_enc = build_positional_encoding(
        k2, n_freqs, in_dim=2, sigma=fourier_sigma
    )

    from functional_autoencoders.util.networks.pooling import DeepSetPooling
    try:
        from functional_autoencoders.util.networks.pooling import TransformerAttentionPooling
    except ImportError:  # pragma: no cover - compatibility with older installs
        TransformerAttentionPooling = None
        from functional_autoencoders.util.networks.pooling import (
            MultiheadAttentionPooling as _LegacyMultiheadAttentionPooling,
        )

    if encoder_pooling == "deepset":
        pooling_fn = DeepSetPooling(
            mlp_dim=encoder_mlp_dim,
            mlp_n_hidden_layers=encoder_mlp_layers,
        )
    elif encoder_pooling == "attention":
        if TransformerAttentionPooling is not None:
            pooling_fn = TransformerAttentionPooling(
                n_heads=encoder_attn_heads,
                mlp_dim=encoder_mlp_dim,
                mlp_n_hidden_layers=encoder_mlp_layers,
            )
        else:
            warnings.warn(
                "TransformerAttentionPooling not available; falling back to "
                "MultiheadAttentionPooling.",
                stacklevel=2,
            )
            pooling_fn = _LegacyMultiheadAttentionPooling(
                n_heads=encoder_attn_heads,
                mlp_dim=encoder_mlp_dim,
                mlp_n_hidden_layers=encoder_mlp_layers,
            )
    else:
        raise ValueError(
            f"Unknown encoder_pooling={encoder_pooling!r}. "
            "Expected one of: {'deepset', 'attention'}."
        )

    encoder = PoolingEncoder(
        latent_dim=latent_dim,
        positional_encoding=encoder_pos_enc,
        pooling_fn=pooling_fn,
        is_variational=False,
    )

    decoder = NonlinearDecoder(
        out_dim=1,
        features=decoder_features,
        positional_encoding=decoder_pos_enc,
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
    """
    import pickle
    from functional_autoencoders.train.state import AutoencoderState

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    state = AutoencoderState(
        params=checkpoint["params"],
        batch_stats=checkpoint["batch_stats"],
    )

    if "best_metric_value" in checkpoint:
        print(f"Loaded checkpoint with MSE={checkpoint['best_metric_value']:.6f}")

    return state


# ---------------------------------------------------------------------------
# Custom metrics for time-invariant decoder
# ---------------------------------------------------------------------------


class MSEMetricTimeInvariant(Metric):
    """MSE metric for time-invariant decoder architecture.

    Handles separate encoder (3D) and decoder (2D) coordinates.
    """

    def __init__(self, autoencoder, domain):
        self.autoencoder = autoencoder
        self.domain = domain

    @property
    def name(self) -> str:
        return "MSE"

    @property
    def batched(self) -> bool:
        return True

    def call_batched(self, state, batch, subkey):
        """Compute MSE for a batch with separate encoder/decoder coords."""
        u_dec, x_dec, u_enc, x_enc = batch

        # Convert to JAX arrays
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        # Get model variables
        vars_ = {"params": state.params}
        if state.batch_stats:
            vars_["batch_stats"] = state.batch_stats

        # Forward pass: encoder uses 3D coords, decoder uses 2D coords
        u_hat = self.autoencoder.apply(vars_, u_enc, x_enc, x_dec, train=False)

        # Compute MSE on decoder outputs
        return float(jnp.mean((u_dec - u_hat) ** 2))


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------


def inverse_log_standardize(
    standardized_data: jnp.ndarray,
    log_mean: float,
    log_std: float,
    log_epsilon: float,
) -> jnp.ndarray:
    """Invert log-standardization to recover original positive values."""
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

    Note: Uses 3D coords for encoder (with time) and 2D coords for decoder.
    """
    results = {}
    for ho in held_out_data:
        u_all = ho["u"]           # [N, res^2, 1]
        x_enc = ho["x_enc"]       # [res^2, 3] - with time
        x_dec = ho["x_dec"]       # [res^2, 2] - without time
        n = u_all.shape[0]

        se_sum = 0.0
        u_norm_sq_sum = 0.0
        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            u_batch = jnp.array(u_all[start:end])           # [B, res^2, 1]
            x_enc_batch = jnp.broadcast_to(
                jnp.array(x_enc)[None], (u_batch.shape[0], *x_enc.shape)
            )                                                 # [B, res^2, 3]
            x_dec_batch = jnp.broadcast_to(
                jnp.array(x_dec)[None], (u_batch.shape[0], *x_dec.shape)
            )                                                 # [B, res^2, 2]

            z = autoencoder.encode(state, u_batch, x_enc_batch, train=False)
            u_hat = autoencoder.decode(state, z, x_dec_batch, train=False)

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
    The dataset returns a masked subset of decoder points (not a full regular
    grid). We therefore visualise reconstructions on the decoder point set
    using triangulation.
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
    Note: Uses 3D coords for encoder and 2D coords for decoder.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)  # [res^2, 2]
    times_norm = data["times_normalized"].astype(np.float32)
    resolution = int(data["resolution"])

    # Get actual marginal keys from npz
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

        # Encoder coords: 3D (x, y, t)
        t_col = np.full((grid_coords.shape[0], 1), t_n, dtype=np.float32)
        x_enc = np.concatenate([grid_coords, t_col], axis=1)  # [res^2, 3]
        x_enc_batch = jnp.broadcast_to(jnp.array(x_enc)[None], (n_show, *x_enc.shape))

        # Decoder coords: 2D (x, y) only
        x_dec = grid_coords  # [res^2, 2]
        x_dec_batch = jnp.broadcast_to(jnp.array(x_dec)[None], (n_show, *x_dec.shape))

        z = autoencoder.encode(state, u_batch, x_enc_batch, train=False)
        u_hat = autoencoder.decode(state, z, x_dec_batch, train=False)

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
        description="Train a time-conditioned encoder with time-invariant decoder FAE."
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
    parser.add_argument(
        "--encoder-pooling",
        type=str,
        default="deepset",
        choices=["deepset", "attention"],
        help=(
            "Encoder pooling operator used to aggregate point-wise features into a "
            "mesh-invariant latent. 'deepset' uses mean pooling; 'attention' uses a "
            "single-query multi-head attention pool."
        ),
    )
    parser.add_argument(
        "--encoder-attn-heads",
        type=int,
        default=4,
        help="Number of attention heads for --encoder-pooling=attention.",
    )

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
    parser.add_argument("--data-generator", type=str, default="", help="Data generator type")
    parser.add_argument("--resolution", type=int, default=None, help="Spatial resolution")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--L-domain", type=float, default=None, help="Domain size")
    parser.add_argument("--scale-mode", type=str, default="", help="Scaling mode")

    # Tran-inclusion specific metadata
    parser.add_argument("--tran-D-large", type=float, default=None)
    parser.add_argument("--tran-H-meso-list", type=str, default="")
    parser.add_argument("--tran-inclusion-value", type=float, default=None)
    parser.add_argument("--tran-vol-frac-large", type=float, default=None)
    parser.add_argument("--tran-vol-frac-small", type=float, default=None)

    # Visualization & Evaluation
    parser.add_argument("--n-vis-samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=1,
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
        help="Save best model based on validation MSE",
    )

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fae-multiscale-tid")
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
        wandb_name = args.wandb_name or f"fae_tid_{os.path.basename(args.output_dir)}"

        # Prepare config with proper types
        config = vars(args).copy()
        config["decoder_features_tuple"] = list(decoder_features)
        config["architecture"] = "time_conditioned_encoder_time_invariant_decoder"

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            dir=args.output_dir,
            tags=[args.data_generator, "time_invariant_decoder"] if args.data_generator else ["time_invariant_decoder"],
        )

    key = jax.random.PRNGKey(args.seed)

    # ------------------------------------------------------------------
    # Print architecture info
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ARCHITECTURE: Time-Conditioned Encoder + Time-Invariant Decoder")
    print("=" * 70)
    print("  Encoder: 3D positional encoding (x, y, t) - time conditioned")
    if args.encoder_pooling == "attention":
        print(f"  Encoder pooling: attention (n_heads={args.encoder_attn_heads})")
    else:
        print("  Encoder pooling: deepset (mean)")
    print("  Decoder: 2D positional encoding (x, y) - time invariant")
    print("  Latent space must encode temporal variations")
    print("=" * 70 + "\n")

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
    # Consistency checks
    # ------------------------------------------------------------------
    if args.scale_mode and dataset_meta.get("scale_mode") and args.scale_mode != dataset_meta["scale_mode"]:
        warnings.warn(
            f"CLI --scale-mode={args.scale_mode!r} does not match npz scale_mode={dataset_meta['scale_mode']!r}."
        )
    if args.data_generator and dataset_meta.get("data_generator") and args.data_generator != dataset_meta["data_generator"]:
        warnings.warn(
            f"CLI --data-generator={args.data_generator!r} does not match npz data_generator={dataset_meta['data_generator']!r}."
        )
    if args.resolution and dataset_meta.get("resolution") and int(args.resolution) != int(dataset_meta["resolution"]):
        warnings.warn(
            f"CLI --resolution={args.resolution} does not match npz resolution={dataset_meta['resolution']}."
        )

    # ------------------------------------------------------------------
    # 1. Load data (using time-invariant dataset)
    # ------------------------------------------------------------------
    print("Loading dataset ...")

    train_dataset = MultiscaleFieldDatasetTimeInvariant(
        npz_path=args.data_path,
        train=True,
        train_ratio=args.train_ratio,
        encoder_point_ratio=args.encoder_point_ratio,
        held_out_indices=held_out_indices,
    )
    test_dataset = MultiscaleFieldDatasetTimeInvariant(
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
        encoder_pooling=args.encoder_pooling,
        encoder_attn_heads=args.encoder_attn_heads,
    )

    # ------------------------------------------------------------------
    # 3. Loss + Trainer
    # ------------------------------------------------------------------
    domain = RandomlySampledEuclidean(s=0.0)
    loss_fn = get_loss_fae_fn(autoencoder, domain, beta=args.beta)

    metrics = [MSEMetricTimeInvariant(autoencoder, domain)]

    # Use wandb-enabled trainer if wandb is active
    wandb_run = None if args.wandb_disabled else wandb.run

    # Create visualization callback that logs at EVERY eval interval
    vis_callback = None
    if wandb_run is not None:
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
    ax.set_title("FAE Training Loss (Time-Invariant Decoder)")
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
    held_out_data = load_held_out_data_time_invariant(
        args.data_path, held_out_indices=held_out_indices
    )
    if held_out_data:
        print("\nEvaluating on held-out times ...")
        ho_results = evaluate_interpolation(
            autoencoder, state, held_out_data, batch_size=args.batch_size
        )

        # Log held-out metrics
        if not args.wandb_disabled:
            for t_norm, metrics_dict in ho_results.items():
                wandb.log({
                    f"final/held_out_mse_t{t_norm:.3f}": metrics_dict["mse"],
                    f"final/held_out_rel_mse_t{t_norm:.3f}": metrics_dict["rel_mse"],
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
        "architecture": "time_conditioned_encoder_time_invariant_decoder",
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_dict, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Save checkpoints
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
                "architecture": "time_conditioned_encoder_time_invariant_decoder",
            },
            f,
        )
    print(f"Final model checkpoint saved to {ckpt_file}")

    # Save best model if enabled
    if args.save_best_model:
        trainer.save_best_model_checkpoint()

    # ------------------------------------------------------------------
    # 8. Visualise
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
