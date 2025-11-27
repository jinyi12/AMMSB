"""
Spatiotemporal multiscale functional autoencoder experiments on mm_data_large.npz.

This script reconstructs the PCA-compressed multimarginal data into spatio-temporal
fields with shape (T, 1, H, W) and trains the SpatioTemporalMultiscaleEncoder from
``functional_autoencoders`` paired with a configurable decoder via the shared
``trainer_loader`` utilities. It includes helpers for training, quantitative
evaluation, and visualising reconstructions on the held-out split.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

# Ensure we can import the local functional_autoencoders checkout
REPO_ROOT = Path(__file__).resolve().parents[1]
FAE_DIR = REPO_ROOT / "functional_autoencoders"
FAE_SRC = FAE_DIR / "src"
for path in [FAE_DIR, FAE_SRC]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import jax
    from experiments.trainer_loader import get_trainer
    from functional_autoencoders.datasets import NumpyLoader
    from functional_autoencoders.util import fit_trainer_using_config

    HAS_JAX = True
except ModuleNotFoundError:
    HAS_JAX = False
    NumpyLoader = None  # type: ignore
    fit_trainer_using_config = None  # type: ignore


def invert_pca(
    coeffs: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    explained_variance: np.ndarray,
    whitened: bool,
) -> np.ndarray:
    """Inverse the PCA transform to recover a flattened field."""
    recon = coeffs @ components
    if whitened:
        recon = recon * np.sqrt(explained_variance)
    recon = recon + mean
    return recon


def load_mm_fields(
    npz_path: Path, normalise: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load mm_data_large.npz and return reconstructed fields.

    Returns
    -------
    fields : np.ndarray
        Array of shape (n_samples, n_times, 1, 32, 32) in float32.
    meta : dict
        Contains dataset normalisation stats and time stamps.
    """
    data = np.load(npz_path)
    time_keys = sorted(
        [k for k in data.files if k.startswith("marginal_")],
        key=lambda key: float(key.split("_")[1]),
    )
    times = np.array([float(k.split("_")[1]) for k in time_keys], dtype=np.float32)

    components = data["pca_components"]
    mean = data["pca_mean"]
    explained_variance = data["pca_explained_variance"]
    whitened = bool(data["is_whitened"])

    recon_fields = []
    for key in time_keys:
        coeffs = data[key]
        recon = invert_pca(coeffs, components, mean, explained_variance, whitened)
        recon_fields.append(recon.reshape(recon.shape[0], 32, 32))

    fields = np.stack(recon_fields, axis=1)  # (n_samples, n_times, 32, 32)
    fields = fields[:, :, None, :, :]  # add channel

    if normalise:
        mu = fields.mean()
        sigma = fields.std() + 1e-6
        fields = (fields - mu) / sigma
    else:
        mu, sigma = 0.0, 1.0

    meta = {
        "mean": np.array(mu, dtype=np.float32),
        "std": np.array(sigma, dtype=np.float32),
        "times": times,
        "time_keys": time_keys,
    }
    return fields.astype(np.float32), meta


def build_coordinate_grid(times: np.ndarray, height: int = 32, width: int = 32):
    """
    Create a (t, y, x) grid of coordinates in [0, 1]^3 to match field ordering.
    Used for time-dependent decoders.
    """
    t_norm = times / (times.max() if times.max() != 0 else 1.0)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t_norm, y, x, indexing="ij")
    return np.stack([tt, yy, xx], axis=-1).reshape(-1, 3).astype(np.float32)


def build_spatial_grid(height: int = 32, width: int = 32) -> np.ndarray:
    """
    Create spatial (x, y) grid coordinates in [0, 1]^2 centered at pixel centers.
    Shape: (H * W, 2)
    """
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)


def build_temporal_coordinates(times: np.ndarray) -> np.ndarray:
    """
    Normalize time coordinates to [0, 1].
    Shape: (T,)
    """
    t_max = times.max() if times.max() > 0 else 1.0
    return (times / t_max).astype(np.float32)


@dataclass
class DecoderPointSampler:
    """Randomly selects decoder points from the full spatiotemporal grid."""

    n_dec: int

    def __call__(self, u, x):
        n_total = x.shape[0]
        if self.n_dec <= 0 or self.n_dec >= n_total:
            return u, x
        dec_idx = np.random.choice(n_total, self.n_dec, replace=False)
        return u[dec_idx], x[dec_idx]


class MMFieldDataset(Dataset):
    """Dataset for spatiotemporal fields with flexible encoder coordinate support."""

    def __init__(
        self,
        fields: np.ndarray,
        decoder_coords: np.ndarray,
        encoder_coords: np.ndarray,
        temporal_coords: np.ndarray | None = None,
        transform=None,
    ):
        """
        Parameters
        ----------
        fields : np.ndarray
            Shape (N, T, C, H, W) spatiotemporal fields.
        decoder_coords : np.ndarray
            Shape (T*H*W, 3) full spatiotemporal coordinates for decoder.
        encoder_coords : np.ndarray
            Spatial coordinates. Shape depends on encoder type:
            - Order-agnostic: (H*W, 2) spatial coords per timestep
            - Order-preserving: (H*W, 2) spatial coords
        temporal_coords : np.ndarray, optional
            Shape (T,) normalized time coordinates for order-preserving encoder.
        transform : callable, optional
            Transform applied to decoder points (e.g., random subsampling).
        """
        self.fields = fields.astype(np.float32)
        self.decoder_coords = decoder_coords.astype(np.float32)
        self.encoder_coords = encoder_coords.astype(np.float32)
        self.temporal_coords = (
            temporal_coords.astype(np.float32) if temporal_coords is not None else None
        )
        self.transform = transform
        self.time_steps = fields.shape[1]
        self.height = fields.shape[3]
        self.width = fields.shape[4]

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        u_field = self.fields[idx]
        u_flat = u_field.reshape(-1, 1)
        if self.transform is None:
            u_dec, x_dec = u_flat, self.decoder_coords
        else:
            u_dec, x_dec = self.transform(u_flat, self.decoder_coords)
        return u_dec, x_dec, u_field, self.encoder_coords

    def get_full(self, idx):
        return self.fields[idx], self.decoder_coords


def _parse_int_list(value: str) -> Tuple[int, ...]:
    tokens = [tok.strip() for tok in value.replace(" ", "").split(",") if tok.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("Provide at least one integer.")
    try:
        return tuple(int(tok) for tok in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse integer list from '{value}'.") from exc


def get_dataloaders(
    fields: np.ndarray,
    decoder_coords: np.ndarray,
    encoder_coords: np.ndarray,
    temporal_coords: np.ndarray | None,
    sampler_train: DecoderPointSampler | None,
    sampler_test: DecoderPointSampler | None,
    batch_size: int,
    test_split: float = 0.1,
):
    """Create train/test dataloaders with optional temporal coordinates."""
    if NumpyLoader is None:
        raise ImportError("NumpyLoader is unavailable; install functional_autoencoders.")

    n_total = fields.shape[0]
    n_test = max(1, int(n_total * test_split))
    indices = np.random.permutation(n_total)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_dataset = MMFieldDataset(
        fields[train_idx],
        decoder_coords,
        encoder_coords,
        temporal_coords=temporal_coords,
        transform=sampler_train,
    )
    test_dataset = MMFieldDataset(
        fields[test_idx],
        decoder_coords,
        encoder_coords,
        temporal_coords=temporal_coords,
        transform=sampler_test,
    )

    train_loader = NumpyLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = NumpyLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset


def build_experiment_config(args: argparse.Namespace):
    """Build configuration dict for functional autoencoder training."""
    # Build encoder configuration based on type
    encoder_type = args.encoder_type
    base_encoder_params = {
        "scales": tuple(args.encoder_scales),
        "point_mlp_features": tuple(args.point_mlp_features),
        "rho_features": tuple(args.rho_features),
        "patch_feature_dim": args.patch_feature_dim,
    }

    if encoder_type == "spatiotemporal_order_preserving":
        # Add temporal-specific parameters for order-preserving encoder
        base_encoder_params["temporal_hidden_dim"] = args.temporal_hidden_dim

    encoder_options = {encoder_type: base_encoder_params}

    decoder_options = {
        "time_dependent": {
            "out_dim": 1,
            "features": tuple(args.decoder_features),
            "time_fourier_features": args.time_fourier_features,
            "space_fourier_features": args.space_fourier_features,
            "fourier_sigma": args.fourier_sigma,
        }
    }

    config = {
        "encoder": {
            "latent_dim": args.latent_dim,
            "is_variational": False,
            "type": encoder_type,
            "options": encoder_options,
        },
        "decoder": {
            "type": "time_dependent",
            "options": decoder_options,
        },
        "domain": {
            "type": "off_grid_randomly_sampled_euclidean",
            "options": {
                "off_grid_randomly_sampled_euclidean": {"s": 0.0},
            },
        },
        "loss": {
            "type": "fae",
            "options": {
                "fae": {
                    "beta": args.beta,
                    "subtract_data_norm": False,
                }
            },
        },
        "positional_encoding": {
            "is_used": False,  # Spatiotemporal encoder handles coords internally
            "dim": args.encoder_posenc_dim,
        },
        "trainer": {
            "max_step": args.max_steps,
            "lr": args.lr,
            "lr_decay_step": args.lr_decay_step,
            "lr_decay_factor": args.lr_decay_factor,
            "eval_interval": args.eval_interval,
        },
    }
    return config


def train_spatiotemporal_fae(
    fields: np.ndarray,
    decoder_coords: np.ndarray,
    encoder_coords: np.ndarray,
    temporal_coords: np.ndarray | None,
    args: argparse.Namespace,
):
    """Train functional autoencoder on spatiotemporal data."""
    if not HAS_JAX:
        raise ImportError("Install jax, flax, and optax to run training.")

    sampler_train = DecoderPointSampler(args.n_dec) if args.n_dec > 0 else None
    sampler_test = DecoderPointSampler(-1)
    train_loader, test_loader, test_dataset = get_dataloaders(
        fields,
        decoder_coords,
        encoder_coords,
        temporal_coords,
        sampler_train,
        sampler_test,
        batch_size=args.batch_size,
        test_split=args.test_split,
    )

    config = build_experiment_config(args)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    trainer = get_trainer(subkey, config, train_loader, test_loader)

    key, subkey = jax.random.split(key)
    results = fit_trainer_using_config(subkey, trainer, config, verbose="metrics")
    state = results["state"]
    return trainer.autoencoder, state, test_dataset, results


def reconstruct_example(
    autoencoder,
    state,
    dataset: MMFieldDataset,
    meta: Dict[str, np.ndarray],
    idx: int = 0,
):
    """Run a forward pass on a held-out sample and return np arrays."""
    u_field, x_full = dataset.get_full(idx)
    encoder_input = u_field[None, ...]
    encoder_coords = dataset.encoder_coords[None, ...]
    decoder_coords = x_full[None, ...]

    variables = {"params": state.params, "batch_stats": state.batch_stats}
    u_hat = autoencoder.apply(
        variables,
        encoder_input,
        encoder_coords,
        decoder_coords,
        train=False,
    )
    u_hat = np.array(u_hat[0]).reshape(dataset.time_steps, dataset.height, dataset.width)
    u_true = u_field.reshape(dataset.time_steps, dataset.height, dataset.width)

    u_hat = (u_hat * meta["std"]) + meta["mean"]
    u_true = (u_true * meta["std"]) + meta["mean"]
    return u_true, u_hat


def compute_reconstruction_error(u_true: np.ndarray, u_hat: np.ndarray):
    diff = u_true - u_hat
    mse = float(np.mean(diff**2))
    mse_per_time = np.mean(diff**2, axis=(1, 2))
    return mse, mse_per_time


def plot_reconstruction(
    u_true,
    u_hat,
    meta: Dict[str, np.ndarray],
    output_path: Path,
    title_prefix: str = "",
):
    n_times = u_true.shape[0]
    fig, axes = plt.subplots(3, n_times, figsize=(3 * n_times, 9))
    rows = [u_true, u_hat, np.abs(u_true - u_hat)]
    titles = ["true", "recon", "|error|"]

    for row, (arr, label) in enumerate(zip(rows, titles)):
        for i in range(n_times):
            cmap = "magma" if label == "|error|" else "coolwarm"
            axes[row, i].imshow(arr[i], cmap=cmap)
            prefix = f"{title_prefix} " if title_prefix else ""
            axes[row, i].set_title(f"{prefix}t={meta['times'][i]:.2f} {label}")
            axes[row, i].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temporal_mse(
    mse_per_time: np.ndarray, meta: Dict[str, np.ndarray], output_path: Path
):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(mse_per_time)), mse_per_time)
    ax.set_xticks(np.arange(len(mse_per_time)))
    ax.set_xticklabels([f"{t:.2f}" for t in meta["times"]], rotation=45)
    ax.set_ylabel("MSE")
    ax.set_xlabel("t")
    ax.set_title("Per-time reconstruction MSE")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def visualise_reconstructions(
    autoencoder,
    state,
    dataset: MMFieldDataset,
    meta: Dict[str, np.ndarray],
    indices: Sequence[int],
    output_dir: Path,
):
    """Save reconstruction panels and aggregated temporal error curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mse_per_time_all: List[np.ndarray] = []

    for idx in indices:
        u_true, u_hat = reconstruct_example(autoencoder, state, dataset, meta, idx)
        mse, mse_per_time = compute_reconstruction_error(u_true, u_hat)
        mse_per_time_all.append(mse_per_time)

        fname = output_dir / f"reconstruction_{idx}.png"
        plot_reconstruction(u_true, u_hat, meta, fname, title_prefix=f"idx={idx}")
        print(f"[vis] idx={idx} MSE={mse:.4f} -> {fname}")

    if mse_per_time_all:
        stacked = np.stack(mse_per_time_all, axis=0)
        mean_per_time = stacked.mean(axis=0)
        plot_temporal_mse(mean_per_time, meta, output_dir / "temporal_mse.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatiotemporal multiscale AE on mm_data_large.npz"
    )
    # Data and training parameters
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/mm_data_large.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.1)

    # Encoder architecture selection
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["spatiotemporal_multiscale", "spatiotemporal_order_preserving"],
        default="spatiotemporal_order_preserving",
        help="Type of spatiotemporal encoder to use.",
    )
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--encoder-scales",
        type=_parse_int_list,
        default=_parse_int_list("1,2,4,8"),
        help="Comma-separated spatial pooling scales for the encoder.",
    )
    parser.add_argument(
        "--point-mlp-features",
        type=_parse_int_list,
        default=_parse_int_list("128,128,128"),
        help="Hidden sizes for the point MLP inside the encoder.",
    )
    parser.add_argument(
        "--rho-features",
        type=_parse_int_list,
        default=_parse_int_list("128,128,128"),
        help="Hidden sizes for the rho MLP producing latents.",
    )
    parser.add_argument("--patch-feature-dim", type=int, default=64)
    parser.add_argument(
        "--temporal-hidden-dim",
        type=int,
        default=None,
        help="Hidden dim for GRU in order-preserving encoder (None = auto-sized).",
    )

    # Decoder parameters
    parser.add_argument(
        "--decoder-features",
        type=_parse_int_list,
        default=_parse_int_list("128,128,128"),
        help="Hidden sizes for the time-dependent decoder.",
    )
    parser.add_argument("--time-fourier-features", type=int, default=32)
    parser.add_argument("--space-fourier-features", type=int, default=64)
    parser.add_argument("--fourier-sigma", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-step", type=int, default=200)
    parser.add_argument("--lr-decay-factor", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-dec", type=int, default=1024, help="Decoder point budget.")
    parser.add_argument(
        "--encoder-posenc-dim",
        type=int,
        default=32,
        help="Dimensionality of encoder positional encoding (ignored if disabled).",
    )
    parser.add_argument(
        "--disable-encoder-posenc",
        action="store_true",
        help="Disable random Fourier features in the encoder.",
    )
    parser.add_argument(
        "--vis-samples",
        type=int,
        default=3,
        help="Number of held-out reconstructions to visualise.",
    )
    parser.add_argument(
        "--vis-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset indices to visualise (overrides vis-samples).",
    )
    parser.add_argument(
        "--vis-seed",
        type=int,
        default=0,
        help="Random seed for selecting visualisation samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "wavelet_fae",
        help="Directory to store visualisation figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fields, meta = load_mm_fields(args.data_path)

    # Decoder always uses full spatiotemporal grid
    decoder_coords = build_coordinate_grid(meta["times"], height=32, width=32)

    # Encoder coordinates depend on type
    if args.encoder_type == "spatiotemporal_order_preserving":
        # Order-preserving encoder needs spatial coords + separate temporal coords
        encoder_coords = build_spatial_grid(height=32, width=32)
        temporal_coords = build_temporal_coordinates(meta["times"])
    else:
        # Order-agnostic encoder uses spatial coords (applied per timestep internally)
        encoder_coords = build_spatial_grid(height=32, width=32)
        temporal_coords = None

    if not HAS_JAX:
        print(
            "JAX/Flax/Optax are not installed in this environment. "
            "Install them to train; dataset reconstruction completed."
        )
        return

    autoencoder, state, test_dataset, _ = train_spatiotemporal_fae(
        fields,
        decoder_coords,
        encoder_coords,
        temporal_coords,
        args,
    )

    if args.vis_indices is not None:
        vis_indices = [idx for idx in args.vis_indices if 0 <= idx < len(test_dataset)]
    else:
        rng = np.random.default_rng(args.vis_seed)
        vis_indices = rng.choice(
            len(test_dataset),
            size=min(args.vis_samples, len(test_dataset)),
            replace=False,
        )

    visualise_reconstructions(
        autoencoder,
        state,
        test_dataset,
        meta,
        vis_indices,
        args.output_dir,
    )
    print(f"Saved reconstruction figures to {args.output_dir}")


if __name__ == "__main__":
    main()
