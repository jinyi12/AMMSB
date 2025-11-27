"""
Wavelet-based functional autoencoder experiments on mm_data_large.npz.

This script reconstructs the PCA-compressed multimarginal data into 3D random
fields (t, y, x) with shape (5, 32, 32), then trains the new wavelet functional
encoder/decoder pair from the functional_autoencoders package. It includes
helpers for training, quantitative evaluation, and visualising reconstructions
on the held-out split.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

# Ensure we can import the local functional_autoencoders checkout
REPO_ROOT = Path(__file__).resolve().parents[1]
FAE_SRC = REPO_ROOT.parent / "functional_autoencoders" / "src"
if FAE_SRC.exists() and str(FAE_SRC) not in sys.path:
    sys.path.append(str(FAE_SRC))

try:
    import jax
    import jax.numpy as jnp
    from functional_autoencoders.autoencoder import Autoencoder
    from functional_autoencoders.datasets import NumpyLoader
    from functional_autoencoders.decoders.wavelet_functional_decoder import (
        WaveletFunctionalDecoder,
    )
    from functional_autoencoders.domains.off_grid import RandomlySampledEuclidean
    from functional_autoencoders.encoders.wavelet_functional_encoder import (
        WaveletFunctionalEncoder,
    )
    from functional_autoencoders.losses.fae import get_loss_fae_fn
    from functional_autoencoders.train.autoencoder_trainer import AutoencoderTrainer
    from functional_autoencoders.train.metrics import MSEMetric

    HAS_JAX = True
except ModuleNotFoundError:
    HAS_JAX = False


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
    """
    t_norm = times / (times.max() if times.max() != 0 else 1.0)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    tt, yy, xx = np.meshgrid(t_norm, y, x, indexing="ij")
    return np.stack([tt, yy, xx], axis=-1).reshape(-1, 3).astype(np.float32)


@dataclass
class RandomPointSampler:
    """Randomly selects encoder and decoder points from the full grid."""

    n_enc: int
    n_dec: int

    def __call__(self, u, x):
        n_total = x.shape[0]
        enc_idx = np.random.choice(n_total, self.n_enc, replace=False)
        dec_idx = np.random.choice(n_total, self.n_dec, replace=False)
        return u[enc_idx], x[enc_idx], u[dec_idx], x[dec_idx]


class MMFieldDataset(Dataset):
    def __init__(self, fields: np.ndarray, x_grid: np.ndarray, transform=None):
        self.fields = fields
        self.x_grid = x_grid
        self.transform = transform
        self.time_steps = fields.shape[1]
        self.height = fields.shape[3]
        self.width = fields.shape[4]

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        u = self.fields[idx].reshape(-1, 1)
        x = self.x_grid
        if self.transform is None:
            return u, x, u, x
        return self.transform(u, x)

    def get_full(self, idx):
        u = self.fields[idx].reshape(-1, 1)
        return u, self.x_grid


def _parse_scales_arg(value: str) -> Tuple[Tuple[int, int, int], ...]:
    tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("At least one scale triplet is required.")

    scales = []
    for tok in tokens:
        clean = tok.lower().replace("x", "-")
        parts = clean.split("-")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Invalid scale '{tok}'. Use e.g. '1x4x4,1x8x8'."
            )
        try:
            scales.append(tuple(int(p) for p in parts))  # type: ignore[arg-type]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid integer in scale '{tok}'."
            ) from exc
    return tuple(scales)


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
    x_grid: np.ndarray,
    sampler_train: RandomPointSampler,
    sampler_test: RandomPointSampler,
    batch_size: int,
    test_split: float = 0.1,
):
    n_total = fields.shape[0]
    n_test = int(n_total * test_split)
    indices = np.random.permutation(n_total)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_dataset = MMFieldDataset(fields[train_idx], x_grid, transform=sampler_train)
    test_dataset = MMFieldDataset(fields[test_idx], x_grid, transform=sampler_test)

    train_loader = NumpyLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = NumpyLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset


def build_wavelet_model(
    latent_dim: int,
    scales: Tuple[Tuple[int, int, int], ...],
    hidden_dim: int,
    latent_dim_per_level: Optional[int],
    predict_outflow: bool,
    moment_orders: Sequence[int],
    include_mean: bool,
):
    """
    Assemble a wavelet functional autoencoder for spatio-temporal fields.

    - Encoder: WaveletFunctionalEncoder summarising Haar statistics at each scale
    - Decoder: WaveletFunctionalDecoder predicting multiscale coefficients to reconstruct on query points
    """
    encoder = WaveletFunctionalEncoder(
        is_variational=False,
        latent_dim=latent_dim,
        scales=scales,
        hidden_dim=hidden_dim,
        latent_dim_per_level=latent_dim_per_level,
        moment_orders=tuple(moment_orders),
        include_mean=include_mean,
    )
    decoder = WaveletFunctionalDecoder(
        out_dim=1,
        scales=scales,
        hidden_dim=hidden_dim,
        latent_dim_per_level=latent_dim_per_level,
        predict_outflow=predict_outflow,
    )
    autoencoder = Autoencoder(encoder, decoder)
    domain = RandomlySampledEuclidean(0.0)
    return autoencoder, domain


def train_wavelet_fae(
    fields: np.ndarray,
    x_grid: np.ndarray,
    args: argparse.Namespace,
):
    if not HAS_JAX:
        raise ImportError("Install jax, flax, and optax to run training.")

    sampler_train = RandomPointSampler(args.n_enc, args.n_dec)
    sampler_test = RandomPointSampler(args.n_enc, args.n_dec)
    train_loader, test_loader, test_dataset = get_dataloaders(
        fields,
        x_grid,
        sampler_train,
        sampler_test,
        batch_size=args.batch_size,
        test_split=args.test_split,
    )

    key = jax.random.PRNGKey(args.seed)
    autoencoder, domain = build_wavelet_model(
        latent_dim=args.latent_dim,
        scales=args.scales,
        hidden_dim=args.hidden_dim,
        latent_dim_per_level=args.latent_dim_per_level,
        predict_outflow=args.predict_outflow,
        moment_orders=args.moment_orders,
        include_mean=args.include_mean,
    )
    loss_fn = get_loss_fae_fn(autoencoder, domain, beta=args.beta)
    metrics = [MSEMetric(autoencoder, domain)]

    trainer = AutoencoderTrainer(
        autoencoder,
        loss_fn,
        metrics,
        train_loader,
        test_loader,
    )
    result = trainer.fit(
        key=key,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_factor=args.lr_decay_factor,
        max_step=args.max_steps,
        eval_interval=args.eval_interval,
        verbose="metrics",
    )
    state = result["state"]
    return autoencoder, state, test_dataset, result


def reconstruct_example(
    autoencoder,
    state,
    dataset: MMFieldDataset,
    sampler: RandomPointSampler,
    meta: Dict[str, np.ndarray],
    idx: int = 0,
):
    """Run a forward pass on a held-out sample and return np arrays."""
    u_full, x_full = dataset.get_full(idx)
    u_enc, x_enc, _, _ = sampler(u_full, x_full)

    variables = {"params": state.params, "batch_stats": state.batch_stats}
    u_hat = autoencoder.apply(
        variables,
        u_enc[None, ...],
        x_enc[None, ...],
        x_full[None, ...],
        train=False,
    )
    u_hat = np.array(u_hat[0]).reshape(dataset.time_steps, dataset.height, dataset.width)
    u_true = u_full.reshape(dataset.time_steps, dataset.height, dataset.width)

    # Denormalise for display
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
    sampler: RandomPointSampler,
    meta: Dict[str, np.ndarray],
    indices: Sequence[int],
    output_dir: Path,
):
    """Save reconstruction panels and aggregated temporal error curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mse_per_time_all: List[np.ndarray] = []

    for idx in indices:
        u_true, u_hat = reconstruct_example(autoencoder, state, dataset, sampler, meta, idx)
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
    parser = argparse.ArgumentParser(description="Wavelet functional AE on mm_data_large.npz")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data/mm_data_large.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--latent-dim-per-level", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--scales",
        type=_parse_scales_arg,
        default=_parse_scales_arg("2x4x4,1x8x8"),
        help="Comma-separated Haar scales as n_t x n_y x n_x (e.g. '1x4x4,1x8x8').",
    )
    parser.add_argument(
        "--moment-orders",
        type=_parse_int_list,
        default=_parse_int_list("1,2"),
        help="Comma-separated L_p moments to pool per subband (e.g. '1,2').",
    )
    parser.add_argument(
        "--predict-outflow",
        dest="predict_outflow",
        action="store_true",
        default=True,
        help="Enable low-pass outflow heads at each scale (default: on).",
    )
    parser.add_argument(
        "--no-predict-outflow",
        dest="predict_outflow",
        action="store_false",
        help="Disable low-pass outflow heads.",
    )
    parser.add_argument(
        "--exclude-mean",
        dest="include_mean",
        action="store_false",
        help="Drop the per-cell mean statistic in the encoder.",
    )
    parser.add_argument(
        "--include-mean",
        dest="include_mean",
        action="store_true",
        default=True,
        help="Include the per-cell mean statistic (default).",
    )
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-step", type=int, default=200)
    parser.add_argument("--lr-decay-factor", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-enc", type=int, default=512)
    parser.add_argument("--n-dec", type=int, default=512)
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
    x_grid = build_coordinate_grid(meta["times"], height=32, width=32)

    if not HAS_JAX:
        print(
            "JAX/Flax/Optax are not installed in this environment. "
            "Install them to train; dataset reconstruction completed."
        )
        return

    autoencoder, state, test_dataset, _ = train_wavelet_fae(fields, x_grid, args)

    if args.vis_indices is not None:
        vis_indices = [idx for idx in args.vis_indices if 0 <= idx < len(test_dataset)]
    else:
        rng = np.random.default_rng(args.vis_seed)
        vis_indices = rng.choice(
            len(test_dataset),
            size=min(args.vis_samples, len(test_dataset)),
            replace=False,
        )

    sampler_vis = RandomPointSampler(args.n_enc, args.n_dec)
    visualise_reconstructions(
        autoencoder,
        state,
        test_dataset,
        sampler_vis,
        meta,
        vis_indices,
        args.output_dir,
    )
    print(f"Saved reconstruction figures to {args.output_dir}")


if __name__ == "__main__":
    main()
