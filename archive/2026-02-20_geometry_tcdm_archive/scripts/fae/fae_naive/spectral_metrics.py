"""Off-grid spectral metrics for FAE evaluation.

These metrics avoid full-grid FFT assumptions and operate directly on sampled
points using nonuniform Fourier projections.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from functional_autoencoders.train.metrics import Metric

from scripts.fae.fae_naive.spectral_losses import (
    fourier_mse_by_frequency,
    sample_frequency_vectors,
)


def compute_frequency_binned_error(
    u_true: jnp.ndarray,
    u_pred: jnp.ndarray,
    x: jnp.ndarray,
    freq_vectors: jnp.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute frequency-binned error on off-grid sampled points.

    Parameters
    ----------
    u_true : jnp.ndarray of shape (batch, n_points, 1)
    u_pred : jnp.ndarray of shape (batch, n_points, 1)
    x : jnp.ndarray of shape (batch, n_points, coord_dim)
    freq_vectors : jnp.ndarray of shape (n_freqs, coord_dim)
    n_bins : int

    Returns
    -------
    dict with bin edges/centers and per-bin MSE/relative error/energy.
    """
    error_sq, true_energy = fourier_mse_by_frequency(u_pred, u_true, x, freq_vectors)

    # Average over batch first -> per-frequency values
    error_per_freq = np.array(jnp.mean(error_sq, axis=0))
    energy_per_freq = np.array(jnp.mean(true_energy, axis=0))

    freq_mag = np.array(jnp.linalg.norm(freq_vectors, axis=-1))
    freq_max = float(freq_mag.max()) if freq_mag.size else 1.0
    bin_edges = np.linspace(0.0, freq_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mse_per_bin = np.zeros(n_bins, dtype=np.float64)
    rel_error_per_bin = np.zeros(n_bins, dtype=np.float64)
    energy_per_bin = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (freq_mag >= bin_edges[i]) & (freq_mag < bin_edges[i + 1])
        if not np.any(mask):
            continue
        mse_per_bin[i] = float(np.mean(error_per_freq[mask]))
        energy_per_bin[i] = float(np.mean(energy_per_freq[mask]))
        if energy_per_bin[i] > 1e-12:
            rel_error_per_bin[i] = mse_per_bin[i] / energy_per_bin[i]

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "mse_per_bin": mse_per_bin,
        "rel_error_per_bin": rel_error_per_bin,
        "energy_per_bin": energy_per_bin,
    }


def compute_spectral_metrics_batch(
    u_true_batch: jnp.ndarray,
    u_pred_batch: jnp.ndarray,
    x_batch: jnp.ndarray,
    freq_vectors: jnp.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute off-grid spectral metrics on a batch."""
    return compute_frequency_binned_error(
        u_true=u_true_batch,
        u_pred=u_pred_batch,
        x=x_batch,
        freq_vectors=freq_vectors,
        n_bins=n_bins,
    )


def high_frequency_error_ratio(
    spectral_metrics: dict,
    cutoff_fraction: float = 0.5,
) -> float:
    """Ratio of high-frequency error to total binned error."""
    bin_centers = spectral_metrics["bin_centers"]
    mse_per_bin = spectral_metrics["mse_per_bin"]

    if len(bin_centers) == 0:
        return 0.0

    cutoff = cutoff_fraction * float(bin_centers.max())
    high_freq_mask = bin_centers >= cutoff
    low_freq_mask = bin_centers < cutoff

    high_freq_mse = float(np.sum(mse_per_bin[high_freq_mask]))
    low_freq_mse = float(np.sum(mse_per_bin[low_freq_mask]))
    total = high_freq_mse + low_freq_mse
    if total < 1e-12:
        return 0.0
    return high_freq_mse / total


class SpectralMetric(Metric):
    """Frequency-binned reconstruction diagnostics on sampled points."""

    def __init__(
        self,
        autoencoder,
        n_bins: int = 10,
        n_freqs: int = 128,
        fourier_sigma: float = 1.0,
        multiscale_sigmas: Optional[str | Sequence[float]] = None,
        seed: int = 0,
        name: str = "spectral",
    ):
        self.autoencoder = autoencoder
        self.n_bins = n_bins
        self.n_freqs = n_freqs
        self.fourier_sigma = fourier_sigma
        self.multiscale_sigmas = multiscale_sigmas
        self._name = name

        key = jax.random.PRNGKey(seed)
        self.freq_vectors = sample_frequency_vectors(
            key=key,
            n_freqs=n_freqs,
            coord_dim=2,
            sigma=fourier_sigma,
            multiscale_sigmas=multiscale_sigmas,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        metric_accum = None
        n_batches = 0

        for batch in test_dataloader:
            key, subkey = jax.random.split(key)
            batch_metrics = self.call_batched(state, batch, subkey)
            if metric_accum is None:
                metric_accum = {k: 0.0 for k in batch_metrics.keys()}
            for k, v in batch_metrics.items():
                metric_accum[k] += float(v)
            n_batches += 1

        if n_batches > 0:
            return {k: v / n_batches for k, v in metric_accum.items()}
        return {
            "high_freq_error_ratio": 0.0,
            "low_freq_mse": 0.0,
            "high_freq_mse": 0.0,
        }

    def call_batched(self, state, batch, subkey) -> dict:
        del subkey
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        vars_ = {"params": state.params}
        if state.batch_stats:
            vars_["batch_stats"] = state.batch_stats

        u_hat = self.autoencoder.apply(vars_, u_enc, x_enc, x_dec, train=False)

        metrics = compute_spectral_metrics_batch(
            u_true_batch=u_dec,
            u_pred_batch=u_hat,
            x_batch=x_dec,
            freq_vectors=self.freq_vectors,
            n_bins=self.n_bins,
        )
        hf_ratio = high_frequency_error_ratio(metrics, cutoff_fraction=0.5)

        split = max(self.n_bins // 2, 1)
        low = float(np.mean(metrics["mse_per_bin"][:split]))
        high = float(np.mean(metrics["mse_per_bin"][split:])) if split < self.n_bins else 0.0

        return {
            "high_freq_error_ratio": float(hf_ratio),
            "low_freq_mse": low,
            "high_freq_mse": high,
        }
