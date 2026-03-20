from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def evaluate_pointset_reconstruction(
    autoencoder,
    state,
    test_dataloader,
    n_batches: int = 10,
) -> dict:
    """Estimate reconstruction error on masked decoder point sets."""
    squared_error = 0.0
    squared_norm = 0.0
    count = 0

    for batch_index, batch in enumerate(test_dataloader):
        if batch_index >= n_batches:
            break

        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        latents = autoencoder.encode(state, u_enc, x_enc, train=False)
        reconstruction = autoencoder.decode(state, latents, x_dec, train=False)

        squared_error += float(jnp.sum((u_dec - reconstruction) ** 2))
        squared_norm += float(jnp.sum(u_dec ** 2))
        count += u_dec.shape[0] * u_dec.shape[1]

    mse = squared_error / max(count, 1)
    rel_mse = squared_error / max(squared_norm, 1e-10)
    return {"mse": mse, "rel_mse": rel_mse}


def visualize_pointset_reconstructions(
    autoencoder,
    state,
    test_dataloader,
    n_samples: int = 4,
    n_batches: int = 1,
) -> plt.Figure | None:
    """Visualize masked decoder-point reconstructions with shared color scales."""
    import matplotlib.tri as mtri

    samples = []
    for batch_index, batch in enumerate(test_dataloader):
        if batch_index >= n_batches:
            break

        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        latents = autoencoder.encode(state, u_enc, x_enc, train=False)
        reconstruction = autoencoder.decode(state, latents, x_dec, train=False)

        remaining = min(n_samples - len(samples), u_dec.shape[0])
        for sample_index in range(remaining):
            coords = np.array(x_dec[sample_index])
            if coords.ndim != 2 or coords.shape[-1] < 2:
                continue
            samples.append(
                {
                    "coords": coords[:, :2],
                    "original": np.array(u_dec[sample_index, :, 0]),
                    "reconstructed": np.array(reconstruction[sample_index, :, 0]),
                }
            )
            if len(samples) >= n_samples:
                break

        if len(samples) >= n_samples:
            break

    if not samples:
        return None

    figure, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 6))
    if len(samples) == 1:
        axes = axes[:, None]

    for column, sample in enumerate(samples):
        coords = sample["coords"]
        original = sample["original"]
        reconstructed = sample["reconstructed"]
        vmin = float(min(original.min(), reconstructed.min()))
        vmax = float(max(original.max(), reconstructed.max()))

        x = coords[:, 0]
        y = coords[:, 1]
        try:
            triangulation = mtri.Triangulation(x, y)
        except Exception:
            triangulation = None

        for row, values, title in (
            (0, original, f"Original {column + 1}"),
            (1, reconstructed, f"Reconstructed {column + 1}"),
        ):
            axis = axes[row, column]
            if triangulation is not None:
                axis.tripcolor(
                    triangulation,
                    values,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    shading="gouraud",
                )
            else:
                axis.scatter(x, y, c=values, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
            axis.set_title(title)
            axis.axis("off")
            axis.set_aspect("equal")

        relative_error = np.linalg.norm(original - reconstructed) / max(
            np.linalg.norm(original),
            1e-10,
        )
        axes[1, column].text(
            0.5,
            -0.05,
            f"Rel-Err: {relative_error:.3f}",
            transform=axes[1, column].transAxes,
            ha="center",
            fontsize=8,
        )

    figure.tight_layout()
    return figure
