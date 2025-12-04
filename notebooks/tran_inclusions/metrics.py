from typing import Any, Dict

import numpy as np

from mmsfm.data_utils import to_images, pca_decode

from .data_prep import decode_pseudo_microstates
from .interpolation import sample_latent_at_times
from .lifting import lift_pseudo_latents


def evaluate_interpolation_at_observed_times(
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    interpolation,
    models,
    lifting_metadata,
    config,
    components,
    mean_vec,
    explained_variance,
    is_whitened,
    whitening_epsilon,
    resolution,
) -> Dict[str, Any]:
    print("Evaluating interpolation at observed times...")
    _, phi_interp_true = sample_latent_at_times(interpolation, times_arr, method='frechet')
    embedding_mse = np.mean((tc_embeddings_time - phi_interp_true) ** 2, axis=(1, 2))
    print(f"Embedding MSE per time: {embedding_mse}")
    print(f"Mean Embedding MSE: {np.mean(embedding_mse)}")

    pseudo_micro = lift_pseudo_latents(
        phi_interp_true,
        times_arr,
        models,
        tc_embeddings_time,
        all_frames,
        times_arr,
        config,
        lifting_metadata,
        training_interpolation=interpolation,
    )

    decoded_imgs = decode_pseudo_microstates(
        pseudo_micro, components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution
    )

    X_true_flat = pca_decode(
        all_frames.reshape(-1, all_frames.shape[-1]),
        components,
        mean_vec,
        explained_variance,
        is_whitened,
        whitening_epsilon,
    )
    imgs_true = to_images(X_true_flat, resolution).reshape(len(times_arr), all_frames.shape[1], resolution, resolution)

    metrics = {}
    for method, imgs_pred in decoded_imgs.items():
        mse = np.mean((imgs_pred - imgs_true) ** 2, axis=(1, 2, 3))
        metrics[method] = {
            'per_time_mse': mse,
            'mean_mse': float(np.mean(mse)),
        }
    metrics['embedding_mse'] = embedding_mse
    return metrics
