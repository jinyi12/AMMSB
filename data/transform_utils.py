"""
Utility functions for data transformations in multimarginal generation.

Provides forward and inverse transforms for:
- Min-max scaling
- Log standardization (recommended for strictly positive fields in generative modeling)
- Logit standardization with affine bounds (recommended for bounded fields, e.g. two-phase inclusions)
- Affine standardization with known bounds (linear alternative for bounded fields)

IMPORTANT: Log standardization uses GLOBAL standardization (single scalar mean/std
across all pixels) to preserve geometric structure. Per-pixel standardization would
destroy spatial relationships by forcing each location to have identical statistics.
"""

import numpy as np
from typing import Dict, Any


def minmax_transform(
    data: np.ndarray,
    data_min: float | np.ndarray,
    data_scale: float | np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Forward global min-max scaling."""
    if isinstance(data_min, np.ndarray) and data_min.size == 1:
        data_min = float(data_min.item())
    if isinstance(data_scale, np.ndarray) and data_scale.size == 1:
        data_scale = float(data_scale.item())
    if float(data_scale) <= 0.0:
        raise ValueError(f"minmax_transform requires data_scale > 0, got {data_scale}.")
    return np.asarray((data - data_min) / data_scale + epsilon, dtype=np.float32)


def inverse_minmax_transform(
    scaled_data: np.ndarray,
    data_min: float | np.ndarray,
    data_scale: float | np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Inverse global min-max scaling.

    Following Bunker et al. (2025), min-max uses **global** (scalar) min/scale,
    not per-pixel arrays.  Legacy datasets may still store per-feature arrays;
    scalar extraction is handled here for backwards compatibility.

    Args:
        scaled_data: Scaled data in approximately [eps, 1+eps]
        data_min: Global scalar minimum from training
        data_scale: Global scalar scale (max - min) from training
        epsilon: Epsilon used in forward transform

    Returns:
        Original scale data
    """
    if isinstance(data_min, np.ndarray) and data_min.size == 1:
        data_min = float(data_min.item())
    if isinstance(data_scale, np.ndarray) and data_scale.size == 1:
        data_scale = float(data_scale.item())

    scaled_data = np.clip(scaled_data, epsilon, 1.0 + epsilon)
    return (scaled_data - epsilon) * data_scale + data_min


def affine_standardize_transform(
    data: np.ndarray,
    affine_mean: float | np.ndarray,
    affine_std: float | np.ndarray,
    affine_min: float,
    affine_max: float,
    affine_delta: float = 0.0,
) -> np.ndarray:
    """Forward affine-standardization transform for bounded data."""
    if isinstance(affine_mean, np.ndarray) and affine_mean.size == 1:
        affine_mean = float(affine_mean.item())
    if isinstance(affine_std, np.ndarray) and affine_std.size == 1:
        affine_std = float(affine_std.item())
    if affine_max <= affine_min:
        raise ValueError(
            f"affine_standardize_transform requires affine_max > affine_min. "
            f"Got affine_min={affine_min}, affine_max={affine_max}."
        )
    if float(affine_std) <= 0.0:
        raise ValueError(f"affine_standardize_transform requires affine_std > 0, got {affine_std}.")
    if affine_delta < 0.0 or affine_delta >= 0.5:
        raise ValueError(
            f"affine_standardize_transform requires 0 <= affine_delta < 0.5. "
            f"Got affine_delta={affine_delta}."
        )
    scale = float(affine_max - affine_min)
    p = (data - affine_min) / scale
    if affine_delta > 0.0:
        p = np.clip(p, affine_delta, 1.0 - affine_delta)
    else:
        p = np.clip(p, 0.0, 1.0)
    return np.asarray((p - affine_mean) / affine_std, dtype=np.float32)


def inverse_affine_standardize_transform(
    standardized_data: np.ndarray,
    affine_mean: float | np.ndarray,
    affine_std: float | np.ndarray,
    affine_min: float,
    affine_max: float,
    affine_delta: float = 0.0,
) -> np.ndarray:
    """Inverse affine→standardisation transform for bounded data.

    Forward:
      p = (x - affine_min)/(affine_max-affine_min)
      p = clip(p, delta, 1-delta)
      z = (p - affine_mean) / affine_std

    Inverse:
      p = z * affine_std + affine_mean
      p = clip(p, delta, 1-delta)
      x = affine_min + p * (affine_max - affine_min)

    This is a *linear* alternative to logit-standardisation that preserves the
    additivity of filtered indicator fields (Tran convolution) while still
    keeping outputs bounded after inversion via clipping.
    """
    if isinstance(affine_mean, np.ndarray) and affine_mean.size == 1:
        affine_mean = float(affine_mean.item())
    if isinstance(affine_std, np.ndarray) and affine_std.size == 1:
        affine_std = float(affine_std.item())

    if affine_max <= affine_min:
        raise ValueError(
            f"inverse_affine_standardize_transform requires affine_max > affine_min. "
            f"Got affine_min={affine_min}, affine_max={affine_max}."
        )
    if affine_delta < 0.0 or affine_delta >= 0.5:
        raise ValueError(
            f"inverse_affine_standardize_transform requires 0 <= affine_delta < 0.5. "
            f"Got affine_delta={affine_delta}."
        )

    p = standardized_data * affine_std + affine_mean
    if affine_delta > 0.0:
        p = np.clip(p, affine_delta, 1.0 - affine_delta)
    else:
        p = np.clip(p, 0.0, 1.0)
    return affine_min + p * (affine_max - affine_min)


def log_standardize_transform(
    data: np.ndarray,
    log_mean: float | np.ndarray,
    log_std: float | np.ndarray,
    log_epsilon: float,
) -> np.ndarray:
    """Forward log-standardization transform."""
    if isinstance(log_mean, np.ndarray) and log_mean.size == 1:
        log_mean = float(log_mean.item())
    if isinstance(log_std, np.ndarray) and log_std.size == 1:
        log_std = float(log_std.item())
    if np.min(data) + float(log_epsilon) <= 0.0:
        raise ValueError(
            "log_standardize_transform requires strictly positive shifted data. "
            f"Found min(data + log_epsilon)={float(np.min(data) + float(log_epsilon))}."
        )
    if float(log_std) <= 0.0:
        raise ValueError(f"log_standardize_transform requires log_std > 0, got {log_std}.")
    log_data = np.log(data + log_epsilon)
    return np.asarray((log_data - log_mean) / log_std, dtype=np.float32)


def inverse_log_standardize_transform(
    standardized_data: np.ndarray,
    log_mean: float | np.ndarray,
    log_std: float | np.ndarray,
    log_epsilon: float,
) -> np.ndarray:
    """Inverse log-standardization transform.

    This reverses the log+standardization pipeline:
    1. Unstandardize: multiply by std, add mean
    2. Exp transform: map back to positive domain

    Guarantees strict positivity of output.

    IMPORTANT: Uses GLOBAL standardization (scalar log_mean, log_std) to preserve
    geometric structure. Per-pixel standardization would destroy spatial relationships.

    Args:
        standardized_data: Standardized log-space data (mean≈0, std≈1)
        log_mean: Global (scalar) mean of log-transformed training data.
                  Legacy datasets may have per-feature arrays.
        log_std: Global (scalar) std of log-transformed training data.
                 Legacy datasets may have per-feature arrays.
        log_epsilon: Epsilon added before log in forward transform

    Returns:
        Original scale strictly positive data
    """
    # Handle both scalar (new) and array (legacy) formats
    if isinstance(log_mean, np.ndarray):
        if log_mean.size == 1:
            log_mean = float(log_mean.item())
    if isinstance(log_std, np.ndarray):
        if log_std.size == 1:
            log_std = float(log_std.item())

    # Step 1: Unstandardize (global mean/std preserves geometry)
    log_data = standardized_data * log_std + log_mean

    # Step 2: Inverse log transform
    positive_data = np.exp(log_data) - log_epsilon

    return positive_data


def load_transform_info(npz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Load transformation metadata from dataset npz file.

    Args:
        npz_data: Loaded npz file (use np.load(...))

    Returns:
        Dictionary with transform type and parameters
    """
    scale_mode = str(npz_data.get('scale_mode', 'none'))

    if scale_mode == 'minmax':
        # Global scalar min/scale (Bunker et al. convention).
        # Legacy datasets may store per-pixel arrays; extract scalar.
        _min = npz_data['minmax_data_min']
        _scale = npz_data['minmax_data_scale']
        return {
            'type': 'minmax',
            'data_min': float(_min.item()) if hasattr(_min, 'item') else float(_min),
            'data_scale': float(_scale.item()) if hasattr(_scale, 'item') else float(_scale),
            'epsilon': float(npz_data.get('scaling_epsilon', 0.0)),
        }
    elif scale_mode == 'affine_standardize':
        return {
            'type': 'affine_standardize',
            'affine_mean': npz_data['affine_mean'],
            'affine_std': npz_data['affine_std'],
            'affine_delta': float(npz_data.get('affine_delta', 0.0)),
            'affine_min': float(npz_data['affine_min']),
            'affine_max': float(npz_data['affine_max']),
        }
    elif scale_mode == 'log_standardize':
        return {
            'type': 'log_standardize',
            'log_mean': npz_data['log_mean'],
            'log_std': npz_data['log_std'],
            'log_epsilon': float(npz_data['log_epsilon']),
        }
    else:
        return {'type': 'none'}


def apply_inverse_transform(
    data: np.ndarray,
    transform_info: Dict[str, Any],
) -> np.ndarray:
    """Apply inverse transform based on transform_info.

    Args:
        data: Transformed data
        transform_info: Dictionary from load_transform_info()

    Returns:
        Data in original scale
    """
    transform_type = transform_info['type']

    if transform_type == 'minmax':
        return inverse_minmax_transform(
            data,
            transform_info['data_min'],
            transform_info['data_scale'],
            transform_info['epsilon'],
        )
    elif transform_type == 'affine_standardize':
        return inverse_affine_standardize_transform(
            data,
            transform_info['affine_mean'],
            transform_info['affine_std'],
            transform_info['affine_min'],
            transform_info['affine_max'],
            transform_info.get('affine_delta', 0.0),
        )
    elif transform_type == 'log_standardize':
        return inverse_log_standardize_transform(
            data,
            transform_info['log_mean'],
            transform_info['log_std'],
            transform_info['log_epsilon'],
        )
    else:
        # No transform applied
        return data


def apply_forward_transform(
    data: np.ndarray,
    transform_info: Dict[str, Any],
) -> np.ndarray:
    """Apply the dataset forward transform based on transform_info."""
    transform_type = transform_info["type"]

    if transform_type == "minmax":
        return minmax_transform(
            data,
            transform_info["data_min"],
            transform_info["data_scale"],
            transform_info["epsilon"],
        )
    if transform_type == "affine_standardize":
        return affine_standardize_transform(
            data,
            transform_info["affine_mean"],
            transform_info["affine_std"],
            transform_info["affine_min"],
            transform_info["affine_max"],
            transform_info.get("affine_delta", 0.0),
        )
    if transform_type == "log_standardize":
        return log_standardize_transform(
            data,
            transform_info["log_mean"],
            transform_info["log_std"],
            transform_info["log_epsilon"],
        )
    return np.asarray(data, dtype=np.float32)


# Example usage
if __name__ == "__main__":
    print("Transform utilities for multimarginal generation")
    print("\nUsage example:")
    print("""
    import numpy as np
    from data.transform_utils import load_transform_info, apply_inverse_transform

    # Load dataset
    data = np.load('data/tran_inclusions.npz')

    # Load transform metadata
    transform_info = load_transform_info(data)
    print(f"Transform type: {transform_info['type']}")

    # Load transformed marginals
    marginal_t0 = data['marginal_0.0']  # PCA coefficients, already transformed

    # After PCA reconstruction to ambient space (still transformed)
    # Apply inverse transform to get back to physical field values
    physical_fields = apply_inverse_transform(reconstructed_ambient, transform_info)
    """)
