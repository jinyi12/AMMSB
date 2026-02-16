"""Utility functions for loading GRF datasets as image-like tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


@dataclass
class GRFDataset:
    """Container for GRF data split across time marginals."""

    trainset: List[torch.Tensor]
    testset: List[torch.Tensor]
    classes: Tuple[str, ...]
    dims: Tuple[int, int, int]
    metadata: Dict[str, torch.Tensor | np.ndarray | float | bool]


def _reconstruct_fields(
    coeffs: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    explained_variance: np.ndarray,
    is_whitened: bool,
) -> np.ndarray:
    """Reconstruct spatial fields from stored PCA coefficients."""
    if is_whitened:
        sqrt_eigs = np.sqrt(np.maximum(explained_variance, 1e-12))
        scaled = coeffs * sqrt_eigs[np.newaxis, :]
        fields = scaled @ pca_components + pca_mean
    else:
        fields = coeffs @ pca_components + pca_mean
    return fields


def _reshape_to_images(fields: np.ndarray, resolution: int) -> np.ndarray:
    """Convert flattened fields to (N, 1, H, W) shape."""
    return fields.reshape(fields.shape[0], 1, resolution, resolution)


def _normalise_to_unit_interval(
    tensors: Sequence[torch.Tensor],
    *,
    use_min: torch.Tensor,
    use_max: torch.Tensor,
) -> List[torch.Tensor]:
    """Linearly map tensors to [-1, 1] range."""
    scale = torch.clamp(use_max - use_min, min=1e-6)
    normalised = [((t - use_min) / scale) * 2.0 - 1.0 for t in tensors]
    return normalised


def load_grf_dataset(
    npz_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    normalise: bool = True,
    prefer_raw: bool = True,
) -> GRFDataset:
    """Load GRF marginals saved by ``multimarginal_generation.py``.

    Parameters
    ----------
    npz_path:
        Path to the dataset generated via ``multimarginal_generation.py``.
    test_size:
        Fraction of samples reserved for evaluation while preserving pairing.
    seed:
        RNG seed for reproducible train/test splits.
    normalise:
        Whether to map pixel values to [-1, 1] using training statistics.
    prefer_raw:
        If both raw fields and PCA coefficients are present, load raw data.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"GRF dataset not found: {npz_path}")

    npz = np.load(npz_path)
    files = set(npz.files)
    data_dim = int(npz["data_dim"])
    resolution = int(np.sqrt(data_dim))

    dataset_format = "pca"
    if "dataset_format" in files:
        dataset_format = str(np.array(npz["dataset_format"]).item())

    raw_keys = sorted(
        [k for k in files if k.startswith("raw_marginal_")],
        key=lambda x: float(x.split("_")[-1]),
    )
    coeff_keys = sorted(
        [k for k in files if k.startswith("marginal_") and not k.startswith("raw_")],
        key=lambda x: float(x.split("_")[-1]),
    )

    data_source = None
    times = None

    if prefer_raw and raw_keys:
        arrays = [npz[k] for k in raw_keys]
        images = [_reshape_to_images(arr, resolution) for arr in arrays]
        data_source = "raw"
        times = np.array([float(k.split("_")[-1]) for k in raw_keys])
        is_whitened = False
    elif coeff_keys:
        if not {"pca_components", "pca_mean", "pca_explained_variance"}.issubset(files):
            raise ValueError("PCA metadata missing from dataset.")
        pca_components = npz["pca_components"]
        pca_mean = npz["pca_mean"]
        explained_variance = npz["pca_explained_variance"]
        is_whitened = bool(npz["is_whitened"]) if "is_whitened" in files else True
        coeffs = [npz[key] for key in coeff_keys]
        reconstructed = [
            _reconstruct_fields(c, pca_components, pca_mean, explained_variance, is_whitened)
            for c in coeffs
        ]
        images = [_reshape_to_images(r, resolution) for r in reconstructed]
        data_source = "pca"
        times = np.array([float(k.split("_")[-1]) for k in coeff_keys])
    elif raw_keys:
        arrays = [npz[k] for k in raw_keys]
        images = [_reshape_to_images(arr, resolution) for arr in arrays]
        data_source = "raw"
        times = np.array([float(k.split("_")[-1]) for k in raw_keys])
        is_whitened = False
    else:
        raise ValueError("No marginals found in GRF dataset.")

    n_samples = images[0].shape[0]
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )

    trainset = [torch.from_numpy(img[train_idx]).float() for img in images]
    testset = [torch.from_numpy(img[test_idx]).float() for img in images]

    if normalise:
        train_concat = torch.cat(trainset, dim=0)
        train_min = train_concat.amin(dim=0, keepdim=True)
        train_max = train_concat.amax(dim=0, keepdim=True)
        trainset = _normalise_to_unit_interval(trainset, use_min=train_min, use_max=train_max)
        testset = _normalise_to_unit_interval(testset, use_min=train_min, use_max=train_max)
    else:
        train_min = torch.zeros(1, 1, resolution, resolution)
        train_max = torch.ones(1, 1, resolution, resolution)

    classes = tuple(f"t={t:.2f}" for t in times)
    dims = trainset[0].shape[-3:]

    metadata: Dict[str, torch.Tensor | np.ndarray | float | bool] = {
        "times": times,
        "resolution": float(resolution),
        "is_whitened": is_whitened,
        "normalise": normalise,
        "train_min": train_min,
        "train_max": train_max,
        "data_source": data_source,
        "dataset_format": dataset_format,
        "pca_available": bool(coeff_keys),
        "raw_available": bool(raw_keys),
    }

    return GRFDataset(trainset=trainset, testset=testset, classes=classes, dims=dims, metadata=metadata)
