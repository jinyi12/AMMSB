"""Modular utilities for TCDM pipeline in `pca_precomputed_main.py`.

This module provides decoupled functions for each stage of the time-coupled
diffusion map (TCDM) pipeline:
1. Raw TCDM computation (epsilon selection + diffusion map)
2. Parsimonious dimension selection (LLR-based)
3. Lifter construction
4. Dense interpolation (Stiefel/Frechet)

Each step can run independently and save/load results via caching helpers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from tran_inclusions.interpolation import InterpolationResult


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pca_data(
    data_path,
    test_size=0.2,
    seed=42,
    *,
    return_indices: bool = False,
    return_full: bool = False,
    return_times: bool = False,
):
    """Load PCA coefficient data from an npz file and split into train/test.

    The same sample indices are used across all marginals to preserve pairing.
    """
    from sklearn.model_selection import train_test_split

    npz_data = np.load(data_path)
    held_out_indices = np.asarray(npz_data.get("held_out_indices", []), dtype=int)

    marginal_keys = sorted(
        [k for k in npz_data.keys() if k.startswith("marginal_")],
        key=lambda x: float(x.split("_")[1]),
    )
    marginal_times = [float(k.split("_")[1]) for k in marginal_keys]

    all_marginals = [npz_data[key] for key in marginal_keys]

    # Remove held-out time marginals to match tran_inclusions workflow.
    if held_out_indices.size > 0:
        keep_mask = np.ones(len(all_marginals), dtype=bool)
        keep_mask[np.clip(held_out_indices, 0, len(all_marginals) - 1)] = False
        all_marginals = [m for m, keep in zip(all_marginals, keep_mask) if keep]
        marginal_times = [t for t, keep in zip(marginal_times, keep_mask) if keep]

    n_samples = all_marginals[0].shape[0]
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, shuffle=True, random_state=seed
    )

    data = [marginal[train_idx] for marginal in all_marginals]
    testdata = [marginal[test_idx] for marginal in all_marginals]

    pca_info = {
        "components": npz_data["pca_components"],
        "mean": npz_data["pca_mean"],
        "explained_variance": npz_data["pca_explained_variance"],
        "data_dim": int(npz_data["data_dim"]),
    }

    if "is_whitened" in npz_data:
        pca_info["is_whitened"] = bool(npz_data["is_whitened"])
    else:
        print(
            "Warning: 'is_whitened' flag not found in dataset. Assuming coefficients are whitened."
        )
        pca_info["is_whitened"] = True

    outputs = [data, testdata, pca_info]
    if return_indices:
        outputs.append((train_idx, test_idx))
    if return_full:
        outputs.append(all_marginals)
    if return_times:
        outputs.append(np.array(marginal_times, dtype=float))
    return tuple(outputs)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _make_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    return value


def _normalize_meta(meta_dict: dict) -> dict:
    return {k: _make_jsonable(v) for k, v in meta_dict.items()}


def _meta_hash(meta_dict: dict) -> str:
    normalized = _normalize_meta(meta_dict)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _array_checksum(arr) -> Optional[str]:
    if arr is None:
        return None
    arr_np = np.asarray(arr)
    return hashlib.md5(arr_np.tobytes()).hexdigest()


def _meta_matches(
    cached_meta: dict, expected_meta: dict, *, allow_extra_meta: bool
) -> bool:
    normalized_cached = _normalize_meta(cached_meta)
    normalized_expected = _normalize_meta(expected_meta)
    if allow_extra_meta:
        for key, value in normalized_expected.items():
            if key not in normalized_cached or normalized_cached[key] != value:
                return False
        return True
    return _meta_hash(normalized_cached) == _meta_hash(normalized_expected)


def _load_cached_result(
    cache_path: Path,
    expected_meta: dict,
    label: str,
    *,
    refresh: bool = False,
    allow_extra_meta: bool = False,
):
    """Attempt to load a cached payload; return None on mismatch or failure."""
    if refresh:
        print(f"Skipping {label} cache because refresh was requested.")
        return None
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"Failed to load cached {label} from {cache_path}: {exc}")
        return None
    if not isinstance(payload, dict) or "meta" not in payload or "data" not in payload:
        print(f"Cache file at {cache_path} is malformed; recomputing {label}.")
        return None
    cached_meta = payload["meta"]
    if not _meta_matches(
        cached_meta, expected_meta, allow_extra_meta=allow_extra_meta
    ):
        print(f"{label.capitalize()} cache metadata mismatch; recomputing.")
        return None
    print(f"Loaded cached {label} from {cache_path}")
    return payload["data"]


def _save_cached_result(cache_path: Path, data, meta: dict, label: str):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_meta = _normalize_meta(meta)
    with cache_path.open("wb") as f:
        pickle.dump({"meta": normalized_meta, "data": data}, f)
    print(f"Saved {label} to cache: {cache_path}")


def _resolve_cache_base(cache_dir: Optional[str], data_path: str) -> Path:
    base = (
        Path(cache_dir)
        if cache_dir is not None
        else (REPO_ROOT / "data" / "cache_pca_precomputed")
    )
    base = base.expanduser().resolve()
    cache_base = base / Path(data_path).stem
    cache_base.mkdir(parents=True, exist_ok=True)
    return cache_base


def load_selected_embeddings(
    cache_path: Path | str,
    *,
    validate_checksums: bool = True,
    expected_train_checksum: Optional[str] = None,
    expected_test_checksum: Optional[str] = None,
) -> dict:
    """Load dimension-selected TCDM embeddings from a cache file.

    This function loads pre-computed embeddings (typically with dimension selection
    already applied, e.g. 70 dims instead of 128) from tc_selected_embeddings.pkl.

    Args:
        cache_path: Path to the tc_selected_embeddings.pkl file.
        validate_checksums: If True, validate train/test idx checksums if expected ones provided.
        expected_train_checksum: Optional checksum to validate train_idx.
        expected_test_checksum: Optional checksum to validate test_idx.

    Returns:
        Dictionary with keys compatible with tc_info:
        - latent_train: (T, N_train, latent_dim) array
        - latent_test: (T, N_test, latent_dim) array
        - latent_train_tensor: same as latent_train (for compatibility)
        - latent_test_tensor: same as latent_test (for compatibility)
        - selected_result: dict containing selected TCDM result
        - raw_result: TCDMRawResult if available
        - train_idx: array of training indices
        - test_idx: array of test indices
        - marginal_times: array of marginal time values
        - frames: (T, N, D) full PCA frames if available
        - meta: metadata dict from cache

    Raises:
        FileNotFoundError: If cache file does not exist.
        ValueError: If checksums do not match (when validation enabled).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Selected embeddings cache not found: {cache_path}")

    print(f"Loading dimension-selected embeddings from {cache_path}")
    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "meta" not in payload or "data" not in payload:
        raise ValueError(f"Cache file at {cache_path} is malformed (missing 'meta' or 'data').")

    meta = payload["meta"]
    data = payload["data"]

    # Validate checksums if requested
    if validate_checksums:
        if expected_train_checksum is not None:
            cached_train_checksum = meta.get("train_idx_checksum")
            if cached_train_checksum != expected_train_checksum:
                raise ValueError(
                    f"train_idx checksum mismatch: cache={cached_train_checksum}, "
                    f"expected={expected_train_checksum}"
                )
        if expected_test_checksum is not None:
            cached_test_checksum = meta.get("test_idx_checksum")
            if cached_test_checksum != expected_test_checksum:
                raise ValueError(
                    f"test_idx checksum mismatch: cache={cached_test_checksum}, "
                    f"expected={expected_test_checksum}"
                )

    # Extract and normalize data
    latent_train = np.asarray(data["latent_train"], dtype=np.float64)
    latent_test = np.asarray(data["latent_test"], dtype=np.float64)

    result = {
        "latent_train": latent_train,
        "latent_test": latent_test,
        "latent_train_tensor": latent_train,
        "latent_test_tensor": latent_test,
        "selected_result": data.get("selected_result"),
        "raw_result": data.get("raw_result"),
        "train_idx": np.asarray(data["train_idx"], dtype=np.int64) if "train_idx" in data else None,
        "test_idx": np.asarray(data["test_idx"], dtype=np.int64) if "test_idx" in data else None,
        "marginal_times": np.asarray(data["marginal_times"], dtype=np.float64) if "marginal_times" in data else None,
        "marginal_times_raw": np.asarray(data["marginal_times_raw"], dtype=np.float64) if "marginal_times_raw" in data else None,
        "frames": np.asarray(data["frames"], dtype=np.float32) if "frames" in data else None,
        "meta": meta,
    }

    latent_dim = int(latent_train.shape[2]) if latent_train.ndim == 3 else int(latent_train.shape[1])
    print(f"Loaded selected embeddings: latent_dim={latent_dim}, "
          f"train_shape={latent_train.shape}, test_shape={latent_test.shape}")

    return result


# ---------------------------------------------------------------------------
# Step 1: Raw TCDM computation
# ---------------------------------------------------------------------------


@dataclass
class TCDMRawResult:
    """Result from Step 1: raw TCDM embeddings (before dimension selection)."""

    embeddings_time: np.ndarray  # (T, N, k)
    singular_values: list[np.ndarray]
    stationaries: list[np.ndarray]
    left_singular_vectors: list[np.ndarray]
    right_singular_vectors: list[np.ndarray]
    transition_operators: list  # list of sparse transition matrices
    epsilons: np.ndarray
    kde_bandwidths: np.ndarray | None
    semigroup_df: object  # pandas DataFrame from semigroup selection
    times: np.ndarray
    tc_k: int


def compute_bandwidths_and_epsilons(
    frames: np.ndarray,
    times: np.ndarray,
    *,
    tc_alpha: float,
    tc_beta: float,
    epsilon_scales_min: float,
    epsilon_scales_max: float,
    epsilon_scales_num: int,
) -> tuple[np.ndarray, np.ndarray | None, object]:
    """Compute semigroup-optimal epsilons for TCDM.

    Returns:
        epsilons: array of per-time epsilon values
        kde_bandwidths: optional array of KDE bandwidths
        semigroup_df: DataFrame with semigroup error metrics
    """
    from diffmap.diffusion_maps import select_epsilons_by_semigroup
    from tran_inclusions.data_prep import compute_bandwidth_statistics

    bandwidth_stats = compute_bandwidth_statistics(frames)
    base_epsilons = bandwidth_stats["median"]
    epsilon_scales = np.geomspace(
        epsilon_scales_min, epsilon_scales_max, num=epsilon_scales_num
    )

    epsilons, kde_bandwidths, semigroup_df = select_epsilons_by_semigroup(
        frames,
        times=times,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        alpha=tc_alpha,
        sample_size=min(1024, frames.shape[1]),
        rng_seed=0,
        norm="operator",
        variable_bandwidth=False,
        beta=tc_beta,
        selection="first_local_minimum",
    )

    return epsilons, kde_bandwidths, semigroup_df


def compute_tcdm_raw(
    frames: np.ndarray,
    times: np.ndarray,
    *,
    tc_k: int,
    tc_alpha: float,
    tc_beta: float,
    epsilons: np.ndarray,
    kde_bandwidths: np.ndarray | None,
    power_iter_tol: float = 1e-12,
    power_iter_maxiter: int = 10_000,
) -> TCDMRawResult:
    """Compute raw time-coupled diffusion map embeddings (Step 1).

    This does NOT perform dimension selection.

    Args:
        frames: (T, N, D) array of data at each time
        times: (T,) array of time values
        tc_k: number of diffusion coordinates to retain
        tc_alpha: density normalization exponent
        tc_beta: drift parameter
        epsilons: per-time epsilon values
        kde_bandwidths: optional per-time KDE bandwidths
        power_iter_tol: tolerance for power iteration
        power_iter_maxiter: max iterations for power iteration

    Returns:
        TCDMRawResult with raw embeddings and metadata
    """
    from diffmap.diffusion_maps import (
        build_time_coupled_trajectory,
        time_coupled_diffusion_map,
    )

    tc_result = time_coupled_diffusion_map(
        list(frames),
        k=tc_k,
        alpha=tc_alpha,
        epsilons=epsilons,
        variable_bandwidth=False,
        beta=tc_beta,
        density_bandwidths=(
            kde_bandwidths.tolist() if kde_bandwidths is not None else None
        ),
        t=frames.shape[0],
        power_iter_tol=power_iter_tol,
        power_iter_maxiter=power_iter_maxiter,
    )

    traj_result = build_time_coupled_trajectory(
        tc_result.transition_operators,
        embed_dim=tc_k,
        power_iter_tol=power_iter_tol,
        power_iter_maxiter=power_iter_maxiter,
    )
    coords_time_major, stationaries, sigma_traj = traj_result

    return TCDMRawResult(
        embeddings_time=coords_time_major,
        singular_values=sigma_traj,
        stationaries=stationaries,
        left_singular_vectors=traj_result.left_singular_vectors,
        right_singular_vectors=traj_result.right_singular_vectors,
        transition_operators=tc_result.transition_operators,
        epsilons=epsilons,
        kde_bandwidths=kde_bandwidths,
        semigroup_df=traj_result if hasattr(traj_result, "semigroup_df") else None,
        times=times,
        tc_k=tc_k,
    )


# ---------------------------------------------------------------------------
# Step 2: Dimension selection (LLR-based parsimonious)
# ---------------------------------------------------------------------------


@dataclass
class TCDMSelectedResult:
    """Result from Step 2: TCDM embeddings with selected dimension."""

    embeddings_time: np.ndarray  # (T, N, selected_dim)
    singular_values: list[np.ndarray]
    left_singular_vectors: list[np.ndarray]
    right_singular_vectors: list[np.ndarray]
    selected_dim: int
    parsimonious_counts: list[int] | None
    raw_result: TCDMRawResult


def select_parsimonious_dimension(
    raw_result: TCDMRawResult,
    *,
    strategy: str = "max_parsimonious",
    selection: str = "auto",
    residual_threshold: Optional[float] = 0.1,
    min_coordinates: int = 1,
    max_eigenvectors: Optional[int] = 32,
    llr_neighbors: Optional[int] = 128,
    llr_max_searches: Optional[int] = 12,
    llr_interpolation: str = "log_pchip",
) -> TCDMSelectedResult:
    """Select parsimonious TCDM dimension via Local-Linear Regression (Step 2).

    Args:
        raw_result: output from compute_tcdm_raw
        strategy: 'fixed' (use tc_k) or 'max_parsimonious' (LLR selection)
        selection: LLR selection mode ('auto', 'kmeans', 'gap', 'threshold')
        residual_threshold: cutoff for 'threshold' selection
        min_coordinates: minimum coordinates to keep per marginal
        max_eigenvectors: cap on the number of eigenvectors tested by LLR
        llr_neighbors: number of nearest neighbors for local regression
        llr_max_searches: cap on semigroup searches for LLR
        llr_interpolation: interpolation for LLR ('log_linear', 'log_pchip')

    Returns:
        TCDMSelectedResult with potentially reduced dimension
    """
    from diffmap.diffusion_maps import select_non_harmonic_coordinates

    embeddings_time = raw_result.embeddings_time
    singular_values = raw_result.singular_values
    times = raw_result.times
    tc_k = raw_result.tc_k

    strategy = strategy.lower()
    if strategy not in {"fixed", "max_parsimonious"}:
        raise ValueError("strategy must be 'fixed' or 'max_parsimonious'.")

    if strategy == "fixed":
        return TCDMSelectedResult(
            embeddings_time=embeddings_time,
            singular_values=singular_values,
            left_singular_vectors=raw_result.left_singular_vectors,
            right_singular_vectors=raw_result.right_singular_vectors,
            selected_dim=tc_k,
            parsimonious_counts=None,
            raw_result=raw_result,
        )

    # max_parsimonious
    selection = selection.lower()
    if selection not in {"auto", "kmeans", "gap", "threshold"}:
        raise ValueError("selection must be 'auto', 'kmeans', 'gap', or 'threshold'.")
    if selection == "threshold" and residual_threshold is None:
        raise ValueError("residual_threshold is required for selection='threshold'.")

    max_searches = llr_max_searches
    if max_searches is not None and max_searches <= 0:
        max_searches = None

    counts: list[int] = []
    for idx in range(embeddings_time.shape[0]):
        _, mask, _ = select_non_harmonic_coordinates(
            singular_values[idx],
            embeddings_time[idx],
            residual_threshold=residual_threshold,
            min_coordinates=min_coordinates,
            llr_bandwidth_strategy="semigroup",
            llr_semigroup_selection="first_local_minimum",
            llr_semigroup_max_searches=max_searches,
            llr_semigroup_interpolation=llr_interpolation,
            selection=selection,
            max_eigenvectors=max_eigenvectors,
            llr_neighbors=llr_neighbors,
            coords_are_eigenvectors=False,
        )
        counts.append(int(mask.sum()))

    for t_val, count in zip(times, counts):
        print(f"Parsimonious directions at t={float(t_val):.3f}: {count}")

    selected_dim = int(max(counts)) if counts else tc_k
    selected_dim = min(selected_dim, tc_k)
    if selected_dim < 1:
        raise ValueError("Selected TCDM dimension must be positive.")

    print(f"Selected TCDM dimension (max across time): {selected_dim}")

    # Truncate arrays
    new_embeddings = embeddings_time[:, :, :selected_dim]
    new_singular_values = [vals[:selected_dim] for vals in singular_values]
    new_left_sv = [vec[:, :selected_dim] for vec in raw_result.left_singular_vectors]
    new_right_sv = [vec[:selected_dim, :] for vec in raw_result.right_singular_vectors]

    return TCDMSelectedResult(
        embeddings_time=new_embeddings,
        singular_values=new_singular_values,
        left_singular_vectors=new_left_sv,
        right_singular_vectors=new_right_sv,
        selected_dim=selected_dim,
        parsimonious_counts=counts,
        raw_result=raw_result,
    )



# ---------------------------------------------------------------------------
# Landmark utilities (global landmark selection + out-of-sample embedding)
# ---------------------------------------------------------------------------


def _compute_landmark_features(
    frames: np.ndarray,
    *,
    mode: str,
    projection_dim: int | None,
    rng_seed: int,
) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError("frames must be (T, N, D).")

    mode = mode.lower()
    if mode == "time_mean":
        features = frames.mean(axis=0)
    elif mode == "time_first":
        features = frames[0]
    elif mode == "time_last":
        features = frames[-1]
    elif mode == "time_concat":
        features = np.transpose(frames, (1, 0, 2)).reshape(frames.shape[1], -1)
    else:
        raise ValueError(
            "mode must be one of: time_mean, time_first, time_last, time_concat."
        )

    features = np.asarray(features, dtype=np.float64)
    if projection_dim is None:
        return features
    projection_dim = int(projection_dim)
    if projection_dim < 1:
        raise ValueError("projection_dim must be positive when provided.")
    if features.shape[1] <= projection_dim:
        return features

    rng = np.random.default_rng(int(rng_seed))
    proj = rng.normal(size=(features.shape[1], projection_dim)).astype(np.float64)
    proj /= np.sqrt(float(projection_dim))
    return features @ proj


def select_landmark_indices(
    frames: np.ndarray,
    *,
    n_landmarks: int,
    method: str = "kmedoids",
    feature_mode: str = "time_mean",
    projection_dim: int | None = 64,
    rng_seed: int = 0,
    kmeans_max_iter: int = 300,
    density_k: int = 32,
    density_pool_factor: int = 10,
    density_gamma: float = 0.5,
) -> np.ndarray:
    """Select a global set of landmark sample indices (shared across time).

    Landmarks are selected over the sample dimension N, and then reused for every
    time slice to preserve the time-coupled correspondence assumptions.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError("frames must be (T, N, D).")
    n_samples = int(frames.shape[1])
    if n_samples < 2:
        raise ValueError("Need at least two samples to select landmarks.")
    n_landmarks = int(n_landmarks)
    if n_landmarks < 1:
        raise ValueError("n_landmarks must be positive.")
    if n_landmarks > n_samples:
        raise ValueError(f"n_landmarks={n_landmarks} exceeds n_samples={n_samples}.")

    features = _compute_landmark_features(
        frames, mode=feature_mode, projection_dim=projection_dim, rng_seed=rng_seed
    )
    method = method.lower()

    if method in {"kmedoids", "k-medoids"}:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_landmarks,
            random_state=int(rng_seed),
            n_init="auto",
            max_iter=int(kmeans_max_iter),
        )
        labels = km.fit_predict(features)
        centers = km.cluster_centers_

        chosen: list[int] = []
        for cluster_id in range(n_landmarks):
            members = np.flatnonzero(labels == cluster_id)
            if members.size == 0:
                continue
            diffs = features[members] - centers[cluster_id]
            d2 = np.einsum("nd,nd->n", diffs, diffs)
            chosen.append(int(members[int(np.argmin(d2))]))

        chosen_unique = list(dict.fromkeys(chosen))
        if len(chosen_unique) < n_landmarks:
            rng = np.random.default_rng(int(rng_seed))
            remaining = np.setdiff1d(np.arange(n_samples, dtype=int), np.array(chosen_unique, dtype=int))
            fill = rng.choice(remaining, size=n_landmarks - len(chosen_unique), replace=False)
            chosen_unique.extend([int(x) for x in fill])

        return np.sort(np.asarray(chosen_unique[:n_landmarks], dtype=int))

    if method in {"density", "density_fps"}:
        from scipy.spatial import cKDTree

        density_k = int(density_k)
        if density_k < 1:
            raise ValueError("density_k must be positive.")
        k_query = min(density_k + 1, n_samples)
        tree = cKDTree(features)
        dists, idx = tree.query(features, k=k_query)
        if k_query == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        mean_knn = dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else dists[:, 0]
        density = 1.0 / np.maximum(mean_knn, 1e-12)

        pool_factor = int(density_pool_factor)
        if pool_factor < 1:
            raise ValueError("density_pool_factor must be positive.")
        pool_size = min(n_samples, max(n_landmarks, pool_factor * n_landmarks))
        pool = np.argsort(density)[::-1][:pool_size]
        X = features[pool]
        w = density[pool]
        gamma = float(density_gamma)
        if gamma < 0:
            raise ValueError("density_gamma must be non-negative.")
        first = int(pool[int(np.argmax(w))])
        selected = [first]

        min_d2 = np.full(pool_size, np.inf, dtype=np.float64)
        chosen_mask = np.zeros(pool_size, dtype=bool)
        chosen_mask[int(np.where(pool == first)[0][0])] = True

        x0 = X[int(np.where(pool == first)[0][0])]
        d2 = np.einsum("nd,nd->n", X - x0, X - x0)
        min_d2 = np.minimum(min_d2, d2)

        for _ in range(1, n_landmarks):
            score = min_d2 * np.power(w, gamma)
            score[chosen_mask] = -np.inf
            next_pos = int(np.argmax(score))
            chosen_mask[next_pos] = True
            selected.append(int(pool[next_pos]))
            x_new = X[next_pos]
            d2 = np.einsum("nd,nd->n", X - x_new, X - x_new)
            min_d2 = np.minimum(min_d2, d2)

        return np.sort(np.asarray(selected, dtype=int))

    raise ValueError("method must be 'kmedoids' or 'density'.")


def extend_timecoupled_embeddings_knn(
    frames_full: np.ndarray,
    *,
    landmark_idx: np.ndarray,
    embeddings_landmarks: np.ndarray,
    k_neighbors: int = 32,
    batch_size: int = 1024,
) -> np.ndarray:
    """Extend landmark embeddings to all samples via KNN kernel regression.

    Uses an adaptive bandwidth per query: h_i = (distance to k-th neighbor)^2.
    """
    from scipy.spatial import cKDTree

    frames_full = np.asarray(frames_full, dtype=np.float64)
    if frames_full.ndim != 3:
        raise ValueError("frames_full must be (T, N, D).")
    landmark_idx = np.asarray(landmark_idx, dtype=int).ravel()
    if landmark_idx.size == 0:
        raise ValueError("landmark_idx must be non-empty.")
    if np.any(landmark_idx < 0) or np.any(landmark_idx >= frames_full.shape[1]):
        raise ValueError("landmark_idx contains out-of-range indices.")

    emb_land = np.asarray(embeddings_landmarks, dtype=np.float64)
    if emb_land.ndim != 3:
        raise ValueError("embeddings_landmarks must be (T, L, K).")
    if emb_land.shape[0] != frames_full.shape[0]:
        raise ValueError("Time dimension mismatch between frames_full and embeddings_landmarks.")
    if emb_land.shape[1] != landmark_idx.size:
        raise ValueError("embeddings_landmarks second dim must match landmark_idx size.")

    T, N, _ = frames_full.shape
    K = emb_land.shape[2]
    L = int(landmark_idx.size)
    k_neighbors = int(k_neighbors)
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive.")
    k_query = min(k_neighbors, L)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    out = np.empty((T, N, K), dtype=np.float64)
    for t in range(T):
        X_land = frames_full[t, landmark_idx, :]
        tree = cKDTree(X_land)
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            queries = frames_full[t, start:stop, :]
            dists, idx = tree.query(queries, k=k_query)
            if k_query == 1:
                dists = dists[:, None]
                idx = idx[:, None]
            d2 = dists**2
            h = np.maximum(d2[:, -1], 1e-12)[:, None]
            weights = np.exp(-d2 / h)
            weights_sum = weights.sum(axis=1, keepdims=True)
            weights = weights / np.maximum(weights_sum, 1e-12)

            neigh = emb_land[t, idx, :]  # (B, k, K)
            out[t, start:stop, :] = np.einsum("bk,bkq->bq", weights, neigh)
    return out


def select_parsimonious_directions(
    raw_result: TCDMRawResult,
    *,
    selection: str = "auto",
    residual_threshold: Optional[float] = 0.1,
    min_coordinates: int = 1,
    max_eigenvectors: Optional[int] = None,
    llr_neighbors: Optional[int] = 128,
    llr_max_searches: Optional[int] = 12,
    llr_interpolation: str = "log_pchip",
    combine: str = "union",
) -> dict:
    """Select a fixed set of coordinate indices across time via LLR masks."""
    from diffmap.diffusion_maps import select_non_harmonic_coordinates

    combine = combine.lower()
    if combine not in {"union", "intersection"}:
        raise ValueError("combine must be 'union' or 'intersection'.")

    masks: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    counts: list[int] = []
    for idx in range(raw_result.embeddings_time.shape[0]):
        _, mask, resid = select_non_harmonic_coordinates(
            raw_result.singular_values[idx],
            raw_result.embeddings_time[idx],
            residual_threshold=residual_threshold,
            min_coordinates=min_coordinates,
            llr_bandwidth_strategy="semigroup",
            llr_semigroup_selection="first_local_minimum",
            llr_semigroup_max_searches=llr_max_searches,
            llr_semigroup_interpolation=llr_interpolation,
            selection=selection,
            max_eigenvectors=max_eigenvectors,
            llr_neighbors=llr_neighbors,
            coords_are_eigenvectors=False,
        )
        masks.append(mask)
        residuals.append(resid)
        counts.append(int(mask.sum()))

    combined = masks[0].copy()
    for m in masks[1:]:
        combined = (combined | m) if combine == "union" else (combined & m)
    selected_indices = np.flatnonzero(combined)
    if selected_indices.size == 0:
        selected_indices = np.array([0], dtype=int)

    return {
        "selected_indices": selected_indices.astype(int),
        "masks": masks,
        "residuals": residuals,
        "counts": counts,
    }


def apply_selected_indices(
    raw_result: TCDMRawResult,
    selected_indices: np.ndarray,
) -> TCDMSelectedResult:
    selected_indices = np.asarray(selected_indices, dtype=int).ravel()
    if selected_indices.size == 0:
        raise ValueError("selected_indices must be non-empty.")

    emb = raw_result.embeddings_time[:, :, selected_indices]
    sv = [np.asarray(vals)[selected_indices] for vals in raw_result.singular_values]
    lsv = [np.asarray(vec)[:, selected_indices] for vec in raw_result.left_singular_vectors]
    rsv = [np.asarray(vec)[selected_indices, :] for vec in raw_result.right_singular_vectors]

    return TCDMSelectedResult(
        embeddings_time=emb,
        singular_values=sv,
        left_singular_vectors=lsv,
        right_singular_vectors=rsv,
        selected_dim=int(selected_indices.size),
        parsimonious_counts=None,
        raw_result=raw_result,
    )




# ---------------------------------------------------------------------------
# Step 4: Interpolation (dense trajectories)
# ---------------------------------------------------------------------------


@dataclass
class InterpolationBundle:
    """Bundle of interpolation results for sampler construction."""

    dense_trajs: np.ndarray  # (T_dense, N, latent_dim)
    t_dense: np.ndarray  # (T_dense,)
    frechet_mode: str
    interp_result: "InterpolationResult"


def compute_interpolation(
    selected_result: TCDMSelectedResult | dict,
    latent_train: np.ndarray,
    times_train: np.ndarray,
    *,
    n_dense: int = 200,
    frechet_mode: str = "triplet",
    raw_result: TCDMRawResult | dict | None = None,
) -> InterpolationBundle:
    """Compute dense latent trajectories via Stiefel/Frechet interpolation (Step 4).

    Args:
        selected_result: output from select_parsimonious_dimension (object or dict)
        latent_train: (T, N_train, latent_dim) training embeddings
        times_train: (T,) array of training time values
        n_dense: number of dense interpolation steps
        frechet_mode: 'global' or 'triplet'
        raw_result: optional raw TCDM result for stationaries/A_operators reconstruction

    Returns:
        InterpolationBundle with dense trajectories
    """
    from tran_inclusions.interpolation import build_dense_latent_trajectories

    # Reconstruct traj_result-like object for build_dense_latent_trajectories
    # It needs .left_singular_vectors, .singular_values, .stationary_distributions, .A_operators
    @dataclass
    class _TrajResultProxy:
        left_singular_vectors: list
        right_singular_vectors: list
        singular_values: list
        stationary_distributions: list
        A_operators: list

    # Handle both object and dict
    if isinstance(selected_result, dict):
        lsv = selected_result['left_singular_vectors']
        rsv = selected_result.get('right_singular_vectors')
        sv = selected_result['singular_values']
        stationaries = selected_result.get('stationary_distributions') or selected_result.get('stationaries')
        a_ops = selected_result.get('A_operators')
    else:
        lsv = selected_result.left_singular_vectors
        rsv = selected_result.right_singular_vectors
        sv = selected_result.singular_values
        stationaries = (
            getattr(selected_result, 'stationary_distributions', None)
            or getattr(selected_result, 'stationaries', None)
        )
        a_ops = getattr(selected_result, 'A_operators', None)
        if raw_result is None:
            raw_result = getattr(selected_result, 'raw_result', None)

    if stationaries is None and raw_result is not None:
        if isinstance(raw_result, dict):
            stationaries = raw_result.get('stationaries') or raw_result.get('stationary_distributions')
        else:
            stationaries = (
                getattr(raw_result, 'stationaries', None)
                or getattr(raw_result, 'stationary_distributions', None)
            )
    if a_ops is None and raw_result is not None:
        if isinstance(raw_result, dict):
            a_ops = raw_result.get('A_operators')
        else:
            a_ops = getattr(raw_result, 'A_operators', None)

    if a_ops is None:
        if stationaries is None:
            raise ValueError(
                'Interpolation requires stationary distributions to reconstruct A_operators. '
                'Pass raw_result from Step 1/2 or include stationaries in selected_result.'
            )
        if rsv is None:
            raise ValueError(
                'Interpolation requires right singular vectors to reconstruct A_operators.'
            )

        def _reconstruct_a_operators(
            left_vecs: list[np.ndarray],
            right_vecs: list[np.ndarray],
            sigmas: list[np.ndarray],
            pis: list[np.ndarray],
        ) -> list[np.ndarray]:
            if len(left_vecs) != len(right_vecs) or len(left_vecs) != len(sigmas):
                raise ValueError('Singular vector lists must have matching lengths.')
            if len(left_vecs) != len(pis):
                raise ValueError('Stationary distributions must align with time slices.')
            a_list: list[np.ndarray] = []
            for u, vt, s, pi in zip(left_vecs, right_vecs, sigmas, pis):
                u = np.asarray(u)
                vt = np.asarray(vt)
                s = np.asarray(s)
                pi = np.asarray(pi)
                sqrt_pi = np.sqrt(pi)
                base = np.outer(sqrt_pi, sqrt_pi)
                if s.size == 0:
                    a_list.append(base)
                    continue
                a_list.append(base + (u * s[None, :]) @ vt)
            return a_list

        a_ops = _reconstruct_a_operators(lsv, rsv, sv, stationaries)

    if stationaries is None:
        raise ValueError('Interpolation requires stationary distributions for reconstruction.')

    traj_proxy = _TrajResultProxy(
        left_singular_vectors=lsv,
        right_singular_vectors=rsv,
        singular_values=sv,
        stationary_distributions=stationaries,
        A_operators=a_ops,
    )

    interp_result = build_dense_latent_trajectories(
        traj_proxy,
        times_train=times_train,
        tc_embeddings_time=latent_train,
        n_dense=n_dense,
        frechet_mode=frechet_mode,
        compute_global=(frechet_mode == "global"),
        compute_triplet=(frechet_mode == "triplet"),
        compute_naive=False,
    )

    if frechet_mode == "triplet":
        dense_trajs = interp_result.phi_frechet_triplet_dense
    else:
        dense_trajs = interp_result.phi_frechet_global_dense

    if dense_trajs is None:
        print("Warning: requested mode returned None, falling back to Frechet global.")
        dense_trajs = interp_result.phi_frechet_global_dense

    return InterpolationBundle(
        dense_trajs=dense_trajs,
        t_dense=interp_result.t_dense,
        frechet_mode=frechet_mode,
        interp_result=interp_result,
    )


# ---------------------------------------------------------------------------
# High-level orchestrator (combines all steps)
# ---------------------------------------------------------------------------


@dataclass
class TCDMPipelineResult:
    """Complete result from the TCDM pipeline."""

    latent_train: list[np.ndarray]  # per-time list
    latent_test: list[np.ndarray]  # per-time list
    latent_train_tensor: np.ndarray  # (T, N_train, dim)
    latent_test_tensor: np.ndarray  # (T, N_test, dim)
    epsilons: np.ndarray
    semigroup_df: object
    marginal_times: np.ndarray
    stationaries: list[np.ndarray]
    sigma_traj: list[np.ndarray]
    selected_dim: int
    parsimonious_counts: list[int] | None
    raw_result: TCDMRawResult
    selected_result: TCDMSelectedResult


def prepare_timecoupled_latents(
    full_marginals: list[np.ndarray],
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    zt_rem_idxs: np.ndarray,
    times_raw: np.ndarray,
    tc_k: int,
    tc_alpha: float,
    tc_beta: float,
    tc_epsilon_scales_min: float,
    tc_epsilon_scales_max: float,
    tc_epsilon_scales_num: int,
    tc_power_iter_tol: float,
    tc_power_iter_maxiter: int,
    tc_dim_strategy: str = "fixed",
    tc_dim_selection: str = "auto",
    tc_dim_residual_threshold: Optional[float] = 0.1,
    tc_dim_min_coordinates: int = 1,
    tc_dim_max_eigenvectors: Optional[int] = 32,
    tc_dim_llr_neighbors: Optional[int] = 128,
    tc_dim_llr_max_searches: Optional[int] = 12,
    tc_dim_llr_interpolation: str = "log_pchip",
) -> TCDMPipelineResult:
    """Orchestrate full TCDM pipeline: epsilon selection, TCDM, dim selection, lifter.

    This is the main entry point that combines all steps.
    For modular execution, call the individual step functions directly.
    """
    if not full_marginals:
        raise ValueError("full_marginals list is empty.")

    selected = [full_marginals[i] for i in zt_rem_idxs]
    frames = np.stack(selected, axis=0)  # (T, N_all, D)
    times_arr = np.asarray(times_raw, dtype=float)

    # Step 1a: Epsilon selection
    epsilons, kde_bandwidths, semigroup_df = compute_bandwidths_and_epsilons(
        frames,
        times_arr,
        tc_alpha=tc_alpha,
        tc_beta=tc_beta,
        epsilon_scales_min=tc_epsilon_scales_min,
        epsilon_scales_max=tc_epsilon_scales_max,
        epsilon_scales_num=tc_epsilon_scales_num,
    )

    # Step 1b: Raw TCDM
    raw_result = compute_tcdm_raw(
        frames,
        times_arr,
        tc_k=tc_k,
        tc_alpha=tc_alpha,
        tc_beta=tc_beta,
        epsilons=epsilons,
        kde_bandwidths=kde_bandwidths,
        power_iter_tol=tc_power_iter_tol,
        power_iter_maxiter=tc_power_iter_maxiter,
    )

    # Step 2: Dimension selection
    selected_result = select_parsimonious_dimension(
        raw_result,
        strategy=tc_dim_strategy,
        selection=tc_dim_selection,
        residual_threshold=tc_dim_residual_threshold,
        min_coordinates=tc_dim_min_coordinates,
        max_eigenvectors=tc_dim_max_eigenvectors,
        llr_neighbors=tc_dim_llr_neighbors,
        llr_max_searches=tc_dim_llr_max_searches,
        llr_interpolation=tc_dim_llr_interpolation,
    )

    coords_time_major = selected_result.embeddings_time
    embed_dim = selected_result.selected_dim

    # Prepare outputs
    latent_train = coords_time_major[:, train_idx, :]
    latent_test = coords_time_major[:, test_idx, :]

    latent_train_list = [latent_train[t] for t in range(latent_train.shape[0])]
    latent_test_list = [latent_test[t] for t in range(latent_test.shape[0])]

    return TCDMPipelineResult(
        latent_train=latent_train_list,
        latent_test=latent_test_list,
        latent_train_tensor=latent_train,
        latent_test_tensor=latent_test,
        epsilons=epsilons,
        semigroup_df=semigroup_df,
        marginal_times=times_arr,
        stationaries=raw_result.stationaries,
        sigma_traj=selected_result.singular_values,
        selected_dim=embed_dim,
        parsimonious_counts=selected_result.parsimonious_counts,
        raw_result=raw_result,
        selected_result=selected_result,
    )


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def build_lift_times(zt: np.ndarray) -> np.ndarray:
    """Training knots plus midpoints between consecutive knots."""
    lift_times = []
    for idx, t in enumerate(zt):
        lift_times.append(float(t))
        if idx < len(zt) - 1:
            lift_times.append(float(0.5 * (t + zt[idx + 1])))
    return np.array(lift_times, dtype=float)
