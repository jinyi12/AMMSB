from typing import Any, Dict, Optional, Tuple

import numpy as np

from diffmap.continuous_gh import ContinuousGeometricHarmonics
from diffmap.lifting import (
    ConvexHullInterpolator,
    TimeCoupledTrajectoryResult,
    build_training_pairs,
    build_time_augmented_training_pairs,
)

from .config import LiftingConfig, LiftingModels, LatentInterpolationResult


def fit_lifting_models(
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    config: LiftingConfig,
    *,
    trajectory: Optional[TimeCoupledTrajectoryResult] = None,
) -> Tuple[LiftingModels, Dict[str, Any]]:
    """Build training data and fit convex + continuous GH lifting models."""
    _ = trajectory  # retained for backward compatibility; continuous GH does not use it.
    macro_train, micro_train = build_training_pairs(
        tc_embeddings_time,
        all_frames,
        times_arr,
        config.holdout_time,
        tol=config.time_match_tol,
    )
    macro_train_coords, train_time_values, micro_train_states = build_time_augmented_training_pairs(
        macro_time_series=tc_embeddings_time,
        micro_time_series=all_frames,
        snapshot_times=times_arr,
        holdout_snapshot_time=config.holdout_time,
        tol=config.time_match_tol,
    )
    print(f"Training pairs collected: macro {macro_train.shape}, micro {micro_train.shape}")
    time_span = float(times_arr.max() - times_arr.min()) if times_arr.size > 0 else 1.0
    if config.preimage_time_window is None:
        if len(times_arr) >= 2:
            config.preimage_time_window = float(np.min(np.diff(np.sort(times_arr)))) * 0.6
        else:
            config.preimage_time_window = time_span

    chi = ConvexHullInterpolator(macro_states=macro_train, micro_states=micro_train)

    cgh_model: Optional[ContinuousGeometricHarmonics] = None
    if config.use_continuous_gh:
        cgh_model = ContinuousGeometricHarmonics(
            max_modes=config.gh_max_modes,
            eigenvalue_delta=config.gh_delta,
            energy_threshold=config.gh_energy_threshold,
            ridge=config.gh_ridge,
            semigroup_norm=config.gh_semigroup_norm,
            epsilon_grid_size=config.gh_epsilon_grid_size,
            epsilon_log_span=config.gh_epsilon_log_span,
            semigroup_selection=config.gh_semigroup_selection,
        )
        cgh_model.fit(
            times=times_arr,
            latent_points=tc_embeddings_time,
            ambient_points=all_frames,
            bandwidth_candidates=config.gh_bandwidth_candidates,
        )

    models = LiftingModels(convex=chi, continuous_gh=cgh_model)
    metadata = {
        "macro_train": macro_train,
        "micro_train": micro_train,
        "macro_train_coords": macro_train_coords,
        "train_time_values": train_time_values,
        "micro_train_states": micro_train_states,
    }
    return models, metadata


def lift_pseudo_latents(
    phi_pseudo: np.ndarray,
    t_pseudo: np.ndarray,
    models: LiftingModels,
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    config: LiftingConfig,
    lifting_metadata: Dict[str, Any],
    *,
    training_interpolation: Optional[LatentInterpolationResult] = None,
) -> Dict[str, np.ndarray]:
    """Lift latent pseudo-trajectories using convex hull and continuous GH."""
    _ = tc_embeddings_time  # unused in the current lifting routines
    n_pseudo, n_samples, _ = phi_pseudo.shape
    n_components = all_frames.shape[2]

    macro_train_coords = lifting_metadata.get("macro_train_coords")
    train_time_values = lifting_metadata.get("train_time_values")
    micro_train_states = lifting_metadata.get("micro_train_states")

    def _get_time_local_indices(target_time: float) -> Optional[np.ndarray]:
        if train_time_values is None:
            return None
        tt = np.asarray(train_time_values, dtype=np.float64)
        times_grid = np.asarray(times_arr, dtype=np.float64)
        if config.preimage_time_window is not None:
            window = float(config.preimage_time_window)
            mask = np.abs(tt - target_time) <= (window + config.time_match_tol)
        else:
            if times_grid.size == 0:
                return None
            idx = np.searchsorted(times_grid, target_time)
            if idx == 0:
                lo, hi = times_grid[0], times_grid[min(1, times_grid.size - 1)]
            elif idx >= times_grid.size:
                lo, hi = times_grid[max(times_grid.size - 2, 0)], times_grid[-1]
            else:
                lo, hi = times_grid[idx - 1], times_grid[idx]
            mask = (tt >= lo - config.time_match_tol) & (tt <= hi + config.time_match_tol)
        if not np.any(mask):
            dists = np.abs(tt - target_time)
            min_dist = np.min(dists)
            mask = dists <= (min_dist + config.time_match_tol)
        return np.nonzero(mask)[0]

    X_convex = np.zeros((n_pseudo, n_samples, n_components))
    cgh_model = getattr(models, "continuous_gh", None)
    X_cgh = np.zeros_like(X_convex) if cgh_model is not None else None

    if cgh_model is not None and training_interpolation is None:
        raise ValueError("training_interpolation is required for continuous GH lifting.")

    print("Lifting pseudo latents with batched time-local neighbour selection...")

    for i, t_star in enumerate(t_pseudo):
        idx_local = _get_time_local_indices(float(t_star))
        if idx_local is None or idx_local.size == 0:
            continue
        macro_local = macro_train_coords[idx_local]
        micro_local = micro_train_states[idx_local]

        local_chi = ConvexHullInterpolator(macro_states=macro_local, micro_states=micro_local)
        X_convex[i] = local_chi.batch_lift(
            phi_pseudo[i],
            k=config.convex_k,
            max_iter=config.convex_max_iter,
        )

        if cgh_model is not None and training_interpolation is not None:
            X_cgh[i] = cgh_model.predict(
                s=float(t_star),
                g_query=phi_pseudo[i],
                training_interpolation=training_interpolation,
            )

    out = {"convex": X_convex}
    if X_cgh is not None:
        out["cgh"] = X_cgh
    return out
