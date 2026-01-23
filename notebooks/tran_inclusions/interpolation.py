from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from diffmap.diffusion_maps import _orient_svd, evaluate_coordinate_splines, fit_coordinate_splines
from stiefel.stiefel import batch_stiefel_log, Stiefel_Exp, stiefel_exp_batch
from stiefel.barycenter import R_barycenter
from stiefel.projection_retraction import st_inv_retr_orthographic, st_retr_orthographic

from .config import LatentInterpolationResult
from .tc_embeddings import (
    align_frames_procrustes,
    align_frames_procrustes_with_rotations,
    apply_rotations_to_embeddings,
)


def build_dense_latent_trajectories(
    tc_result,
    times_train: np.ndarray,
    tc_embeddings_time: np.ndarray,
    n_dense: int = 200,
    frechet_mode: Literal['global', 'triplet'] = 'triplet',
    compute_global_basepoint: bool = False,
    metric_alpha: float = 0.0,
    compute_global: bool = False,
    compute_triplet: bool = False,
    compute_naive: bool = True,
    stiefel_workers: Optional[int] = None,
    stiefel_chunk_size: int = 8,
) -> LatentInterpolationResult:
    # Auto-enable flags based on mode if not explicitly set
    if frechet_mode == 'global':
        compute_global = True
    # elif frechet_mode == 'triplet':
    #     compute_triplet = True
    
    # If explicit flags are passed, they override/augment the mode-based selection
    # (e.g. if user passed compute_global=True, we respect it regardless of mode)

    # Create dense time grid that INCLUDES marginal times exactly as knots.
    # This ensures PCHIP and Stiefel interpolation satisfy knot constraints:
    # at marginal times, interpolated values equal original values exactly.
    # This is critical for consistency between pretraining (at marginals) and
    # fine-tuning (on dense trajectory) in downstream autoencoder training.
    t_dense_base = np.linspace(times_train.min(), times_train.max(), n_dense)
    t_dense = np.sort(np.unique(np.concatenate([t_dense_base, times_train])))
    n_times, n_samples, latent_dim = tc_embeddings_time.shape

    A_concatenated = np.concatenate([tc_result.A_operators[i] for i in range(len(tc_result.A_operators))], axis=1)
    from sklearn.decomposition import TruncatedSVD  # local import to avoid global dep in other modules

    svd_model = TruncatedSVD(
        n_components=len(tc_result.singular_values[0]) + 1,
        algorithm='randomized',
        random_state=42,
    )
    U_global = svd_model.fit_transform(A_concatenated)
    Vt_global = svd_model.components_
    U_global, Vt_global = _orient_svd(A_concatenated, U_global, Vt_global)
    U_global = U_global[:, 1:]

    U_train_list_raw = np.stack(tc_result.left_singular_vectors, axis=0)
    U_train_global = align_frames_procrustes(U_train_list_raw, U_global)
    tc_embeddings_time_aligned = tc_embeddings_time

    barycenter_kwargs = dict(stepsize=1.0, max_it=200, tol=1e-3, verbosity=False)
    U_frechet, U_bary_iters = R_barycenter(
        points=U_train_global,
        retr=st_retr_orthographic,
        inv_retr=st_inv_retr_orthographic,
        init=U_global,
        **barycenter_kwargs,
    )
    U_train_frechet, frechet_rotations = align_frames_procrustes_with_rotations(
        U_train_list_raw, U_frechet
    )
    # Recompute the aligned embeddings to match the reconstruction logic: (U @ R) * Sigma
    # We use the ORIGINAL U_train_list_raw (singular vectors) and the calculated rotations.
    # We also need the singular values and stationary distributions to reconstruct fully.
    
    tc_embeddings_time_aligned_list = []
    n_times_train = len(times_train)
    
    # Extract components from tc_result for the training knots
    # Note: tc_result.left_singular_vectors is a list of U matrices
    #       tc_result.singular_values is a list of sigma vectors
    #       tc_result.stationary_distributions is a list of pi vectors
    
    for i in range(n_times_train):
        # 1. Get the aligned basis for this knot: U_aligned = U_original @ R
        # frechet_rotations[i] is the R that maps U_original[i] to the Frechet mean frame
        U_orig = tc_result.left_singular_vectors[i]
        R = frechet_rotations[i]
        U_aligned = U_orig @ R
        
        # 2. Get Sigma and Pi
        S = tc_result.singular_values[i]
        Pi = tc_result.stationary_distributions[i]
        
        # 3. Reconstruct Embedding: Phi = (U_aligned * S) / sqrt(Pi)
        # This matches exactly how 'reconstruct_embeddings' works for the dense spline
        phi_aligned = (U_aligned * S[None, :]) / np.sqrt(Pi)[:, None]
        
        tc_embeddings_time_aligned_list.append(phi_aligned)

    tc_embeddings_time_aligned = np.stack(tc_embeddings_time_aligned_list, axis=0)

    deltas_fm = batch_stiefel_log(U_frechet, U_train_frechet, metric_alpha=metric_alpha, tau=1e-2)

    def interpolate_deltas(
        deltas: np.ndarray,
        base_point: np.ndarray,
        method: str = 'pchip',
        workers: Optional[int] = stiefel_workers,
        chunk_size: int = stiefel_chunk_size,
    ):
        deltas_flat = deltas.reshape(deltas.shape[0], -1)
        if method == 'pchip':
            pchip = PchipInterpolator(times_train, deltas_flat, axis=0)
            deltas_dense_flat = pchip(t_dense)
        elif method == 'linear':
            pchip = None
            deltas_dense_flat = np.stack([
                np.interp(t_dense, times_train, deltas_flat[:, j])
                for j in range(deltas_flat.shape[1])
            ], axis=1)
        else:
            raise ValueError("method must be 'pchip' or 'linear'")
        deltas_dense = deltas_dense_flat.reshape(len(t_dense), *base_point.shape)
        U_dense = stiefel_exp_batch(
            base_point,
            deltas_dense,
            metric_alpha=metric_alpha,
            workers=workers,
            chunk_size=chunk_size,
        )
        return deltas_dense, U_dense, pchip

    # Interpolate global deltas to get dense U_global trajectory
    phi_global_dense = None
    pchip_delta_global = None
    if compute_global_basepoint:
        deltas_global = batch_stiefel_log(U_global, U_train_global, metric_alpha=metric_alpha, tau=1e-2)
        deltas_global_dense, U_dense_global, pchip_delta_global = interpolate_deltas(
            deltas_global, U_global, method='pchip'
        )

    def interpolate_deltas_triplet(
        method: str = 'pchip',
        workers: Optional[int] = stiefel_workers,
        chunk_size: int = stiefel_chunk_size,
    ):
        if n_times < 3:
            deltas_dense_fallback, U_dense_fallback, pchip_fallback = interpolate_deltas(
                deltas_fm, U_frechet, method=method
            )
            return deltas_dense_fallback, U_dense_fallback, None

        windows: List[Dict[str, Any]] = []
        for start in range(n_times - 2):
            idx_slice = slice(start, start + 3)
            t_window = times_train[idx_slice]
            U_window = U_train_frechet[idx_slice]
            init_guess = U_window[1] if len(U_window) > 1 else U_window[0]
            U_triplet, _ = R_barycenter(
                points=U_window,
                retr=st_retr_orthographic,
                inv_retr=st_inv_retr_orthographic,
                init=init_guess,
                **barycenter_kwargs,
            )
            # U_window is already aligned to the Global Fréchet Mean. 
            # Do not re-align to the local triplet mean, or we will risk rotate the basis.
            # U_window_aligned = align_frames_procrustes(U_window, U_triplet)
            deltas_window = batch_stiefel_log(U_triplet, U_window, metric_alpha=metric_alpha, tau=1e-2)
            deltas_flat = deltas_window.reshape(deltas_window.shape[0], -1)
            if method == 'pchip':
                interpolator = PchipInterpolator(t_window, deltas_flat, axis=0)
            elif method == 'linear':
                interpolator = interp1d(
                    t_window,
                    deltas_flat,
                    axis=0,
                    kind='linear',
                    assume_sorted=True,
                    fill_value='extrapolate',
                )
            else:
                raise ValueError("method must be 'pchip' or 'linear'")
            windows.append(
                {
                    't_min': float(t_window[0]),
                    't_max': float(t_window[-1]),
                    'base_point': U_triplet,
                    'interpolator': interpolator,
                    'times': t_window,
                }
            )

        tol = 1e-12
        bounds = np.array([(w['t_min'], w['t_max']) for w in windows])

        def _select_window_index(t_star: float) -> int:
            for idx, (t_min, t_max) in enumerate(bounds):
                if (t_min - tol) <= t_star <= (t_max + tol):
                    return idx
            return len(windows) - 1

        window_indices = np.array([
            _select_window_index(float(t_star)) for t_star in t_dense
        ], dtype=int)

        deltas_dense = np.zeros((len(t_dense),) + U_frechet.shape, dtype=U_frechet.dtype)
        U_dense = np.zeros_like(deltas_dense)

        for idx, window in enumerate(windows):
            mask = window_indices == idx
            if not np.any(mask):
                continue
            base_point = window['base_point']
            interpolator = window['interpolator']
            t_eval = t_dense[mask]
            delta_flat = interpolator(t_eval)
            delta_block = delta_flat.reshape(len(t_eval), *base_point.shape)
            deltas_dense[mask] = delta_block
            U_dense[mask] = stiefel_exp_batch(
                base_point,
                delta_block,
                metric_alpha=metric_alpha,
                workers=workers,
                chunk_size=chunk_size,
            )

        return deltas_dense, U_dense, windows

    pchip_delta_fm = None
    frechet_windows = None
    
    # Placeholders for results
    deltas_fm_dense_triplet = None
    U_dense_fm_triplet = None
    deltas_fm_dense_global = None
    U_dense_fm_global = None
    
    # Compute requested interpolations
    if compute_triplet:
        deltas_fm_dense_triplet, U_dense_fm_triplet, frechet_windows = interpolate_deltas_triplet(method='pchip')
        # We also compute linear for triplet if it's the main mode, but for now let's just stick to pchip for the dense trajectories
        # The original code computed linear as well. Let's keep it if possible, but maybe just for the main mode?
        # For simplicity, let's compute linear only if it's the main mode or if we want to be thorough.
        # The return type only has one phi_linear_dense. Let's assume linear is coupled with the main mode.
        
    if compute_global:
        deltas_fm_dense_global, U_dense_fm_global, pchip_delta_fm = interpolate_deltas(deltas_fm, U_frechet, method='pchip')


    # Determine which one is the "main" one for backward compatibility
    if frechet_mode == 'triplet':
        if not compute_triplet:
             # Should not happen due to auto-enable logic
             raise ValueError("frechet_mode is triplet but compute_triplet is False")
        U_dense_fm = U_dense_fm_triplet
        # Compute linear for triplet
        _, U_dense_fm_linear, _ = interpolate_deltas_triplet(method='linear')
    elif frechet_mode == 'global':
        if not compute_global:
             # If compute_global was explicitly disabled but mode is global, we must enable it effectively or error.
             # But here we rely on the flags.
             if deltas_fm_dense_global is None: 
                # Recalculate if it was skipped above
                 deltas_fm_dense_global, U_dense_fm_global, pchip_delta_fm = interpolate_deltas(deltas_fm, U_frechet, method='pchip')

        U_dense_fm = U_dense_fm_global
        # Compute linear for global
        _, U_dense_fm_linear, _ = interpolate_deltas(deltas_fm, U_frechet, method='linear')
    else:
        raise ValueError("frechet_mode must be 'global' or 'triplet'")

    sigmas = np.stack(tc_result.singular_values)
    log_sigmas = np.log(sigmas + 1e-16)
    pchip_sigma = PchipInterpolator(times_train, log_sigmas, axis=0)
    log_sigmas_dense = pchip_sigma(t_dense)
    sigmas_dense = np.exp(log_sigmas_dense)
    log_sigmas_linear = np.stack([
        np.interp(t_dense, times_train, log_sigmas[:, j])
        for j in range(log_sigmas.shape[1])
    ], axis=1)
    sigmas_linear = np.exp(log_sigmas_linear)

    pis = np.stack(tc_result.stationary_distributions)
    log_pis = np.log(pis + 1e-16)
    pchip_pi = PchipInterpolator(times_train, log_pis, axis=0)
    log_pis_dense = pchip_pi(t_dense)
    pis_dense_unnorm = np.exp(log_pis_dense)
    pis_dense = pis_dense_unnorm / pis_dense_unnorm.sum(axis=1, keepdims=True)
    log_pis_linear = np.stack([
        np.interp(t_dense, times_train, log_pis[:, j])
        for j in range(log_pis.shape[1])
    ], axis=1)
    pis_linear_unnorm = np.exp(log_pis_linear)
    pis_linear = pis_linear_unnorm / pis_linear_unnorm.sum(axis=1, keepdims=True)

    def reconstruct_embeddings(U_dense: np.ndarray, sig_series: np.ndarray, pi_series: np.ndarray) -> np.ndarray:
        phi_list = []
        for i in range(len(t_dense)):
            U = U_dense[i]
            S = sig_series[i]
            Pi = pi_series[i]
            phi_list.append((U * S[None, :]) / np.sqrt(Pi)[:, None])
        return np.array(phi_list)

    if compute_global_basepoint:
        phi_global_dense = reconstruct_embeddings(U_dense_global, sigmas_dense, pis_dense)
    
    # Reconstruct specific versions
    phi_frechet_triplet_dense = None
    if U_dense_fm_triplet is not None:
        phi_frechet_triplet_dense = reconstruct_embeddings(U_dense_fm_triplet, sigmas_dense, pis_dense)
        
    phi_frechet_global_dense = None
    if U_dense_fm_global is not None:
        phi_frechet_global_dense = reconstruct_embeddings(U_dense_fm_global, sigmas_dense, pis_dense)

    # Main return value
    phi_frechet_dense = reconstruct_embeddings(U_dense_fm, sigmas_dense, pis_dense)
    phi_linear_dense = reconstruct_embeddings(U_dense_fm_linear, sigmas_linear, pis_linear)

    phi_naive_dense = None
    if compute_naive:
        sample_splines = []
        for sample_idx in range(n_samples):
            coords_sample = tc_embeddings_time_aligned[:, sample_idx, :]
            splines = fit_coordinate_splines(
                coords_sample,
                times_train,
                spline_type='pchip',
                window_mode='triplet',
            )
            sample_splines.append(splines)

        phi_naive_dense = np.stack([
            np.vstack([
                evaluate_coordinate_splines(splines, t).ravel()
                for t in t_dense
            ])
            for splines in sample_splines
        ], axis=1)

    return LatentInterpolationResult(
        t_dense=t_dense,
        phi_global_dense=phi_global_dense,
        phi_frechet_dense=phi_frechet_dense,
        phi_frechet_global_dense=phi_frechet_global_dense,
        phi_frechet_triplet_dense=phi_frechet_triplet_dense,
        phi_linear_dense=phi_linear_dense,
        phi_naive_dense=phi_naive_dense,
        pchip_delta_global=pchip_delta_global,
        pchip_delta_fm=pchip_delta_fm,
        pchip_sigma=pchip_sigma,
        pchip_pi=pchip_pi,
        U_global=U_global,
        U_frechet=U_frechet,
        frechet_rotations=frechet_rotations,
        frechet_mode=frechet_mode,
        frechet_windows=frechet_windows,
        tc_embeddings_aligned=tc_embeddings_time_aligned,
    )


def sample_latent_at_times(
    interpolation: LatentInterpolationResult,
    pseudo_times: np.ndarray,
    method: Literal['naive', 'global', 'frechet'] = 'frechet',
) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'naive':
        phi_dense = interpolation.phi_naive_dense
    elif method == 'global':
        phi_dense = interpolation.phi_global_dense
        if phi_dense is None:
            raise ValueError("Global basepoint interpolation was not computed.")
    elif method == 'frechet':
        phi_dense = interpolation.phi_frechet_dense
    else:
        raise ValueError(f"Unknown method {method}")

    if phi_dense is None:
        raise ValueError(f"Interpolation for method '{method}' is not available.")

    phi_pseudo = np.zeros((len(pseudo_times), phi_dense.shape[1], phi_dense.shape[2]))
    for i in range(phi_dense.shape[1]):
        for j in range(phi_dense.shape[2]):
            pchip = PchipInterpolator(interpolation.t_dense, phi_dense[:, i, j])
            phi_pseudo[:, i, j] = pchip(pseudo_times)

    return pseudo_times, phi_pseudo
