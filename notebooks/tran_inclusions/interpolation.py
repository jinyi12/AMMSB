from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from diffmap.diffusion_maps import _orient_svd, evaluate_coordinate_splines, fit_coordinate_splines
from stiefel.stiefel import batch_stiefel_log, Stiefel_Exp
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
) -> LatentInterpolationResult:
    t_dense = np.linspace(times_train.min(), times_train.max(), n_dense)
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

    barycenter_kwargs = dict(stepsize=1.0, max_it=200, tol=1e-5, verbosity=True)
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
    tc_embeddings_time_aligned = apply_rotations_to_embeddings(
        tc_embeddings_time, frechet_rotations
    )

    deltas_global = batch_stiefel_log(U_global, U_train_global, metric_alpha=1e-8, tau=1e-2)
    deltas_fm = batch_stiefel_log(U_frechet, U_train_frechet, metric_alpha=1e-8, tau=1e-2)

    def interpolate_deltas(deltas: np.ndarray, base_point: np.ndarray, method: str = 'pchip'):
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
        U_dense = [
            Stiefel_Exp(U0=base_point, Delta=d, metric_alpha=1e-8)
            for d in deltas_dense
        ]
        return deltas_dense, np.array(U_dense), pchip

    def interpolate_deltas_triplet(method: str = 'pchip'):
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
            U_window_aligned = align_frames_procrustes(U_window, U_triplet)
            deltas_window = batch_stiefel_log(U_triplet, U_window_aligned, metric_alpha=1e-8, tau=1e-2)
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

        def _select_window(t_star: float):
            tol = 1e-12
            for window in windows:
                if (window['t_min'] - tol) <= t_star <= (window['t_max'] + tol):
                    return window
            return windows[-1]

        deltas_dense_list = []
        U_dense_list = []
        for t_star in t_dense:
            window = _select_window(float(t_star))
            base_point = window['base_point']
            interpolator = window['interpolator']
            delta_flat = interpolator(t_star)
            delta = delta_flat.reshape(base_point.shape)
            U_star = Stiefel_Exp(U0=base_point, Delta=delta, metric_alpha=1e-8)
            deltas_dense_list.append(delta)
            U_dense_list.append(U_star)
        return np.array(deltas_dense_list), np.array(U_dense_list), windows

    pchip_delta_global = None
    pchip_delta_fm = None
    frechet_windows = None
    if frechet_mode == 'triplet':
        deltas_fm_dense, U_dense_fm, frechet_windows = interpolate_deltas_triplet(method='pchip')
        deltas_fm_linear, U_dense_fm_linear, _ = interpolate_deltas_triplet(method='linear')
        pchip_delta_fm = frechet_windows
    elif frechet_mode == 'global':
        deltas_fm_dense, U_dense_fm, pchip_delta_fm = interpolate_deltas(deltas_fm, U_frechet, method='pchip')
        deltas_fm_linear, U_dense_fm_linear, _ = interpolate_deltas(deltas_fm, U_frechet, method='linear')
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

    phi_global_dense = reconstruct_embeddings(U_train_global, sigmas_dense, pis_dense)
    phi_frechet_dense = reconstruct_embeddings(U_dense_fm, sigmas_dense, pis_dense)
    phi_linear_dense = reconstruct_embeddings(U_dense_fm_linear, sigmas_linear, pis_linear)

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
        phi_linear_dense=phi_linear_dense,
        phi_naive_dense=phi_naive_dense,
        pchip_delta_global=pchip_delta_global,
        pchip_delta_fm=pchip_delta_fm,
        pchip_sigma=pchip_sigma,
        pchip_pi=pchip_pi,
        U_global=U_global,
        U_frechet=U_frechet,
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
    elif method == 'frechet':
        phi_dense = interpolation.phi_frechet_dense
    else:
        raise ValueError(f"Unknown method {method}")

    phi_pseudo = np.zeros((len(pseudo_times), phi_dense.shape[1], phi_dense.shape[2]))
    for i in range(phi_dense.shape[1]):
        for j in range(phi_dense.shape[2]):
            pchip = PchipInterpolator(interpolation.t_dense, phi_dense[:, i, j])
            phi_pseudo[:, i, j] = pchip(pseudo_times)

    return pseudo_times, phi_pseudo
