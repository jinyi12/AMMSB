from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import cdist, pdist, squareform

from diffmap.diffusion_maps import select_optimal_bandwidth
from stiefel.barycenter import R_barycenter
from stiefel.stiefel import batch_stiefel_log, Stiefel_Exp
from stiefel.projection_retraction import st_inv_retr_orthographic, st_retr_orthographic

try:  # Prefer the shared Procrustes helpers used throughout the notebook utilities.
    from notebooks.tran_inclusions.tc_embeddings import (
        align_frames_procrustes,
        align_frames_procrustes_with_rotations,
    )
except Exception:  # pragma: no cover - fallback for environments without the notebook package.
    def align_frames_procrustes(U_stack: np.ndarray, U_ref: np.ndarray) -> np.ndarray:
        aligned = []
        for i in range(U_stack.shape[0]):
            Ui = U_stack[i]
            M = Ui.T @ U_ref
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            aligned.append(Ui @ R)
        return np.stack(aligned, axis=0)

    def align_frames_procrustes_with_rotations(
        U_stack: np.ndarray, U_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        aligned = []
        rotations = []
        for i in range(U_stack.shape[0]):
            Ui = U_stack[i]
            M = Ui.T @ U_ref
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            aligned.append(Ui @ R)
            rotations.append(R)
        return np.stack(aligned, axis=0), np.stack(rotations, axis=0)


if TYPE_CHECKING:  # Avoid runtime dependency to prevent circular imports.
    from notebooks.tran_inclusions.config import LatentInterpolationResult


__all__ = ["ContinuousGeometricHarmonics", "ContinuousGeometricHarmonicsResult"]


@dataclass
class ContinuousGeometricHarmonicsResult:
    times: np.ndarray
    epsilons: np.ndarray
    semigroup_errors: np.ndarray
    epsilon_grid: np.ndarray
    semigroup_error_grid: np.ndarray
    eigenvalues: np.ndarray
    basis_frames: np.ndarray
    coefficients: np.ndarray
    barycenter: np.ndarray
    delta_spline: PchipInterpolator
    log_lambda_spline: PchipInterpolator
    log_epsilon_spline: PchipInterpolator
    coeff_spline: PchipInterpolator
    metric_alpha: float
    log_tau: float
    ridge: float
    interpolation_method: str
    semigroup_selection: str


class ContinuousGeometricHarmonics:
    """Continuous-time geometric harmonics lifting via Stiefel interpolation."""

    def __init__(
        self,
        *,
        n_eigenvectors: Optional[int] = None,
        max_modes: Optional[int] = None,
        eigenvalue_delta: float = 1e-3,
        energy_threshold: float = 0.999,
        semigroup_norm: str = "operator",
        epsilon_grid_size: int = 18,
        epsilon_log_span: float = 2.0,
        semigroup_selection: Literal["global_min", "first_local_minimum"] = "first_local_minimum",
        metric_alpha: float = 1e-8,
        log_tau: float = 1e-2,
        ridge: float = 0.0,
        interpolation_method: str = "frechet",
        barycenter_kwargs: Optional[dict[str, Any]] = None,
        semigroup_tolerance: float = 0.1,
    ) -> None:
        self.n_eigenvectors = n_eigenvectors
        self.max_modes = max_modes
        self.eigenvalue_delta = eigenvalue_delta
        self.energy_threshold = energy_threshold
        self.semigroup_norm = semigroup_norm
        self.epsilon_grid_size = epsilon_grid_size
        self.epsilon_log_span = epsilon_log_span
        self.semigroup_selection = semigroup_selection
        self.metric_alpha = metric_alpha
        self.log_tau = log_tau
        self.ridge = ridge
        self.interpolation_method = interpolation_method
        self.barycenter_kwargs = barycenter_kwargs or {
            "stepsize": 1.0,
            "max_it": 200,
            "tol": 1e-5,
            "verbosity": False,
        }
        self.semigroup_tolerance = semigroup_tolerance

        self._result: Optional[ContinuousGeometricHarmonicsResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        times: np.ndarray,
        latent_points: np.ndarray,
        ambient_points: np.ndarray,
        *,
        bandwidth_candidates: Optional[Sequence[float]] = None,
    ) -> ContinuousGeometricHarmonics:
        """Fit the continuous geometric harmonics model.

        Parameters
        ----------
        times: (K,) array
            Snapshot times (will be sorted internally).
        latent_points: (K, N, m) array
            Latent coordinates per time snapshot.
        ambient_points: (K, N, d) array
            Ambient coordinates paired with ``latent_points``.
        bandwidth_candidates: optional sequence
            Optional shared candidate epsilons. When omitted, a per-time
            log-spaced grid around the median pairwise distance is used.
        """

        t_arr = np.asarray(times, dtype=np.float64).reshape(-1)
        latent = np.asarray(latent_points, dtype=np.float64)
        ambient = np.asarray(ambient_points, dtype=np.float64)

        if latent.ndim != 3 or ambient.ndim != 3:
            raise ValueError("latent_points and ambient_points must be 3D arrays.")
        if latent.shape[:2] != ambient.shape[:2]:
            raise ValueError("latent_points and ambient_points must agree on time and sample axes.")
        if t_arr.shape[0] != latent.shape[0]:
            raise ValueError("times length must match the number of snapshots.")

        # Sort times to keep interpolation well-posed.
        sort_idx = np.argsort(t_arr)
        t_sorted = t_arr[sort_idx]
        latent = latent[sort_idx]
        ambient = ambient[sort_idx]

        n_times, n_samples, _ = latent.shape
        _, _, ambient_dim = ambient.shape
        if n_samples < 2:
            raise ValueError("At least two samples are required per snapshot.")

        epsilons: list[float] = []
        semigroup_errors: list[float] = []
        epsilon_grid_list: list[np.ndarray] = []
        semigroup_error_grid_list: list[np.ndarray] = []
        eigenvalues_full: list[np.ndarray] = []
        eigenvectors_full: list[np.ndarray] = []

        # Phase A: per-time kernel scale and eigendecomposition
        for idx in range(n_times):
            coords = latent[idx]
            distances2 = squareform(pdist(coords, metric="sqeuclidean"))
            candidates = self._build_bandwidth_grid(distances2, bandwidth_candidates)
            eps_k, score_k, eps_grid, sge_grid = select_optimal_bandwidth(
                coords,
                candidate_epsilons=np.asarray(candidates, dtype=np.float64),
                alpha=1.0,
                epsilon_scaling=4.0,
                norm=self.semigroup_norm,
                selection=self.semigroup_selection,
                return_all=True,
            )
            epsilons.append(float(eps_k))
            semigroup_errors.append(float(score_k))
            epsilon_grid_list.append(np.asarray(eps_grid, dtype=np.float64))
            semigroup_error_grid_list.append(np.asarray(sge_grid, dtype=np.float64))

            kernel = np.exp(-distances2 / eps_k)
            np.fill_diagonal(kernel, 0.0)
            eigvals, eigvecs = np.linalg.eigh(kernel)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            eigenvalues_full.append(eigvals)
            eigenvectors_full.append(eigvecs)

        # Effective mode selection (max across all snapshots, optional cap)
        n_modes = self._select_mode_count(eigenvalues_full, n_samples)

        # Truncate eigenpairs and align bases to a reference before barycenter
        psi_raw = np.stack([vecs[:, :n_modes] for vecs in eigenvectors_full], axis=0)
        lambda_trunc = np.stack([vals[:n_modes] for vals in eigenvalues_full], axis=0)

        psi_ref_aligned, _ = align_frames_procrustes_with_rotations(psi_raw, psi_raw[0])

        # Compute coefficients in the aligned reference frame
        coeffs_ref = np.zeros((n_times, n_modes, ambient_dim))
        for idx in range(n_times):
            coeffs_ref[idx] = psi_ref_aligned[idx].T @ ambient[idx]

        # Phase B: Frechet mean on the Stiefel manifold and tangent vectors
        barycenter_init = psi_ref_aligned[0]
        barycenter, _ = R_barycenter(
            points=psi_ref_aligned,
            retr=st_retr_orthographic,
            inv_retr=st_inv_retr_orthographic,
            init=barycenter_init,
            **self.barycenter_kwargs,
        )

        psi_aligned, _ = align_frames_procrustes_with_rotations(psi_ref_aligned, barycenter)
        coeffs_aligned = np.zeros_like(coeffs_ref)
        for idx in range(n_times):
            coeffs_aligned[idx] = psi_aligned[idx].T @ ambient[idx]

        deltas = batch_stiefel_log(barycenter, psi_aligned, tau=self.log_tau, metric_alpha=self.metric_alpha)

        # Phase B.3: spline construction (flatten tangent and coefficient trajectories)
        delta_spline = self._fit_pchip(t_sorted, deltas.reshape(n_times, -1))
        log_lambda = np.log(np.maximum(lambda_trunc, 1e-16))
        log_lambda_spline = self._fit_pchip(t_sorted, log_lambda)
        log_eps = np.log(np.maximum(np.asarray(epsilons, dtype=np.float64), 1e-16))
        log_eps_spline = self._fit_pchip(t_sorted, log_eps)
        coeff_spline = self._fit_pchip(t_sorted, coeffs_aligned.reshape(n_times, -1))
        epsilon_grid = np.stack(epsilon_grid_list, axis=0)
        semigroup_error_grid = np.stack(semigroup_error_grid_list, axis=0)

        self._result = ContinuousGeometricHarmonicsResult(
            times=t_sorted,
            epsilons=np.asarray(epsilons, dtype=np.float64),
            semigroup_errors=np.asarray(semigroup_errors, dtype=np.float64),
            epsilon_grid=epsilon_grid,
            semigroup_error_grid=semigroup_error_grid,
            eigenvalues=lambda_trunc,
            basis_frames=psi_aligned,
            coefficients=coeffs_aligned,
            barycenter=barycenter,
            delta_spline=delta_spline,
            log_lambda_spline=log_lambda_spline,
            log_epsilon_spline=log_eps_spline,
            coeff_spline=coeff_spline,
            metric_alpha=self.metric_alpha,
            log_tau=self.log_tau,
            ridge=self.ridge,
            interpolation_method=self.interpolation_method,
            semigroup_selection=self.semigroup_selection,
        )

        if np.any(self._result.semigroup_errors > self.semigroup_tolerance):
            worst = float(np.max(self._result.semigroup_errors))
            raise RuntimeError(
                f"Semigroup error check failed: max SGE={worst:.3e} exceeds tolerance {self.semigroup_tolerance}."
            )
        return self

    def predict(
        self,
        s: float,
        g_query: np.ndarray,
        training_interpolation: "LatentInterpolationResult",
    ) -> np.ndarray:
        """Lift latent query points ``g_query`` at time ``s`` into ambient space."""

        if self._result is None:
            raise RuntimeError("Model has not been fit. Call `fit` before `predict`.")

        res = self._result
        g_query = np.atleast_2d(np.asarray(g_query, dtype=np.float64))

        epsilon = float(np.exp(res.log_epsilon_spline(s)))
        lambdas = np.exp(res.log_lambda_spline(s)).reshape(-1)
        coeff_flat = res.coeff_spline(s)
        coeffs = coeff_flat.reshape(lambdas.shape[0], -1)

        delta_flat = res.delta_spline(s)
        delta = delta_flat.reshape(res.barycenter.shape)
        psi_s = Stiefel_Exp(U0=res.barycenter, Delta=delta, metric_alpha=res.metric_alpha)

        # Retrieve interpolated latent training positions at time s
        G_s = self._interpolate_latents(training_interpolation, s, method=res.interpolation_method)
        distances2 = cdist(g_query, G_s, metric="sqeuclidean")
        kernel_weights = np.exp(-distances2 / epsilon)

        denom = lambdas + res.ridge
        sign = np.sign(denom)
        sign[sign == 0] = 1.0
        safe_denom = sign * np.maximum(np.abs(denom), 1e-12)
        psi_star = kernel_weights @ psi_s
        psi_star = psi_star / safe_denom[np.newaxis, :]

        return psi_star @ coeffs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_bandwidth_grid(
        self,
        distances2: np.ndarray,
        bandwidth_candidates: Optional[Sequence[float]],
    ) -> np.ndarray:
        if bandwidth_candidates is not None:
            return np.asarray(bandwidth_candidates, dtype=np.float64)
        mask = distances2 > 0
        median_d2 = np.median(distances2[mask]) if np.any(mask) else 1.0
        base = float(max(median_d2, 1e-12))
        span = float(max(self.epsilon_log_span, 0.0))
        grid = base * np.logspace(-span, span, num=max(self.epsilon_grid_size, 3))
        return grid

    def _select_mode_count(self, eigenvalues: Sequence[np.ndarray], n_samples: int) -> int:
        if self.n_eigenvectors is not None:
            n_modes = int(min(max(self.n_eigenvectors, 1), n_samples))
        elif 0.0 < float(self.energy_threshold) < 1.0:
            energy_dims = []
            for vals in eigenvalues:
                if vals.size == 0:
                    energy_dims.append(1)
                    continue
                pos_vals = np.maximum(np.abs(vals), 0.0)
                total = float(np.sum(pos_vals))
                if total <= 0:
                    energy_dims.append(1)
                    continue
                cumulative = np.cumsum(pos_vals) / total
                k = int(np.searchsorted(cumulative, float(self.energy_threshold)) + 1)
                energy_dims.append(min(k, n_samples))
            n_modes = max(energy_dims) if energy_dims else 1
            if self.max_modes is not None:
                n_modes = min(n_modes, int(self.max_modes))
        else:
            eff_dims = []
            for vals in eigenvalues:
                if vals.size == 0:
                    eff_dims.append(1)
                    continue
                ref = float(np.max(vals))
                if ref <= 0:
                    eff_dims.append(1)
                    continue
                cutoff = ref * float(self.eigenvalue_delta)
                eff_dims.append(int(np.sum(vals >= cutoff)))
            n_modes = max(eff_dims) if eff_dims else 1
            n_modes = max(n_modes, 1)
            if self.max_modes is not None:
                n_modes = min(n_modes, int(self.max_modes))
        return n_modes

    def _fit_pchip(self, x: np.ndarray, y: np.ndarray) -> PchipInterpolator:
        return PchipInterpolator(x, y, axis=0, extrapolate=True)

    def _interpolate_latents(
        self,
        interpolation: "LatentInterpolationResult",
        s: float,
        *,
        method: str = "frechet",
    ) -> np.ndarray:
        # Prefer reconstructing the diffusion embedding directly from the stored
        # Stiefel/log-eigenvalue/stationary splines. Fall back to dense grids if needed.
        g_s = self._eval_embedding_from_splines(interpolation, s, method=method)
        if g_s is not None:
            return g_s

        # Fallback: interpolate the precomputed dense trajectory.
        t_dense = np.asarray(interpolation.t_dense, dtype=np.float64)
        g_dense = None
        if method == "frechet":
            g_dense = getattr(interpolation, "phi_frechet_dense", None)
        elif method == "global":
            g_dense = getattr(interpolation, "phi_global_dense", None)
        elif method == "naive":
            g_dense = getattr(interpolation, "phi_naive_dense", None)
        if g_dense is None:
            for candidate in (
                getattr(interpolation, "phi_frechet_dense", None),
                getattr(interpolation, "phi_global_dense", None),
                getattr(interpolation, "phi_naive_dense", None),
                getattr(interpolation, "phi_linear_dense", None),
            ):
                if candidate is not None:
                    g_dense = candidate
                    break
        if g_dense is None:
            raise ValueError("LatentInterpolationResult must provide a dense latent trajectory.")

        interp = PchipInterpolator(t_dense, np.asarray(g_dense, dtype=np.float64), axis=0, extrapolate=True)
        g_val = interp(s)
        return np.asarray(g_val, dtype=np.float64)

    def _eval_embedding_from_splines(
        self,
        interpolation: "LatentInterpolationResult",
        s: float,
        *,
        method: str,
    ) -> Optional[np.ndarray]:
        # Sigma/pi interpolation is shared across modes.
        pchip_sigma = getattr(interpolation, "pchip_sigma", None)
        pchip_pi = getattr(interpolation, "pchip_pi", None)
        sigma_s = None
        pi_s = None
        if pchip_sigma is not None:
            sigma_s = np.exp(pchip_sigma(s))
        if pchip_pi is not None:
            pi_raw = np.exp(pchip_pi(s))
            pi_sum = float(np.sum(pi_raw))
            pi_s = pi_raw / (pi_sum if pi_sum > 0 else 1.0)

        def _reconstruct(U: Optional[np.ndarray], sigma: Optional[np.ndarray], pi: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if U is None or sigma is None or pi is None:
                return None
            if U.shape[0] != pi.shape[0]:
                return None
            if U.shape[1] != sigma.shape[0]:
                return None
            return (U * sigma[None, :]) / np.sqrt(pi)[:, None]

        # Mode-specific reconstruction of the Stiefel frame.
        if method == "frechet":
            frechet_mode = getattr(interpolation, "frechet_mode", "global")
            if frechet_mode == "triplet":
                windows = getattr(interpolation, "frechet_windows", None)
                if windows:
                    window = self._select_frechet_window(windows, s)
                    delta_flat = window["interpolator"](s)
                    base_point = window["base_point"]
                    delta = delta_flat.reshape(base_point.shape)
                    U_s = Stiefel_Exp(U0=base_point, Delta=delta, metric_alpha=self.metric_alpha)
                    phi_s = _reconstruct(U_s, sigma_s, pi_s)
                    if phi_s is not None:
                        return phi_s
            else:
                delta_interp = getattr(interpolation, "pchip_delta_fm", None)
                U0 = getattr(interpolation, "U_frechet", None)
                if delta_interp is not None and U0 is not None:
                    delta_flat = delta_interp(s)
                    delta = delta_flat.reshape(U0.shape)
                    U_s = Stiefel_Exp(U0=U0, Delta=delta, metric_alpha=self.metric_alpha)
                    phi_s = _reconstruct(U_s, sigma_s, pi_s)
                    if phi_s is not None:
                        return phi_s
        elif method == "global":
            delta_interp = getattr(interpolation, "pchip_delta_global", None)
            U0 = getattr(interpolation, "U_global", None)
            if delta_interp is not None and U0 is not None:
                delta_flat = delta_interp(s)
                delta = delta_flat.reshape(U0.shape)
                U_s = Stiefel_Exp(U0=U0, Delta=delta, metric_alpha=self.metric_alpha)
                phi_s = _reconstruct(U_s, sigma_s, pi_s)
                if phi_s is not None:
                    return phi_s

        return None

    @staticmethod
    def _select_frechet_window(windows: Sequence[dict[str, Any]], t_star: float) -> dict[str, Any]:
        tol = 1e-12
        for window in windows:
            if (window["t_min"] - tol) <= t_star <= (window["t_max"] + tol):
                return window
        return windows[-1]
