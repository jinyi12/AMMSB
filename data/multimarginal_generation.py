from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import scipy.stats as stats
from torch import Tensor
from tqdm import trange


# ---------------------------------------------------------------------------
# Gaussian smoothing utilities
# ---------------------------------------------------------------------------


def gaussian_blur_periodic(input_tensor: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply Gaussian blur with circular padding to emulate periodic boundaries."""
    if sigma <= 1e-9 or kernel_size <= 1:
        return input_tensor

    if kernel_size % 2 == 0:
        kernel_size += 1

    k = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device)
    center = (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * ((k - center) / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)

    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be [B, C, H, W]")

    channels = input_tensor.shape[1]
    kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2

    padded = torch.nn.functional.pad(
        input_tensor,
        (padding, padding, padding, padding),
        mode="circular",
    )
    output = torch.nn.functional.conv2d(padded, kernel, padding=0, groups=channels)
    return output


# ---------------------------------------------------------------------------
# Tran-style truncated Gaussian filter (Eq. 9 in Tran et al.)
# ---------------------------------------------------------------------------


def build_tran_kernel(
    H: float,
    pixel_size: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_half_width_pix: int | None = None,
) -> Tuple[Tensor | None, int]:
    """
    Discrete analogue of Tran et al.'s truncated Gaussian filter:

        ρ_H(x) ∝ exp(-||x||^2 / (2 γ^2)),  |x_i| ≤ H/2,  γ = H / 6.

    Returns:
        kernel_2d : [kH, kW] tensor with sum(kernel_2d) == 1 (or None if H≈0)
        half_width_pix : integer half-width in pixels
    """
    if H <= 1e-9:
        return None, 0

    half_width_pix = max(1, math.ceil(H / (2.0 * pixel_size)))
    if max_half_width_pix is not None:
        half_width_pix = max(1, min(half_width_pix, max_half_width_pix))

    coords = (
        torch.arange(-half_width_pix, half_width_pix + 1, device=device, dtype=dtype)
        * pixel_size
    )

    gamma = H / 6.0
    gauss_1d = torch.exp(-0.5 * (coords / gamma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    gauss_2d = gauss_2d / gauss_2d.sum()

    return gauss_2d, half_width_pix


def tran_filter_periodic(
    input_tensor: Tensor,
    H: float,
    pixel_size: float,
) -> Tensor:
    """
    Apply Tran-style truncated Gaussian filter with periodic padding.

    Args
    ----
    input_tensor : [B, C, H, W] tensor
    H           : physical filter size
    pixel_size  : physical size of one pixel (L_domain / resolution)
    """
    if H <= 1e-9:
        return input_tensor

    if input_tensor.dim() != 4:
        raise ValueError("tran_filter_periodic expects a [B, C, H, W] tensor.")

    min_hw = min(input_tensor.shape[-2:])
    max_half_width_pix = max(1, (min_hw - 1) // 2)

    kernel_2d, half_width_pix = build_tran_kernel(
        H=H,
        pixel_size=pixel_size,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
        max_half_width_pix=max_half_width_pix,
    )

    channels = input_tensor.shape[1]
    kernel = kernel_2d.expand(channels, 1, *kernel_2d.shape)

    padding = half_width_pix
    padded = torch.nn.functional.pad(
        input_tensor,
        (padding, padding, padding, padding),
        mode="circular",
    )

    out = torch.nn.functional.conv2d(
        padded,
        kernel,
        padding=0,
        groups=channels,
    )
    return out


# ---------------------------------------------------------------------------
# Non-Gaussian transforms
# ---------------------------------------------------------------------------


def transform_to_non_gaussian(
    gaussian_field: np.ndarray,
    mu_target: float,
    sigma_target: float,
    distribution: str = "gamma",
) -> np.ndarray:
    """Map a Gaussian random field to a target distribution via PIT."""
    g_mean = np.mean(gaussian_field)
    g_std = np.std(gaussian_field)

    if g_std < 1e-9:
        standard_gaussian = np.zeros_like(gaussian_field)
    else:
        standard_gaussian = (gaussian_field - g_mean) / g_std

    z_normcdf = stats.norm.cdf(standard_gaussian, 0, 1)
    z_normcdf = np.clip(z_normcdf, 1e-9, 1 - 1e-9)

    if distribution == "gamma":
        shape = (mu_target / sigma_target) ** 2
        scale = (sigma_target**2) / mu_target
        return stats.gamma.ppf(z_normcdf, shape, scale=scale)

    if distribution == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + (sigma_target / mu_target) ** 2))
        mu_ln = np.log(mu_target) - 0.5 * sigma_ln**2
        return stats.lognorm.ppf(z_normcdf, s=sigma_ln, scale=np.exp(mu_ln))

    raise ValueError(f"Unsupported distribution type: {distribution}")


# ---------------------------------------------------------------------------
# Random field generation
# ---------------------------------------------------------------------------


@dataclass
class RandomFieldGenerator2D:
    """Generate 2D Gaussian random fields with either FFT or KL methods."""

    nx: int = 100
    ny: int = 100
    lx: float = 1.0
    ly: float = 1.0
    device: str = "cpu"
    generation_method: str = "kl"
    kl_error_threshold: float = 1e-3
    H_divisor: float = 6.0

    def __post_init__(self) -> None:
        method = self.generation_method.lower()
        if method not in {"fft", "kl"}:
            raise ValueError("generation_method must be 'fft' or 'kl'")
        self.generation_method = method

        self.kl_cache: Dict[Tuple[float, str, float], np.ndarray] = {}
        if self.generation_method == "kl":
            self._initialize_kl_mesh()

    def _initialize_kl_mesh(self) -> None:
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.xy_coords = np.column_stack((X.flatten(), Y.flatten()))

    def _get_kl_transform_matrix(self, correlation_length: float, covariance_type: str) -> np.ndarray:
        cache_key = (correlation_length, covariance_type, self.kl_error_threshold)
        if cache_key in self.kl_cache:
            return self.kl_cache[cache_key]

        distances = squareform(pdist(self.xy_coords, "euclidean"))
        l_val = correlation_length

        if covariance_type == "exponential":
            cov_matrix = np.exp(-distances / l_val)
        elif covariance_type == "gaussian":
            cov_matrix = np.exp(-((distances / l_val) ** 2) / 2.0)
        else:
            raise ValueError(f"Invalid covariance_type for KL method: {covariance_type}")

        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = np.maximum(0, eig_vals[idx])
        eig_vecs = eig_vecs[:, idx]

        total_variance = np.sum(eig_vals)
        if total_variance > 1e-9:
            error_func = 1 - (np.cumsum(eig_vals) / total_variance)
            truncation_idx = np.where(error_func <= self.kl_error_threshold)[0]
            n_truncate = truncation_idx[0] + 1 if truncation_idx.size > 0 else len(eig_vals)
        else:
            n_truncate = 0

        if n_truncate == 0:
            kl_transform = np.empty((len(self.xy_coords), 0))
        else:
            sqrt_eigs = np.sqrt(eig_vals[:n_truncate])
            kl_transform = eig_vecs[:, :n_truncate] * sqrt_eigs[np.newaxis, :]

        self.kl_cache[cache_key] = kl_transform
        return kl_transform

    def generate_random_field(
        self,
        mean: float = 10.0,
        std: float = 2.0,
        correlation_length: float = 0.2,
        covariance_type: str = "exponential",
    ) -> np.ndarray:
        if self.generation_method == "fft":
            return self._generate_fft(mean, std, correlation_length, covariance_type)
        return self._generate_kl(mean, std, correlation_length, covariance_type)

    def _generate_fft(
        self,
        mean: float,
        std: float,
        correlation_length: float,
        covariance_type: str,
    ) -> np.ndarray:
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coeff = np.fft.fft2(white_noise)

        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2)

        l_val = correlation_length
        if covariance_type == "exponential":
            denom = 1 + (l_val * K) ** 2
            power = (2 * np.pi * l_val**2) / np.maximum(1e-9, denom ** 1.5)
        elif covariance_type == "gaussian":
            power = np.pi * l_val**2 * np.exp(-((l_val * K) ** 2) / 4)
        else:
            raise ValueError("Invalid covariance_type")

        power = np.nan_to_num(power)
        fourier_coeff *= np.sqrt(power)
        field = np.fft.ifft2(fourier_coeff).real

        field_std = np.std(field)
        if field_std > 1e-9:
            field = (field - np.mean(field)) / field_std * std + mean
        else:
            field = np.full_like(field, mean)
        return field

    def _generate_kl(
        self,
        mean: float,
        std: float,
        correlation_length: float,
        covariance_type: str,
    ) -> np.ndarray:
        kl_transform = self._get_kl_transform_matrix(correlation_length, covariance_type)
        n_kl = kl_transform.shape[1]
        if n_kl == 0:
            return np.full((self.nx, self.ny), mean)

        xi = np.random.normal(0, 1, n_kl)
        field_flat = kl_transform @ xi
        field_flat = field_flat * std + mean
        return field_flat.reshape((self.nx, self.ny))

    def coarsen_field(self, field: Tensor | np.ndarray, H: float) -> Tensor:
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field).to(self.device)

        original_dim = field.dim()
        if original_dim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            field = field.unsqueeze(1)
        elif original_dim != 4:
            raise ValueError("Unsupported field dimensions")

        pixel_size = self.lx / self.nx
        filter_sigma_phys = H / self.H_divisor
        filter_sigma_pix = filter_sigma_phys / pixel_size

        if filter_sigma_pix < 1e-6:
            smooth = field
        else:
            kernel_size = int(6 * filter_sigma_pix)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)

        if original_dim == 2:
            return smooth.squeeze(0).squeeze(0)
        if original_dim == 3:
            return smooth.squeeze(1)
        return smooth


# ---------------------------------------------------------------------------
# Bidisperse inclusion microstructure utilities (Tran-style)
# ---------------------------------------------------------------------------


def generate_bidisperse_microstructure_2d(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    D_large: float,
    vol_frac_large: float = 0.2,
    vol_frac_small: float = 0.1,
    matrix_value: float = 1.0,
    inclusion_value: float = 1000.0,
    max_attempts: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate 2D bidisperse stiff inclusions (toy analogue of Tran et al.).

    Returns:
        field : [nx, ny] array with matrix_value in the matrix and
                inclusion_value inside circular inclusions.
    """
    if rng is None:
        rng = np.random.default_rng()

    field = np.full((nx, ny), matrix_value, dtype=np.float32)

    pixel_size_x = lx / nx
    pixel_size_y = ly / ny

    R_large = 0.5 * D_large
    D_small = D_large / 3.0
    R_small = 0.5 * D_small

    area_domain = lx * ly
    area_large = math.pi * R_large**2
    area_small = math.pi * R_small**2

    n_large_target = max(1, int(round(vol_frac_large * area_domain / area_large)))
    n_small_target = max(1, int(round(vol_frac_small * area_domain / area_small)))

    centers: list[tuple[float, float, float]] = []  # (x, y, r)

    def non_overlapping(x: float, y: float, r: float, min_gap: float = 0.0) -> bool:
        for (xc, yc, rc) in centers:
            if (x - xc) ** 2 + (y - yc) ** 2 < (r + rc + min_gap) ** 2:
                return False
        return True

    attempts = 0
    for radius, n_target in ((R_large, n_large_target), (R_small, n_small_target)):
        count = 0
        while count < n_target and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(0.0, lx)
            y = rng.uniform(0.0, ly)
            if non_overlapping(x, y, radius):
                centers.append((x, y, radius))
                count += 1

    xs = (np.arange(nx) + 0.5) * pixel_size_x
    ys = (np.arange(ny) + 0.5) * pixel_size_y
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    for (xc, yc, radius) in centers:
        mask = (X - xc) ** 2 + (Y - yc) ** 2 <= radius**2
        field[mask] = inclusion_value

    return field


# ---------------------------------------------------------------------------
# Scheduling utilities
# ---------------------------------------------------------------------------


def calculate_filter_schedule(
    num_constraints: int,
    H_max: float,
    L_domain: float,
    resolution: int,
    micro_corr_length: float,
    schedule_type: str,
    H_divisor: float = 6.0,
    concentration: float = 2.0,
) -> np.ndarray:
    if num_constraints <= 1:
        return np.array([H_max] if num_constraints == 1 else [])

    if schedule_type == "linear" or num_constraints == 2:
        return np.linspace(0, H_max, num_constraints)

    # Precompute numerical H_min used by multiple schedule types
    delta_x = L_domain / resolution
    H_min_numerical = 2 * H_divisor * delta_x
    H_min = max(H_min_numerical, micro_corr_length, 1e-6)

    if schedule_type == "geometric":
        if H_max <= H_min * 1.01:
            return np.linspace(0, H_max, num_constraints)

        H_non_zero = np.geomspace(H_min, H_max, num_constraints - 1)
        return np.concatenate(([0.0], H_non_zero))

    # 'power' or 'concentrated' schedule: concentrate more points near H_min (early times)
    # concentration > 1 clusters points closer to H_min; concentration == 1 gives linear in log-space
    if schedule_type in {"power", "concentrated"}:
        if num_constraints - 1 <= 0:
            return np.array([H_max] if num_constraints == 1 else [])

        # normalized positions from 0..1 for the non-zero entries
        u = np.linspace(0.0, 1.0, num_constraints - 1)
        # transform to cluster near 0 when concentration > 1
        u_t = u ** float(concentration)

        # interpolate multiplicatively between H_min and H_max in log-space
        # avoid division by zero; ensure H_min>0
        H_min = max(H_min, 1e-12)
        H_non_zero = H_min * ( (H_max / H_min) ** u_t )
        return np.concatenate(([0.0], H_non_zero))

    raise ValueError(f"Unknown schedule type: {schedule_type}")


# ---------------------------------------------------------------------------
# Data generation entry points
# ---------------------------------------------------------------------------


def generate_multiscale_grf_data(
    N_samples: int,
    T: float = 1.0,
    N_constraints: int = 5,
    resolution: int = 32,
    L_domain: float = 1.0,
    micro_corr_length: float = 0.1,
    H_max_factor: float = 0.5,
    mean_val: float = 10.0,
    std_val: float = 2.0,
    covariance_type: str = "exponential",
    device: str = "cpu",
    generation_method: str = "fft",
    kl_error_threshold: float = 1e-3,
    schedule_type: str = "geometric",
    H_divisor: float = 6.0,
    concentration: float = 2.0,
) -> Tuple[Dict[float, Tensor], int]:
    time_steps = torch.linspace(0, T, N_constraints)
    H_max = L_domain * H_max_factor
    H_schedule = calculate_filter_schedule(
        N_constraints,
        H_max,
        L_domain,
        resolution,
        micro_corr_length,
        schedule_type,
        H_divisor,
        concentration=concentration
    )

    marginal_data: Dict[float, Tensor] = {}
    data_dim = resolution * resolution

    generator = RandomFieldGenerator2D(
        nx=resolution,
        ny=resolution,
        lx=L_domain,
        ly=L_domain,
        device=device,
        generation_method=generation_method,
        kl_error_threshold=kl_error_threshold,
        H_divisor=H_divisor,
    )

    if generation_method == "kl":
        generator._get_kl_transform_matrix(micro_corr_length, covariance_type)

    micro_fields = []
    for _ in trange(N_samples, leave=False):
        field = generator.generate_random_field(
            mean=mean_val,
            std=std_val,
            correlation_length=micro_corr_length,
            covariance_type=covariance_type,
        )
        micro_fields.append(field)

    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)

    for i, t in enumerate(time_steps):
        t_val = float(t.item())
        H_t = H_schedule[i]
        coarsened = generator.coarsen_field(micro_fields_tensor, H=H_t)
        flattened = coarsened.reshape(N_samples, data_dim)
        marginal_data[t_val] = flattened

    return marginal_data, data_dim


def generate_tran_multiscale_inclusion_data(
    N_samples: int,
    resolution: int = 64,
    L_domain: float = 6.0,
    D_large: float = 1.0,
    vol_frac_large: float = 0.2,
    vol_frac_small: float = 0.1,
    matrix_value: float = 1.0,
    inclusion_value: float = 1000.0,
    H_meso_list: list[float] | None = None,
    H_macro: float | None = None,
    device: str = "cpu",
) -> Tuple[Dict[float, Tensor], int]:
    """
    Multi-marginal dataset using Tran-style filtering of bidisperse inclusions.

    Returns:
        marginal_data : dict mapping time t -> [N_samples, data_dim] tensors
                        t = 0.0  : microscale
                        0<t<1.0 : mesoscales (Tran filter with finite H)
                        t = 1.0  : macroscale (volume average)
        data_dim      : resolution * resolution
    """
    if H_meso_list is None:
        H_meso_list = [2.0 * D_large, 2.5 * D_large, 3.0 * D_large]

    if H_macro is None:
        # Default to a large filter size to represent the 'macro' state.
        # To be consistent with the Tran procedure, this is just the filter limit H -> Large.
        # We choose max(5*D, L_domain) to ensure it covers the domain structure.
        H_macro = max(5.0 * D_large, L_domain)

    # Calculate and report theoretical mean (Volume Average)
    vol_frac_total = vol_frac_large + vol_frac_small
    theoretical_mean = vol_frac_total * inclusion_value + (1 - vol_frac_total) * matrix_value
    print(f"Generating Tran-style data.")
    print(f"  Theoretical Mean (Vol Avg): {theoretical_mean:.4f}")
    print(f"  Params: Vf_tot={vol_frac_total:.2f}, Inc={inclusion_value}, Mat={matrix_value}")
    print(f"  Filter Schedule: Meso={H_meso_list}, Macro={H_macro:.2f}")

    data_dim = resolution * resolution
    pixel_size = L_domain / resolution

    rng = np.random.default_rng()

    micro_fields = []
    for _ in trange(N_samples, leave=False):
        micro_np = generate_bidisperse_microstructure_2d(
            nx=resolution,
            ny=resolution,
            lx=L_domain,
            ly=L_domain,
            D_large=D_large,
            vol_frac_large=vol_frac_large,
            vol_frac_small=vol_frac_small,
            matrix_value=matrix_value,
            inclusion_value=inclusion_value,
            rng=rng,
        )
        micro_fields.append(micro_np)

    micro_np_stack = np.stack(micro_fields, axis=0)  # [N, H, W]
    micro_torch = torch.tensor(micro_np_stack, dtype=torch.float32, device=device).unsqueeze(1)

    marginal_data: Dict[float, Tensor] = {}

    t_micro = 0.0
    marginal_data[t_micro] = micro_torch.reshape(N_samples, data_dim)

    for j, H in enumerate(H_meso_list, start=1):
        t_val = float(j) / (len(H_meso_list) + 1)
        filtered = tran_filter_periodic(micro_torch, H=H, pixel_size=pixel_size)
        marginal_data[t_val] = filtered.reshape(N_samples, data_dim)

    t_macro = 1.0
    # Apply filter with H_macro to retain some spatial structure
    filtered_macro = tran_filter_periodic(micro_torch, H=H_macro, pixel_size=pixel_size)
    marginal_data[t_macro] = filtered_macro.reshape(N_samples, data_dim)

    return marginal_data, data_dim


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_multimarginal_data(
    marginal_data: Dict[float, Tensor]
) -> Tuple[Dict[float, Tensor], Tensor, Tensor]:
    """Apply global standardization (single scalar mean/std across all pixels).

    Uses global standardization instead of per-pixel to preserve geometric structure.
    Per-pixel standardization destroys spatial relationships by forcing each location
    to have identical statistics, erasing natural spatial gradients and correlations.
    """
    concatenated = torch.cat(list(marginal_data.values()), dim=0)
    # Global standardization: single scalar mean and std across all features
    mean = torch.mean(concatenated)
    std = torch.std(concatenated)
    if std < 1e-6:
        std = torch.tensor(1.0, device=std.device, dtype=std.dtype)

    normalized: Dict[float, Tensor] = {}
    for t, samples in marginal_data.items():
        normalized[t] = (samples - mean) / std

    return normalized, mean, std


def minmax_scale_marginals(
    marginal_arrays: Dict[float, np.ndarray],
    *,
    eps: float,
) -> Tuple[Dict[float, np.ndarray], float, float]:
    """Apply global min–max scaling with epsilon for numerical safety.

    Following the functional_autoencoders convention (Bunker et al., 2025),
    scaling uses a **single global min and max** across all samples and pixels.
    Per-pixel normalization would destroy spatial structure and is incompatible
    with the FAE's mesh-invariant architecture (arbitrary query points, masking).

    Forward:  z = (u - data_min) / (data_max - data_min) + eps
    Inverse:  u = (z - eps) * scale + data_min

    Returns
    -------
    scaled : dict[float, ndarray]
    data_min : float   — global scalar minimum
    scale : float      — global scalar (data_max - data_min)
    """
    concatenated = np.concatenate(list(marginal_arrays.values()), axis=0)
    data_min = float(concatenated.min())
    data_max = float(concatenated.max())
    scale = data_max - data_min

    scaled: Dict[float, np.ndarray] = {}
    for t, arr in marginal_arrays.items():
        scaled_arr = ((arr - data_min) / scale) + eps
        scaled[t] = scaled_arr.astype(np.float32)
    return scaled, data_min, scale


def affine_standardize_marginals(
    marginal_arrays: Dict[float, np.ndarray],
    *,
    data_min: float,
    data_max: float,
    delta: float,
) -> Tuple[Dict[float, np.ndarray], float, float, float, float, float]:
    """Affine-normalise to [0,1], then global standardisation (optionally clipped).

    This transform is intended for bounded fields with known physical limits
    ``[data_min, data_max]``. It preserves the *linearity* of the underlying
    Tran filter response (convolution of indicator fields) while still producing
    roughly unit-scale values for diffusion-style training.

    Forward:
      p = (x - data_min) / (data_max - data_min)
      p = clip(p, delta, 1 - delta)   (optional; delta=0 disables)
      z = (p - mean_p) / std_p        (GLOBAL scalar mean/std)

    Inverse:
      p = z * std_p + mean_p
      p = clip(p, delta, 1 - delta)
      x = data_min + p * (data_max - data_min)

    Notes
    -----
    This linear transform avoids nonlinear distortions (e.g. logit) on the
    already-linear filtered volume fraction signal, preserving spatial structure.
    """
    if data_max <= data_min:
        raise ValueError(
            f"affine_standardize requires data_max > data_min. "
            f"Got data_min={data_min}, data_max={data_max}."
        )
    if delta < 0.0 or delta >= 0.5:
        raise ValueError(f"affine_standardize requires 0 <= delta < 0.5. Got delta={delta}.")

    scale = float(data_max - data_min)
    p_arrays: Dict[float, np.ndarray] = {}
    for t, arr in marginal_arrays.items():
        p = (arr - data_min) / scale
        if delta > 0:
            p = np.clip(p, delta, 1.0 - delta)
        p_arrays[t] = p.astype(np.float32)

    concatenated = np.concatenate(list(p_arrays.values()), axis=0)
    mean_p = float(concatenated.mean())
    std_p = float(concatenated.std())
    if std_p < 1e-10:
        std_p = 1.0

    scaled: Dict[float, np.ndarray] = {}
    for t, p in p_arrays.items():
        scaled[t] = ((p - mean_p) / std_p).astype(np.float32)

    return scaled, float(delta), mean_p, std_p, float(data_min), float(data_max)


def log_standardize_marginals(
    marginal_arrays: Dict[float, np.ndarray],
    *,
    eps: float,
) -> Tuple[Dict[float, np.ndarray], float, float, float]:
    """Apply log transform followed by global standardization for strictly positive data.

    This transformation is recommended for generative modeling of strictly positive fields:
    1. Log transform: R^+ → R (unbounded space)
    2. Global standardization: mean=0, std=1 using single scalar values

    Benefits:
    - Guarantees strict positivity after inverse transform (via exp)
    - Unbounded support (no hard boundaries to violate during generation)
    - Handles large dynamic ranges naturally
    - Works well with Gaussian-like data after filtering
    - PRESERVES GEOMETRY: Global standardization maintains spatial structure

    Why global (not per-pixel) standardization:
    - Per-pixel standardization destroys spatial relationships by forcing each pixel
      to have identical statistics, erasing natural gradients and correlations
    - Global standardization preserves relative differences between spatial locations
    - Critical for maintaining the geometric structure of physical fields

    Args:
        marginal_arrays: Dictionary mapping time → field arrays (N, data_dim)
        eps: Small epsilon added before log to ensure numerical stability

    Returns:
        scaled: Transformed marginal arrays
        log_epsilon: Epsilon used in log transform
        log_mean: Scalar mean of log-transformed data (global)
        log_std: Scalar std of log-transformed data (global)
    """
    # Check positivity
    concatenated = np.concatenate(list(marginal_arrays.values()), axis=0)
    if concatenated.min() <= 0:
        raise ValueError(
            f"log_standardize requires strictly positive data. "
            f"Found minimum value: {concatenated.min()}"
        )

    # Step 1: Log transform (positive → unbounded)
    log_arrays = {}
    for t, arr in marginal_arrays.items():
        log_arrays[t] = np.log(arr + eps)

    # Step 2: Compute GLOBAL statistics on log-transformed data (single scalar values)
    concatenated_log = np.concatenate(list(log_arrays.values()), axis=0)
    log_mean = float(concatenated_log.mean())  # Scalar
    log_std = float(concatenated_log.std())    # Scalar
    if log_std < 1e-10:
        log_std = 1.0  # Avoid division by zero

    # Step 3: Standardize using global mean/std
    scaled: Dict[float, np.ndarray] = {}
    for t, log_arr in log_arrays.items():
        standardized = (log_arr - log_mean) / log_std
        scaled[t] = standardized.astype(np.float32)

    return scaled, eps, log_mean, log_std


def parse_held_out_indices(
    raw_arg: str,
    sorted_times: list[float],
) -> tuple[list[int], list[float]]:
    """Parse a comma-separated list of indices and return valid indices/times."""

    indices: list[int] = []
    times: list[float] = []
    if not raw_arg:
        return indices, times

    seen = set()
    for token in raw_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            print("Warning: held_out_indices must contain integers. Ignoring invalid entry:", token)
            continue
        if idx < 0 or idx >= len(sorted_times):
            print(f"Warning: Held-out index {idx} is out of bounds (0-{len(sorted_times)-1}). Ignoring.")
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
        times.append(sorted_times[idx])
    return indices, times


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-marginal data (GRF or Tran-style inclusions).")
    parser.add_argument('--output_path', type=str, required=True, help='Path to save generated data.')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('--resolution', type=int, default=32, help='Spatial resolution of the fields.')
    parser.add_argument('--n_constraints', type=int, default=5, help='Number of marginals/time steps.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for generation.')
    parser.add_argument('--data_generator', type=str, default='grf',
                        choices=['grf', 'tran_inclusion'],
                        help='Choose base generator: Gaussian random field (grf) or bidisperse Tran-style inclusions.')
    parser.add_argument('--generation_method', type=str, default='kl', help='Method for random field generation: fft or kl.')
    parser.add_argument('--covariance_type', type=str, default='exponential', help='Covariance type: exponential or gaussian.')
    parser.add_argument('--schedule_type', type=str, default='geometric', help='Filter schedule type: linear or geometric.')
    parser.add_argument('--L_domain', type=float, default=1.0, help='Domain size.')
    parser.add_argument('--micro_corr_length', type=float, default=0.1, help='Micro-scale correlation length.')
    parser.add_argument('--H_max_factor', type=float, default=0.5, help='Max filter size as a fraction of domain size.')
    parser.add_argument('--T', type=float, default=1.0, help='Total time duration.')
    parser.add_argument('--H_divisor', type=float, default=6.0, help='Divisor for filter size to determine Gaussian kernel sigma.')
    parser.add_argument('--mean_val', type=float, default=10.0, help='Mean value of the random field.')
    parser.add_argument('--std_val', type=float, default=2.0, help='Standard deviation of the random field.')
    parser.add_argument('--tran_D_large', type=float, default=1.0, help='Large inclusion diameter for Tran-style generator.')
    parser.add_argument('--tran_vol_frac_large', type=float, default=0.2, help='Volume fraction of large inclusions.')
    parser.add_argument('--tran_vol_frac_small', type=float, default=0.1, help='Volume fraction of small inclusions.')
    parser.add_argument('--tran_matrix_value', type=float, default=1.0, help='Matrix value (e.g., shear modulus) for Tran-style generator.')
    parser.add_argument('--tran_inclusion_value', type=float, default=1000.0, help='Inclusion value for Tran-style generator.')
    parser.add_argument('--tran_H_macro', type=float, default=None,
                        help='Filter size for the macro scale (t=1.0). Defaults to max(5.0 * D_large, L_domain) to ensure coverage of domain structure.')
    parser.add_argument('--tran_H_meso_list', type=str, default='',
                        help='Comma-separated list of H values for mesoscales (physical units). Leave empty for defaults {2D, 2.5D, 3D}.')
    parser.add_argument('--dataset_format', type=str, default='both',
                        choices=['pca', 'raw', 'both'],
                        help='Choose whether to save PCA coefficients, raw fields, or both.')
    parser.add_argument('--n_components', type=float, default=0.999, help='Number of PCA components accounting for specified variance (if <1) or fixed number (if >=1).')
    parser.add_argument('--kl_error_threshold', type=float, default=1e-3, help='KL truncation error threshold.')
    parser.add_argument('--use_whitening', action='store_true',
                        help='Apply PCA whitening (scale by inverse sqrt eigenvalues). Defaults to standard PCA coefficients.')
    parser.add_argument('--whitening_epsilon', type=float, default=1e-6,
                        help='Floor added to PCA eigenvalues during whitening to avoid amplifying tiny-variance modes.')
    parser.add_argument(
        '--scale_mode',
        type=str,
        default='none',
        choices=['none', 'minmax', 'log_standardize', 'affine_standardize'],
        help=(
            "Optional scaling applied before saving/PCA; "
            "'minmax' scales to [0,1] using global scalar min/max, "
            "'log_standardize' applies log transform then global standardization, "
            "'affine_standardize' maps bounded data to [0,1] then global standardization."
        ),
    )
    parser.add_argument('--scaling_epsilon', type=float, default=1e-6,
                        help='Small epsilon added to min–max ranges to avoid zero-variance issues (useful before KDE).')
    parser.add_argument('--concentration', type=float, default=2.0, help='Concentration parameter for the schedule.')
    parser.add_argument('--held_out_indices', type=str, default='', help='Comma-separated list of time step indices (0-based) to exclude from PCA fitting.')
    args = parser.parse_args()

    dataset_format = args.dataset_format.lower()
    if dataset_format not in {'pca', 'raw', 'both'}:
        raise ValueError(f"Unsupported dataset_format '{args.dataset_format}'.")
    if dataset_format == 'raw' and args.use_whitening:
        raise ValueError("--use_whitening flag is incompatible with dataset_format='raw'.")

    tran_H_meso_list = None
    if args.tran_H_meso_list:
        tran_H_meso_list = []
        for h_str in args.tran_H_meso_list.split(","):
            h_str = h_str.strip()
            if not h_str:
                continue
            if h_str.upper().endswith("D"):
                # Parse "2D", "2.5D" etc. as multiples of D_large
                factor = float(h_str[:-1])
                tran_H_meso_list.append(factor * args.tran_D_large)
            else:
                tran_H_meso_list.append(float(h_str))

    # Generate data based on selected generator
    if args.data_generator == 'tran_inclusion':
        marginal_data, data_dim = generate_tran_multiscale_inclusion_data(
            N_samples=args.n_samples,
            resolution=args.resolution,
            L_domain=args.L_domain,
            D_large=args.tran_D_large,
            vol_frac_large=args.tran_vol_frac_large,
            vol_frac_small=args.tran_vol_frac_small,
            matrix_value=args.tran_matrix_value,
            inclusion_value=args.tran_inclusion_value,
            H_meso_list=tran_H_meso_list,
            H_macro=args.tran_H_macro,
            device=args.device,
        )
    else:
        marginal_data, data_dim = generate_multiscale_grf_data(
            T=args.T,
            N_samples=args.n_samples,
            N_constraints=args.n_constraints,
            resolution=args.resolution,
            L_domain=args.L_domain,
            micro_corr_length=args.micro_corr_length,
            H_max_factor=args.H_max_factor,
            mean_val=args.mean_val,
            std_val=args.std_val,
            covariance_type=args.covariance_type,
            device=args.device,
            generation_method=args.generation_method,
            kl_error_threshold=args.kl_error_threshold,
            schedule_type=args.schedule_type,
            H_divisor=args.H_divisor,
            concentration=args.concentration,
        )

    marginal_arrays = {t: samples.cpu().numpy() for t, samples in marginal_data.items()}
    sorted_times = sorted(marginal_arrays.keys())
    held_out_indices_list, held_out_times_list = parse_held_out_indices(
        args.held_out_indices,
        sorted_times,
    )
    save_dict: Dict[str, np.ndarray | float | bool] = {
        'data_dim': data_dim,
        'dataset_format': dataset_format,
        'scale_mode': args.scale_mode,
        'scaling_epsilon': float(args.scaling_epsilon),
        'data_generator': args.data_generator,
    }
    save_dict.update({
        'held_out_indices': np.array(held_out_indices_list, dtype=np.int32),
        'held_out_times': np.array(held_out_times_list, dtype=np.float32),
    })

    if args.scale_mode == 'minmax':
        print(f"Applying global min–max scaling with epsilon={args.scaling_epsilon}.")
        marginal_arrays, data_min, data_scale = minmax_scale_marginals(
            marginal_arrays, eps=float(args.scaling_epsilon)
        )
        save_dict.update({
            'minmax_data_min': np.float32(data_min),
            'minmax_data_scale': np.float32(data_scale),
        })
        print(f"  Global data_min={data_min:.4f}, scale={data_scale:.4f}")
    elif args.scale_mode == 'log_standardize':
        print(f"Applying log+standardization (GLOBAL) with epsilon={args.scaling_epsilon}.")
        print("  Step 1: Log transform (R^+ → R)")
        print("  Step 2: Global standardize (single scalar mean=0, std=1)")
        print("  → Guarantees strict positivity after inverse: exp(unstandardize(z))")
        print("  → Preserves geometry (per-pixel standardization would destroy spatial structure)")
        marginal_arrays, log_eps, log_mean, log_std = log_standardize_marginals(
            marginal_arrays, eps=float(args.scaling_epsilon)
        )
        save_dict.update({
            'log_epsilon': np.float32(log_eps),
            'log_mean': np.float32(log_mean),
            'log_std': np.float32(log_std),
        })
        print(f"  ✓ Global log-space mean: {log_mean:.4f}")
        print(f"  ✓ Global log-space std: {log_std:.4f}")
    elif args.scale_mode == 'affine_standardize':
        if args.data_generator != 'tran_inclusion':
            raise ValueError(
                "scale_mode='affine_standardize' requires bounded data with known min/max. "
                "Use it with data_generator='tran_inclusion'."
            )
        delta = float(args.scaling_epsilon)
        data_min = float(min(args.tran_matrix_value, args.tran_inclusion_value))
        data_max = float(max(args.tran_matrix_value, args.tran_inclusion_value))
        print(
            "Applying affine→global-standardization (GLOBAL): "
            f"bounds=[{data_min}, {data_max}], delta={delta}."
        )
        marginal_arrays, affine_delta, affine_mean, affine_std, affine_min, affine_max = affine_standardize_marginals(
            marginal_arrays,
            data_min=data_min,
            data_max=data_max,
            delta=delta,
        )
        save_dict.update({
            'affine_delta': np.float32(affine_delta),
            'affine_mean': np.float32(affine_mean),
            'affine_std': np.float32(affine_std),
            'affine_min': np.float32(affine_min),
            'affine_max': np.float32(affine_max),
        })
        print(f"  ✓ Global affine-space mean: {affine_mean:.4f}")
        print(f"  ✓ Global affine-space std: {affine_std:.4f}")

    if dataset_format in {'raw', 'both'}:
        print("Saving raw (uncompressed) fields.")
        for t, arr in marginal_arrays.items():
            save_dict[f'raw_marginal_{t}'] = arr.astype(np.float32)

    if dataset_format in {'pca', 'both'}:
        # Fit PCA and compute coefficients using [η_d] = [μ]^(-1/2)[φ]^T ([x_d] - [x])
        held_out_times_set = set(held_out_times_list)
        if held_out_times_set:
            print(
                f"Held out times from PCA fit (indices {held_out_indices_list}): {sorted(held_out_times_set)}"
            )

        fit_data_list = []
        for t, arr in marginal_arrays.items():
            # Check if t is explicitly held out
            is_held_out = t in held_out_times_set

            if args.data_generator == 'tran_inclusion' and t < 1e-9:
                # Exclude t=0 (microscale) from PCA training data to focus on meso/macro structures
                if not is_held_out:
                    print("Excluding t=0 (microscale) from PCA fit for tran_inclusion.")
                is_held_out = True
            
            if not is_held_out:
                fit_data_list.append(arr)
        
        if not fit_data_list:
            print("Warning: No time steps selected for PCA fit. Reverting to using all data.")
            fit_data_list = list(marginal_arrays.values())

        all_data_fit = np.concatenate(fit_data_list, axis=0)

        print("Concatenated data shape for PCA fit:", all_data_fit.shape)
        pca = PCA(n_components=args.n_components)
        pca.fit(all_data_fit)

        # Compute coefficients: [μ]^(-1/2)[φ]^T ([x_d] - [x])
        eigenvectors = pca.components_.T  # [φ] is (n × ν)
        eigenvalues = pca.explained_variance_  # [μ] is (ν,)
        mean = pca.mean_  # [x]

        is_whitened = bool(args.use_whitening)
        if is_whitened:
            print("Using WHITENED PCA coefficients.")
            eig_floor = np.maximum(eigenvalues, float(args.whitening_epsilon))
            inv_sqrt_eig = 1.0 / np.sqrt(eig_floor)
        else:
            print("Using STANDARD PCA coefficients.")
            inv_sqrt_eig = None

        marginal_coeffs = {}
        for t, arr in marginal_arrays.items():
            centered = arr - mean  # [x_d] - [x]
            coeffs = centered @ eigenvectors
            if is_whitened:
                coeffs *= inv_sqrt_eig
            marginal_coeffs[t] = coeffs.astype(np.float32)

        print("Shapes of marginal coefficients:")
        for t, coeffs in marginal_coeffs.items():
            print(f"Time {t}: {coeffs.shape}")

        save_dict.update({f'marginal_{t}': coeffs for t, coeffs in marginal_coeffs.items()})
        save_dict.update({
            'pca_components': pca.components_.astype(np.float32),
            'pca_mean': pca.mean_.astype(np.float32),
            'pca_explained_variance': pca.explained_variance_.astype(np.float32),
            'is_whitened': is_whitened,
            'whitening_epsilon': float(args.whitening_epsilon),
        })
        print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    np.savez(args.output_path, **save_dict)
    print(f"Saved to {args.output_path}")


if __name__ == '__main__':
    main()
