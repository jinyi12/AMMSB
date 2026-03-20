from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm as _norm
from scipy.stats import wasserstein_distance as _w1_1d

from .sample import sample_batch
from .sde import DriftNet, SigmaFn, integrate_interval
from scripts.images.field_visualization import EASTERN_HUES, format_for_paper


matplotlib.use("Agg")
from matplotlib import pyplot as plt


HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME = "hierarchical_gaussian_path"
C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_FILL = EASTERN_HUES[0]
C_ACCENT = EASTERN_HUES[3]
C_GRID = "#D8D2CA"
C_TEXT = "#2A2621"
C_MEAN = "#1B1B1B"
FONT_LABEL = 8
FONT_TICK = 7
FONT_TITLE = 9


@dataclass(frozen=True)
class HierarchicalGaussianBenchmarkConfig:
    coarse_to_fine_dims: tuple[int, ...] = (4, 8, 16, 32)
    sigma_root: float = 0.75
    sigma_levels: tuple[float, ...] = (0.20, 0.15, 0.10)
    detail_strength: tuple[float, ...] = (0.18, 0.14, 0.10)
    tanh_gain: float = 0.85
    weight_seed: int = 12_345
    tau_max: float = 1.0
    tau_min: float = 0.0

    def __post_init__(self) -> None:
        dims = tuple(int(value) for value in self.coarse_to_fine_dims)
        if len(dims) < 2:
            raise ValueError("coarse_to_fine_dims must contain at least two levels.")
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"coarse_to_fine_dims must be positive, got {dims}.")
        if any(next_dim <= dim for dim, next_dim in zip(dims[:-1], dims[1:], strict=True)):
            raise ValueError(f"coarse_to_fine_dims must be strictly increasing, got {dims}.")
        if len(self.sigma_levels) != len(dims) - 1:
            raise ValueError(
                "sigma_levels must contain one value per refinement interval; "
                f"expected {len(dims) - 1}, got {len(self.sigma_levels)}."
            )
        if len(self.detail_strength) != len(dims) - 1:
            raise ValueError(
                "detail_strength must contain one value per refinement interval; "
                f"expected {len(dims) - 1}, got {len(self.detail_strength)}."
            )
        if any(float(value) <= 0.0 for value in self.sigma_levels):
            raise ValueError(f"sigma_levels must be positive, got {self.sigma_levels}.")
        if any(float(value) <= 0.0 for value in self.detail_strength):
            raise ValueError(f"detail_strength must be positive, got {self.detail_strength}.")
        if float(self.sigma_root) <= 0.0:
            raise ValueError(f"sigma_root must be positive, got {self.sigma_root}.")
        if float(self.tanh_gain) <= 0.0:
            raise ValueError(f"tanh_gain must be positive, got {self.tanh_gain}.")
        if float(self.tau_max) <= float(self.tau_min):
            raise ValueError(f"Expected tau_max > tau_min, got {self.tau_max} and {self.tau_min}.")

    @property
    def num_levels(self) -> int:
        return len(self.coarse_to_fine_dims)

    @property
    def num_intervals(self) -> int:
        return self.num_levels - 1

    @property
    def root_dim(self) -> int:
        return int(self.coarse_to_fine_dims[0])

    @property
    def max_current_dim(self) -> int:
        return int(self.coarse_to_fine_dims[-1])

    @property
    def latent_dim(self) -> int:
        return int(self.max_current_dim + self.root_dim)

    def data_zt(self) -> np.ndarray:
        """Stored data-order grid aligned with fine-to-coarse latent trajectories."""
        return (1.0 - np.linspace(float(self.tau_max), float(self.tau_min), self.num_levels, dtype=np.float32)).astype(
            np.float32
        )

    def tau_knots(self) -> np.ndarray:
        return np.linspace(float(self.tau_max), float(self.tau_min), self.num_levels, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["coarse_to_fine_dims"] = [int(value) for value in self.coarse_to_fine_dims]
        data["sigma_levels"] = [float(value) for value in self.sigma_levels]
        data["detail_strength"] = [float(value) for value in self.detail_strength]
        data["sigma_root"] = float(self.sigma_root)
        data["tanh_gain"] = float(self.tanh_gain)
        data["weight_seed"] = int(self.weight_seed)
        data["tau_max"] = float(self.tau_max)
        data["tau_min"] = float(self.tau_min)
        return data

    def fine_to_coarse_dims(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.coarse_to_fine_dims[::-1])

    def data_level_current_dim(self, data_level: int) -> int:
        level_int = int(data_level)
        if level_int < 0 or level_int >= self.num_levels:
            raise ValueError(f"data_level must lie in [0, {self.num_levels - 1}], got {data_level}.")
        return int(self.fine_to_coarse_dims()[level_int])


class HierarchicalGaussianPathProblem:
    """Reusable synthetic coarse-to-fine conditional problem."""

    config: HierarchicalGaussianBenchmarkConfig

    def __init__(self, config: HierarchicalGaussianBenchmarkConfig | None = None) -> None:
        self.config = HierarchicalGaussianBenchmarkConfig() if config is None else config

    @property
    def benchmark_name(self) -> str:
        return HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME

    @property
    def num_levels(self) -> int:
        return int(self.config.num_levels)

    @property
    def num_intervals(self) -> int:
        return int(self.config.num_intervals)

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    def tau_knots(self) -> np.ndarray:
        return self.config.tau_knots()

    def zt(self) -> np.ndarray:
        return self.config.data_zt()

    def fine_to_coarse_dims(self) -> tuple[int, ...]:
        return self.config.fine_to_coarse_dims()

    def interval_key(self, coarse_level: int) -> str:
        return _interval_key(int(coarse_level))

    def extract_root_field(self, states: np.ndarray) -> np.ndarray:
        return _extract_root_field(states, self.config)

    def extract_current_field(self, states: np.ndarray, *, data_level: int) -> np.ndarray:
        return _extract_current_field(states, self.config, data_level=int(data_level))

    def pack_state(self, current_field: np.ndarray, root_field: np.ndarray) -> np.ndarray:
        return _pack_state(current_field, root_field, self.config)

    def interval_mean(self, condition_states: np.ndarray, coarse_level: int) -> np.ndarray:
        return hierarchical_gaussian_interval_mean(condition_states, int(coarse_level), config=self.config)

    def sample_interval(
        self,
        condition_states: np.ndarray,
        coarse_level: int,
        *,
        n_realizations: int,
        seed: int,
    ) -> np.ndarray:
        return sample_hierarchical_gaussian_interval(
            condition_states,
            int(coarse_level),
            n_realizations=int(n_realizations),
            seed=int(seed),
            config=self.config,
        )

    def sample_dataset(self, n_samples: int, *, seed: int) -> np.ndarray:
        return sample_hierarchical_gaussian_dataset(int(n_samples), seed=int(seed), config=self.config)

    def make_splits(
        self,
        *,
        train_samples: int,
        test_samples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        return make_hierarchical_gaussian_benchmark_splits(
            train_samples=int(train_samples),
            test_samples=int(test_samples),
            seed=int(seed),
            config=self.config,
        )

    def sample_rollouts(
        self,
        coarse_states: np.ndarray,
        *,
        n_realizations: int,
        seed: int,
    ) -> np.ndarray:
        return sample_hierarchical_gaussian_rollouts(
            coarse_states,
            n_realizations=int(n_realizations),
            seed=int(seed),
            config=self.config,
        )

    def interval_logpdf(
        self,
        samples: np.ndarray,
        condition_states: np.ndarray,
        coarse_level: int,
    ) -> np.ndarray:
        return hierarchical_gaussian_interval_logpdf(
            samples,
            condition_states,
            int(coarse_level),
            config=self.config,
        )

    def path_logpdf(
        self,
        rollouts: np.ndarray,
        coarse_states: np.ndarray,
    ) -> np.ndarray:
        return hierarchical_gaussian_path_logpdf(rollouts, coarse_states, config=self.config)

    def path_score(
        self,
        rollouts: np.ndarray,
        coarse_states: np.ndarray,
    ) -> np.ndarray:
        return hierarchical_gaussian_path_score(rollouts, coarse_states, config=self.config)

    def metadata(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "config": self.config.to_dict(),
            "tau_knots": self.tau_knots().astype(float).tolist(),
            "zt": self.zt().astype(float).tolist(),
            "data_order": "fine_to_coarse",
            "conditioning_direction": "coarse_to_fine",
            "conditioning_level_index": int(self.num_levels - 1),
            "coarse_to_fine_dims": [int(value) for value in self.config.coarse_to_fine_dims],
            "fine_to_coarse_dims": [int(value) for value in self.fine_to_coarse_dims()],
            "latent_dim": int(self.latent_dim),
        }


def _root_slice(config: HierarchicalGaussianBenchmarkConfig) -> slice:
    return slice(int(config.max_current_dim), int(config.max_current_dim + config.root_dim))


def _extract_root_field(
    states: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig,
) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    return np.asarray(arr[..., _root_slice(config)], dtype=np.float32)


def _extract_current_field(
    states: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig,
    *,
    data_level: int,
) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    current_dim = config.data_level_current_dim(int(data_level))
    if arr.shape[-1] < current_dim:
        raise ValueError(f"states has last dimension {arr.shape[-1]}, expected at least {current_dim}.")
    return np.asarray(arr[..., :current_dim], dtype=np.float32)


def _pack_state(
    current_field: np.ndarray,
    root_field: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig,
) -> np.ndarray:
    field = np.asarray(current_field, dtype=np.float32)
    root = np.asarray(root_field, dtype=np.float32)
    if field.shape[:-1] != root.shape[:-1]:
        raise ValueError(f"current_field and root_field batch shapes must match, got {field.shape} versus {root.shape}.")
    if root.shape[-1] != config.root_dim:
        raise ValueError(f"Expected root_field last dimension {config.root_dim}, got {root.shape}.")
    if field.shape[-1] > config.max_current_dim:
        raise ValueError(f"current_field dimension {field.shape[-1]} exceeds max_current_dim {config.max_current_dim}.")

    packed = np.zeros(field.shape[:-1] + (config.latent_dim,), dtype=np.float32)
    packed[..., : field.shape[-1]] = field
    packed[..., _root_slice(config)] = root
    return packed


def _interval_index_from_coarse_level(
    config: HierarchicalGaussianBenchmarkConfig,
    coarse_level: int,
) -> int:
    coarse_level_int = int(coarse_level)
    if coarse_level_int < 1 or coarse_level_int >= config.num_levels:
        raise ValueError(f"coarse_level must lie in [1, {config.num_levels - 1}], got {coarse_level}.")
    return int(config.num_levels - 1 - coarse_level_int)


def _linear_prolongation_matrix(
    out_dim: int,
    in_dim: int,
) -> np.ndarray:
    out_dim_int = int(out_dim)
    in_dim_int = int(in_dim)
    if out_dim_int <= 0 or in_dim_int <= 0:
        raise ValueError(f"Expected positive dimensions, got {out_dim} and {in_dim}.")
    if out_dim_int == 1 or in_dim_int == 1:
        return np.ones((out_dim_int, in_dim_int), dtype=np.float32)

    x_in = np.linspace(0.0, 1.0, in_dim_int, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, out_dim_int, dtype=np.float64)
    matrix = np.zeros((out_dim_int, in_dim_int), dtype=np.float64)
    for row, x_val in enumerate(x_out):
        idx_hi = int(np.searchsorted(x_in, x_val, side="left"))
        if idx_hi <= 0:
            matrix[row, 0] = 1.0
        elif idx_hi >= in_dim_int:
            matrix[row, -1] = 1.0
        else:
            idx_lo = idx_hi - 1
            left = float(x_in[idx_lo])
            right = float(x_in[idx_hi])
            weight_hi = 0.0 if right <= left else (float(x_val) - left) / (right - left)
            matrix[row, idx_lo] = 1.0 - weight_hi
            matrix[row, idx_hi] = weight_hi
    return np.asarray(matrix, dtype=np.float32)


def _spectral_rescale(
    matrix: np.ndarray,
    target_norm: float,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(arr, compute_uv=False)
    scale = float(np.max(singular_values)) if singular_values.size else 0.0
    if scale <= 1e-12:
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(arr * (float(target_norm) / scale), dtype=np.float32)


@lru_cache(maxsize=8)
def _benchmark_parameters(
    config: HierarchicalGaussianBenchmarkConfig,
) -> dict[str, tuple[np.ndarray, ...]]:
    dims = tuple(int(value) for value in config.coarse_to_fine_dims)
    prolongations = tuple(
        _linear_prolongation_matrix(dims[idx + 1], dims[idx])
        for idx in range(len(dims) - 1)
    )

    direct_prolongations: list[np.ndarray] = []
    direct = None
    for prolongation in prolongations:
        direct = prolongation if direct is None else np.asarray(prolongation @ direct, dtype=np.float32)
        direct_prolongations.append(np.asarray(direct, dtype=np.float32))

    rng = np.random.default_rng(int(config.weight_seed))
    detail_b = []
    detail_c = []
    for interval_idx, target_dim in enumerate(dims[1:]):
        raw_b = rng.normal(0.0, 1.0, size=(target_dim, target_dim)).astype(np.float32)
        raw_c = rng.normal(0.0, 1.0, size=(target_dim, target_dim)).astype(np.float32)
        detail_b.append(_spectral_rescale(raw_b, float(config.detail_strength[interval_idx])))
        detail_c.append(_spectral_rescale(raw_c, float(config.tanh_gain)))

    return {
        "prolongations": tuple(np.asarray(value, dtype=np.float32) for value in prolongations),
        "direct_prolongations": tuple(np.asarray(value, dtype=np.float32) for value in direct_prolongations),
        "detail_b": tuple(np.asarray(value, dtype=np.float32) for value in detail_b),
        "detail_c": tuple(np.asarray(value, dtype=np.float32) for value in detail_c),
    }


def _interval_current_mean_from_parts(
    previous_field: np.ndarray,
    root_field: np.ndarray,
    *,
    interval_idx: int,
    config: HierarchicalGaussianBenchmarkConfig,
) -> np.ndarray:
    params = _benchmark_parameters(config)
    prev = np.asarray(previous_field, dtype=np.float32)
    root = np.asarray(root_field, dtype=np.float32)
    prolongation = params["prolongations"][int(interval_idx)]
    direct = params["direct_prolongations"][int(interval_idx)]
    detail_b = params["detail_b"][int(interval_idx)]
    detail_c = params["detail_c"][int(interval_idx)]
    prolongation_term = np.matmul(prev, prolongation.T)
    root_projection = np.matmul(root, direct.T)
    nonlinear_detail = np.tanh(np.matmul(root_projection, detail_c.T))
    detail_term = np.matmul(nonlinear_detail, detail_b.T)
    return np.asarray(prolongation_term + detail_term, dtype=np.float32)


def hierarchical_gaussian_interval_mean(
    condition_states: np.ndarray,
    coarse_level: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    cond = np.asarray(condition_states, dtype=np.float32)
    if cond.ndim != 2 or cond.shape[1] != cfg.latent_dim:
        raise ValueError(f"condition_states must have shape (n_conditions, {cfg.latent_dim}), got {cond.shape}.")
    interval_idx = _interval_index_from_coarse_level(cfg, coarse_level)
    previous_field = _extract_current_field(cond, cfg, data_level=int(coarse_level))
    root_field = _extract_root_field(cond, cfg)
    target_field = _interval_current_mean_from_parts(
        previous_field,
        root_field,
        interval_idx=interval_idx,
        config=cfg,
    )
    return _pack_state(target_field, root_field, cfg)


def sample_hierarchical_gaussian_interval(
    condition_states: np.ndarray,
    coarse_level: int,
    *,
    n_realizations: int,
    seed: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    cond = np.asarray(condition_states, dtype=np.float32)
    if cond.ndim != 2 or cond.shape[1] != cfg.latent_dim:
        raise ValueError(f"condition_states must have shape (n_conditions, {cfg.latent_dim}), got {cond.shape}.")
    n_draws = int(n_realizations)
    if n_draws <= 0:
        raise ValueError(f"n_realizations must be positive, got {n_realizations}.")

    interval_idx = _interval_index_from_coarse_level(cfg, coarse_level)
    mean_state = hierarchical_gaussian_interval_mean(cond, int(coarse_level), config=cfg)
    mean_field = _extract_current_field(mean_state, cfg, data_level=int(coarse_level) - 1)
    root_field = _extract_root_field(cond, cfg)
    rng = np.random.default_rng(int(seed))
    sigma = float(cfg.sigma_levels[interval_idx])
    noise = rng.normal(
        0.0,
        sigma,
        size=(cond.shape[0], n_draws, mean_field.shape[-1]),
    ).astype(np.float32)
    target_field = mean_field[:, None, :] + noise
    root_rep = np.repeat(root_field[:, None, :], n_draws, axis=1)
    return _pack_state(target_field, root_rep, cfg)


def sample_hierarchical_gaussian_dataset(
    n_samples: int,
    *,
    seed: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    n_samples_int = int(n_samples)
    if n_samples_int <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    rng = np.random.default_rng(int(seed))
    root = rng.normal(0.0, float(cfg.sigma_root), size=(n_samples_int, cfg.root_dim)).astype(np.float32)
    fields = [root]
    for interval_idx in range(cfg.num_intervals):
        mean_field = _interval_current_mean_from_parts(
            fields[-1],
            root,
            interval_idx=interval_idx,
            config=cfg,
        )
        sigma = float(cfg.sigma_levels[interval_idx])
        next_field = mean_field + rng.normal(0.0, sigma, size=mean_field.shape).astype(np.float32)
        fields.append(np.asarray(next_field, dtype=np.float32))

    states_by_plan = [_pack_state(field, root, cfg) for field in fields]
    return np.stack(states_by_plan[::-1], axis=0).astype(np.float32)


def make_hierarchical_gaussian_benchmark_splits(
    *,
    train_samples: int,
    test_samples: int,
    seed: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    latent_train = sample_hierarchical_gaussian_dataset(int(train_samples), seed=int(seed), config=cfg)
    latent_test = sample_hierarchical_gaussian_dataset(int(test_samples), seed=int(seed) + 1, config=cfg)
    tau_knots = cfg.tau_knots()
    zt = cfg.data_zt()
    extras = {
        "benchmark_interval_coarse_levels": np.arange(cfg.num_levels - 1, 0, -1, dtype=np.int64),
        "benchmark_train_samples": np.asarray(int(train_samples), dtype=np.int64),
        "benchmark_test_samples": np.asarray(int(test_samples), dtype=np.int64),
        "benchmark_latent_dim": np.asarray(int(cfg.latent_dim), dtype=np.int64),
        "benchmark_root_dim": np.asarray(int(cfg.root_dim), dtype=np.int64),
        "benchmark_coarse_to_fine_dims": np.asarray(cfg.coarse_to_fine_dims, dtype=np.int64),
        "benchmark_fine_to_coarse_dims": np.asarray(cfg.fine_to_coarse_dims(), dtype=np.int64),
    }
    metadata = {
        "benchmark_name": HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
        "config": cfg.to_dict(),
        "tau_knots": tau_knots.astype(float).tolist(),
        "zt": zt.astype(float).tolist(),
        "data_order": "fine_to_coarse",
        "conditioning_direction": "coarse_to_fine",
        "conditioning_level_index": int(cfg.num_levels - 1),
        "coarse_to_fine_dims": [int(value) for value in cfg.coarse_to_fine_dims],
        "fine_to_coarse_dims": [int(value) for value in cfg.fine_to_coarse_dims()],
        "latent_dim": int(cfg.latent_dim),
    }
    return latent_train, latent_test, zt, extras, metadata


def sample_hierarchical_gaussian_rollouts(
    coarse_states: np.ndarray,
    *,
    n_realizations: int,
    seed: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    coarse = np.asarray(coarse_states, dtype=np.float32)
    if coarse.ndim != 2 or coarse.shape[1] != cfg.latent_dim:
        raise ValueError(f"coarse_states must have shape (n_conditions, {cfg.latent_dim}), got {coarse.shape}.")
    n_draws = int(n_realizations)
    if n_draws <= 0:
        raise ValueError(f"n_realizations must be positive, got {n_realizations}.")

    rng = np.random.default_rng(int(seed))
    root = _extract_root_field(coarse, cfg)
    root_rep = np.repeat(root[:, None, :], n_draws, axis=1)
    current_field = np.repeat(root[:, None, :], n_draws, axis=1).astype(np.float32)
    fields_by_plan = [current_field]

    for interval_idx in range(cfg.num_intervals):
        mean_field = _interval_current_mean_from_parts(
            current_field,
            root_rep,
            interval_idx=interval_idx,
            config=cfg,
        )
        sigma = float(cfg.sigma_levels[interval_idx])
        current_field = mean_field + rng.normal(0.0, sigma, size=mean_field.shape).astype(np.float32)
        fields_by_plan.append(np.asarray(current_field, dtype=np.float32))

    states_by_plan = [_pack_state(field, root_rep, cfg) for field in fields_by_plan]
    return np.stack(states_by_plan[::-1], axis=2).astype(np.float32)


def hierarchical_gaussian_interval_logpdf(
    samples: np.ndarray,
    condition_states: np.ndarray,
    coarse_level: int,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    cond = np.asarray(condition_states, dtype=np.float32)
    sample_arr = np.asarray(samples, dtype=np.float32)
    squeeze = False
    if sample_arr.ndim == 2:
        sample_arr = sample_arr[:, None, :]
        squeeze = True
    if cond.ndim != 2 or cond.shape[1] != cfg.latent_dim:
        raise ValueError(f"condition_states must have shape (n_conditions, {cfg.latent_dim}), got {cond.shape}.")
    if sample_arr.ndim != 3 or sample_arr.shape[0] != cond.shape[0] or sample_arr.shape[-1] != cfg.latent_dim:
        raise ValueError(
            "samples must have shape (n_conditions, n_realizations, latent_dim); "
            f"got {sample_arr.shape} for latent_dim={cfg.latent_dim}."
        )

    interval_idx = _interval_index_from_coarse_level(cfg, coarse_level)
    mean_state = hierarchical_gaussian_interval_mean(cond, int(coarse_level), config=cfg)
    mean_field = _extract_current_field(mean_state, cfg, data_level=int(coarse_level) - 1)
    sample_field = _extract_current_field(sample_arr, cfg, data_level=int(coarse_level) - 1)
    sigma = float(cfg.sigma_levels[interval_idx])
    variance = sigma * sigma
    residual = np.asarray(sample_field - mean_field[:, None, :], dtype=np.float64)
    dim = int(sample_field.shape[-1])
    quad = np.sum(np.square(residual), axis=-1) / max(variance, 1e-12)
    normalizer = dim * np.log(2.0 * np.pi * max(variance, 1e-12))
    logpdf = -0.5 * (normalizer + quad)
    return logpdf[:, 0] if squeeze else logpdf


def hierarchical_gaussian_path_logpdf(
    rollouts: np.ndarray,
    coarse_states: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    coarse = np.asarray(coarse_states, dtype=np.float32)
    rollout_arr = np.asarray(rollouts, dtype=np.float32)
    squeeze = False
    if rollout_arr.ndim == 3:
        rollout_arr = rollout_arr[:, None, :, :]
        squeeze = True
    if coarse.ndim != 2 or coarse.shape[1] != cfg.latent_dim:
        raise ValueError(f"coarse_states must have shape (n_conditions, {cfg.latent_dim}), got {coarse.shape}.")
    if rollout_arr.ndim != 4 or rollout_arr.shape[0] != coarse.shape[0] or rollout_arr.shape[2:] != (
        cfg.num_levels,
        cfg.latent_dim,
    ):
        raise ValueError(
            "rollouts must have shape (n_conditions, n_realizations, num_levels, latent_dim); "
            f"got {rollout_arr.shape} for num_levels={cfg.num_levels}, latent_dim={cfg.latent_dim}."
        )

    root = _extract_root_field(coarse, cfg)
    root_rep = np.repeat(root[:, None, :], rollout_arr.shape[1], axis=1)
    previous_field = np.asarray(root_rep, dtype=np.float32)
    total = np.zeros(rollout_arr.shape[:2], dtype=np.float64)

    for interval_idx in range(cfg.num_intervals):
        target_data_level = cfg.num_levels - 2 - interval_idx
        target_field = _extract_current_field(rollout_arr[:, :, target_data_level, :], cfg, data_level=target_data_level)
        mean_field = _interval_current_mean_from_parts(
            previous_field,
            root_rep,
            interval_idx=interval_idx,
            config=cfg,
        )
        sigma = float(cfg.sigma_levels[interval_idx])
        variance = sigma * sigma
        residual = np.asarray(target_field - mean_field, dtype=np.float64)
        dim = int(target_field.shape[-1])
        quad = np.sum(np.square(residual), axis=-1) / max(variance, 1e-12)
        normalizer = dim * np.log(2.0 * np.pi * max(variance, 1e-12))
        total += -0.5 * (normalizer + quad)
        previous_field = target_field

    return total[:, 0] if squeeze else total


def hierarchical_gaussian_path_score(
    rollouts: np.ndarray,
    coarse_states: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
) -> np.ndarray:
    cfg = HierarchicalGaussianBenchmarkConfig() if config is None else config
    coarse = np.asarray(coarse_states, dtype=np.float32)
    rollout_arr = np.asarray(rollouts, dtype=np.float32)
    squeeze = False
    if rollout_arr.ndim == 3:
        rollout_arr = rollout_arr[:, None, :, :]
        squeeze = True
    if coarse.ndim != 2 or coarse.shape[1] != cfg.latent_dim:
        raise ValueError(f"coarse_states must have shape (n_conditions, {cfg.latent_dim}), got {coarse.shape}.")
    if rollout_arr.ndim != 4 or rollout_arr.shape[2:] != (cfg.num_levels, cfg.latent_dim):
        raise ValueError(
            "rollouts must have shape (n_conditions, n_realizations, num_levels, latent_dim); "
            f"got {rollout_arr.shape}."
        )

    params = _benchmark_parameters(cfg)
    root = _extract_root_field(coarse, cfg)
    root_rep = np.repeat(root[:, None, :], rollout_arr.shape[1], axis=1)
    previous_field = np.asarray(root_rep, dtype=np.float32)
    residuals: list[np.ndarray] = []

    for interval_idx in range(cfg.num_intervals):
        target_data_level = cfg.num_levels - 2 - interval_idx
        target_field = _extract_current_field(rollout_arr[:, :, target_data_level, :], cfg, data_level=target_data_level)
        mean_field = _interval_current_mean_from_parts(
            previous_field,
            root_rep,
            interval_idx=interval_idx,
            config=cfg,
        )
        residuals.append(np.asarray(target_field - mean_field, dtype=np.float32))
        previous_field = target_field

    score = np.zeros_like(rollout_arr, dtype=np.float32)
    for interval_idx in range(cfg.num_intervals - 1, -1, -1):
        plan_scale = interval_idx + 1
        data_level = cfg.num_levels - 1 - plan_scale
        sigma = float(cfg.sigma_levels[interval_idx])
        score_field = -residuals[interval_idx] / max(sigma * sigma, 1e-12)
        if interval_idx < cfg.num_intervals - 1:
            next_sigma = float(cfg.sigma_levels[interval_idx + 1])
            next_prolongation = params["prolongations"][interval_idx + 1]
            score_field += np.matmul(
                residuals[interval_idx + 1] / max(next_sigma * next_sigma, 1e-12),
                next_prolongation,
            )
        score[:, :, data_level, : score_field.shape[-1]] = score_field

    return score[:, 0, :, :] if squeeze else score


@eqx.filter_jit
def _sample_interval_batch(
    drift_net: DriftNet,
    z_start_batch: jax.Array,
    tau_start: jax.Array,
    tau_end: jax.Array,
    dt0: jax.Array,
    keys: jax.Array,
    sigma_fn: Any,
) -> jax.Array:
    return jax.vmap(
        lambda y0, key: integrate_interval(
            drift_net,
            y0,
            tau_start,
            tau_end,
            dt0,
            key,
            sigma_fn,
        )
    )(z_start_batch, keys)


def sample_model_conditionals(
    drift_net: DriftNet,
    condition_states: np.ndarray,
    *,
    tau_start: float,
    tau_end: float,
    dt0: float,
    sigma_fn: SigmaFn,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    conditions = np.asarray(condition_states, dtype=np.float32)
    if conditions.ndim != 2 or conditions.shape[1] != drift_net.latent_dim:
        raise ValueError(
            f"condition_states must have shape (n_conditions, {drift_net.latent_dim}), got {conditions.shape}."
        )

    repeated = np.repeat(conditions, int(n_realizations), axis=0)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), repeated.shape[0])
    generated = _sample_interval_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(float(tau_start), dtype=jnp.float32),
        jnp.asarray(float(tau_end), dtype=jnp.float32),
        jnp.asarray(float(dt0), dtype=jnp.float32),
        keys,
        sigma_fn,
    )
    return np.asarray(generated, dtype=np.float32).reshape(
        conditions.shape[0],
        int(n_realizations),
        conditions.shape[1],
    )


def sample_model_rollouts(
    drift_net: DriftNet,
    coarse_states: np.ndarray,
    *,
    tau_knots: np.ndarray,
    dt0: float,
    sigma_fn: SigmaFn,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    coarse = np.asarray(coarse_states, dtype=np.float32)
    if coarse.ndim != 2 or coarse.shape[1] != drift_net.latent_dim:
        raise ValueError(f"coarse_states must have shape (n_conditions, {drift_net.latent_dim}), got {coarse.shape}.")

    repeated = np.repeat(coarse, int(n_realizations), axis=0)
    generated = sample_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(tau_knots, dtype=jnp.float32),
        sigma_fn,
        float(dt0),
        jax.random.PRNGKey(int(seed)),
    )
    return np.asarray(generated, dtype=np.float32).reshape(
        coarse.shape[0],
        int(n_realizations),
        int(tau_knots.shape[0]),
        coarse.shape[1],
    )


def _resolve_problem(
    config: HierarchicalGaussianBenchmarkConfig | None = None,
    problem: HierarchicalGaussianPathProblem | None = None,
) -> HierarchicalGaussianPathProblem:
    if problem is not None:
        if config is not None and problem.config != config:
            raise ValueError("Provide either problem or config, or ensure they match.")
        return problem
    return HierarchicalGaussianPathProblem(config=config)


def _normalise_weights(
    weights: np.ndarray | None,
    n_samples: int,
) -> np.ndarray:
    if weights is None:
        out = np.ones(int(n_samples), dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64).reshape(-1)
        if out.size != int(n_samples):
            raise ValueError(f"Weight length mismatch: {out.size} vs {n_samples}.")
        out = np.maximum(out, 0.0)
    total = float(np.sum(out))
    if total <= 0.0:
        return np.ones(int(n_samples), dtype=np.float64) / float(n_samples)
    return out / total


def _weighted_projection_quantiles(
    samples: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(samples)
    x_sorted = np.asarray(samples[order], dtype=np.float64)
    w_sorted = _normalise_weights(weights[order], len(order))
    cdf = np.cumsum(w_sorted)
    cdf[-1] = 1.0
    return np.interp(grid, cdf, x_sorted)


def wasserstein1_wasserstein2_latents(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> tuple[float, float]:
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    wa = _normalise_weights(weights_a, len(a))
    wb = _normalise_weights(weights_b, len(b))

    try:
        import ot

        m2 = cdist(a, b, metric="sqeuclidean").astype(np.float64)
        m1 = np.sqrt(np.maximum(m2, 0.0))
        w1 = ot.emd2(wa, wb, m1)
        w2_sq = ot.emd2(wa, wb, m2)
        return float(max(w1, 0.0)), float(np.sqrt(max(w2_sq, 0.0)))

    except ImportError:
        n_proj = 128
        grid = (np.arange(512, dtype=np.float64) + 0.5) / 512.0
        rng = np.random.default_rng(0)
        dim = int(a.shape[1])
        directions = rng.standard_normal((n_proj, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        sw1 = 0.0
        sw2_sq = 0.0
        for direction in directions:
            proj_a = a @ direction
            proj_b = b @ direction
            sw1 += _w1_1d(proj_a, proj_b, u_weights=wa, v_weights=wb)
            qa = _weighted_projection_quantiles(proj_a, wa, grid)
            qb = _weighted_projection_quantiles(proj_b, wb, grid)
            sw2_sq += float(np.mean((qa - qb) ** 2))

        sw1 /= float(n_proj)
        sw2_sq /= float(n_proj)
        return float(max(sw1, 0.0)), float(np.sqrt(max(sw2_sq, 0.0)))


def _standardize_condition_vectors(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    mean = np.mean(z_arr, axis=0, keepdims=True)
    std = np.std(z_arr, axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (z_arr - mean) / std


def _build_directed_knn_indices(
    standardized_conditions: np.ndarray,
    k: int,
) -> np.ndarray:
    z_std = np.asarray(standardized_conditions, dtype=np.float64)
    n = int(z_std.shape[0])
    if n < 2:
        raise ValueError("Need at least two conditions to build a K-NN graph.")

    k_eff = int(max(1, min(k, n - 1)))
    dists = cdist(z_std, z_std, metric="euclidean").astype(np.float64)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    order = np.argsort(nn_dists, axis=1)
    return np.take_along_axis(nn_idx, order, axis=1)


def _rbf_kernel_from_sqdist(
    sqdist: np.ndarray | float,
    bandwidth: float,
) -> np.ndarray | float:
    bw = float(max(bandwidth, 1e-12))
    return np.exp(-np.maximum(sqdist, 0.0) / (bw * bw))


def _select_ecmmd_bandwidth(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
) -> float:
    x = np.asarray(real_samples, dtype=np.float64)
    y = np.asarray(generated_samples, dtype=np.float64)
    if y.ndim != 3 or y.shape[1] < 1:
        raise ValueError(
            "generated_samples must have shape (n_eval, n_realizations, dim) with n_realizations >= 1."
        )
    if x.ndim == 2:
        first_real = x
    elif x.ndim == 3:
        first_real = x[:, 0, :]
    else:
        raise ValueError(f"real_samples must have shape (n_eval, dim) or (n_eval, n_realizations, dim), got {x.shape}.")

    first_draw = y[:, 0, :]
    direct = np.linalg.norm(first_real - first_draw, axis=1)
    positive = direct[np.isfinite(direct) & (direct > 1e-12)]
    if positive.size:
        return float(np.median(positive))

    pooled = np.concatenate([first_real, first_draw], axis=0)
    pooled_dists = cdist(pooled, pooled, metric="euclidean")
    upper = pooled_dists[np.triu_indices_from(pooled_dists, k=1)]
    positive = upper[np.isfinite(upper) & (upper > 1e-12)]
    if positive.size:
        return float(np.median(positive))
    return 1.0


def _studentized_gaussian_test(
    score: float,
    variance_estimate: float,
    n_eval: int,
    k_eff: int,
) -> tuple[float, float, float, float]:
    eta = float(np.sqrt(float(n_eval * k_eff)) * score)
    scale_sq = float(max(variance_estimate, 0.0))
    scale = float(np.sqrt(scale_sq))
    if scale <= 1e-12:
        z_score = 0.0 if abs(eta) <= 1e-12 else float(np.sign(eta) * np.inf)
    else:
        z_score = float(eta / scale)
    p_value = float(2.0 * _norm.sf(abs(z_score))) if np.isfinite(z_score) else 0.0
    return eta, scale, z_score, p_value


def compute_ecmmd_metrics(
    conditions: np.ndarray,
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    k_values: list[int] | tuple[int, ...],
    bandwidth_override: float | None = None,
) -> dict[str, object]:
    z_eval = np.asarray(conditions, dtype=np.float64)
    x_eval = np.asarray(real_samples, dtype=np.float64)
    y_eval = np.asarray(generated_samples, dtype=np.float64)

    if y_eval.ndim != 3:
        raise ValueError(f"generated_samples must have shape (n_eval, n_realizations, dim), got {y_eval.shape}.")

    if x_eval.ndim == 2:
        x_single = x_eval
        x_multi = None
    elif x_eval.ndim == 3:
        x_single = x_eval[:, 0, :]
        x_multi = x_eval
        if x_eval.shape[1] != y_eval.shape[1]:
            raise ValueError("When real_samples is 3-D, it must match generated_samples in n_realizations.")
    else:
        raise ValueError(
            f"real_samples must have shape (n_eval, dim) or (n_eval, n_realizations, dim), got {x_eval.shape}."
        )

    n_eval = int(z_eval.shape[0])
    if x_single.shape[0] != n_eval or y_eval.shape[0] != n_eval:
        raise ValueError("conditions, real_samples, and generated_samples must share the same first dimension.")
    if n_eval < 2:
        return {
            "skipped_reason": "Need at least two evaluation conditions for ECMMD.",
            "n_eval": n_eval,
            "n_realizations": int(y_eval.shape[1]),
            "k_values": {},
        }

    bandwidth = (
        float(bandwidth_override)
        if bandwidth_override is not None
        else _select_ecmmd_bandwidth(x_eval, y_eval)
    )
    standardized_z = _standardize_condition_vectors(z_eval)

    x_sqdist = cdist(x_single, x_single, metric="sqeuclidean").astype(np.float64)
    k_xx = _rbf_kernel_from_sqdist(x_sqdist, bandwidth).astype(np.float64)

    single_h = np.zeros((n_eval, n_eval), dtype=np.float64)
    multi_h = np.zeros((n_eval, n_eval), dtype=np.float64)

    for i in range(n_eval):
        x_i = x_single[i]
        x_i_all = x_multi[i] if x_multi is not None else None
        y_i_single = y_eval[i, 0]
        y_i_all = y_eval[i]
        for j in range(i + 1, n_eval):
            x_j = x_single[j]
            x_j_all = x_multi[j] if x_multi is not None else None
            y_j_single = y_eval[j, 0]
            y_j_all = y_eval[j]
            k_xx_ij = float(k_xx[i, j])

            single_h_ij = (
                k_xx_ij
                + float(_rbf_kernel_from_sqdist(np.sum((y_i_single - y_j_single) ** 2), bandwidth))
                - float(_rbf_kernel_from_sqdist(np.sum((x_i - y_j_single) ** 2), bandwidth))
                - float(_rbf_kernel_from_sqdist(np.sum((x_j - y_i_single) ** 2), bandwidth))
            )
            single_h[i, j] = single_h_ij
            single_h[j, i] = single_h_ij

            if x_multi is None:
                d_yy = np.sum((y_i_all - y_j_all) ** 2, axis=1)
                d_xy = np.sum((x_i[None, :] - y_j_all) ** 2, axis=1)
                d_yx = np.sum((x_j[None, :] - y_i_all) ** 2, axis=1)
                multi_h_ij = k_xx_ij + float(
                    np.mean(
                        _rbf_kernel_from_sqdist(d_yy, bandwidth)
                        - _rbf_kernel_from_sqdist(d_xy, bandwidth)
                        - _rbf_kernel_from_sqdist(d_yx, bandwidth)
                    )
                )
            else:
                d_xx = np.sum((x_i_all - x_j_all) ** 2, axis=1)
                d_yy = np.sum((y_i_all - y_j_all) ** 2, axis=1)
                d_xy = np.sum((x_i_all - y_j_all) ** 2, axis=1)
                d_yx = np.sum((x_j_all - y_i_all) ** 2, axis=1)
                multi_h_ij = float(
                    np.mean(
                        _rbf_kernel_from_sqdist(d_xx, bandwidth)
                        + _rbf_kernel_from_sqdist(d_yy, bandwidth)
                        - _rbf_kernel_from_sqdist(d_xy, bandwidth)
                        - _rbf_kernel_from_sqdist(d_yx, bandwidth)
                    )
                )
            multi_h[i, j] = multi_h_ij
            multi_h[j, i] = multi_h_ij

    k_results: dict[str, object] = {}
    for requested_k in sorted(set(int(k) for k in k_values if int(k) > 0)):
        k_eff = int(max(1, min(requested_k, n_eval - 1)))
        nn_idx = _build_directed_knn_indices(standardized_z, k_eff)
        rows = np.repeat(np.arange(n_eval, dtype=np.int64), k_eff)
        cols = nn_idx.reshape(-1)

        adjacency = np.zeros((n_eval, n_eval), dtype=bool)
        adjacency[rows, cols] = True
        mutual = adjacency & adjacency.T
        edge_weight = 1.0 + mutual[rows, cols].astype(np.float64)

        single_edges = single_h[rows, cols]
        multi_edges = multi_h[rows, cols]

        single_score = float(np.mean(single_edges))
        single_var = float(np.mean(np.square(single_edges) * edge_weight))
        single_eta, single_scale, single_z, single_p = _studentized_gaussian_test(
            single_score,
            single_var,
            n_eval=n_eval,
            k_eff=k_eff,
        )

        multi_score = float(np.mean(multi_edges))
        multi_var = float(np.mean(np.square(multi_edges) * edge_weight))
        multi_eta, multi_scale, multi_z, multi_p = _studentized_gaussian_test(
            multi_score,
            multi_var,
            n_eval=n_eval,
            k_eff=k_eff,
        )

        k_results[str(requested_k)] = {
            "k_requested": int(requested_k),
            "k_effective": int(k_eff),
            "single_draw": {
                "score": single_score,
                "eta": single_eta,
                "scale": single_scale,
                "z_score": single_z,
                "p_value": single_p,
            },
            "derandomized": {
                "score": multi_score,
                "eta": multi_eta,
                "scale": multi_scale,
                "z_score": multi_z,
                "p_value": multi_p,
            },
        }

    return {
        "bandwidth": float(bandwidth),
        "n_eval": int(n_eval),
        "n_realizations": int(y_eval.shape[1]),
        "k_values": k_results,
    }


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_stability(samples: np.ndarray, norm_threshold: float) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, np.asarray(samples).shape[-1])
    finite_mask = np.all(np.isfinite(arr), axis=1)
    safe_arr = np.where(np.isfinite(arr), arr, 0.0)
    norms = np.linalg.norm(safe_arr, axis=1)
    unstable_mask = (~finite_mask) | (norms > float(norm_threshold))
    finite_norms = norms[finite_mask]
    return {
        "norm_threshold": float(norm_threshold),
        "finite_fraction": float(np.mean(finite_mask)),
        "unstable_fraction": float(np.mean(unstable_mask)),
        "max_norm_finite": float(np.max(finite_norms)) if finite_norms.size else float("nan"),
        "mean_norm_finite": float(np.mean(finite_norms)) if finite_norms.size else float("nan"),
    }


def _evaluate_conditional_clouds(
    conditions: np.ndarray,
    oracle_samples: np.ndarray,
    generated_samples: np.ndarray,
    *,
    ecmmd_k_values: list[int],
    norm_threshold: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    w1_values = []
    w2_values = []
    for idx in range(int(np.asarray(conditions).shape[0])):
        w1, w2 = wasserstein1_wasserstein2_latents(generated_samples[idx], oracle_samples[idx])
        w1_values.append(w1)
        w2_values.append(w2)
    w1_arr = np.asarray(w1_values, dtype=np.float64)
    w2_arr = np.asarray(w2_values, dtype=np.float64)
    return (
        {
            "w1": _metric_summary(w1_arr),
            "w2": _metric_summary(w2_arr),
            "ecmmd": compute_ecmmd_metrics(conditions, oracle_samples, generated_samples, ecmmd_k_values),
            "stability": _summarize_stability(generated_samples, norm_threshold),
        },
        w1_arr,
        w2_arr,
    )


def _summarize_root_copy_error(
    samples: np.ndarray,
    reference_roots: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig,
) -> dict[str, float]:
    root_samples = _extract_root_field(samples, config)
    root_ref = np.asarray(reference_roots, dtype=np.float32)
    batch_shape = root_samples.shape[:-1]
    expand_shape = (root_ref.shape[0],) + (1,) * (len(batch_shape) - 1) + (root_ref.shape[-1],)
    residual = np.asarray(root_samples - root_ref.reshape(expand_shape), dtype=np.float64)
    norms = np.linalg.norm(residual, axis=-1)
    return _metric_summary(norms)


def _interval_key(coarse_level: int) -> str:
    return f"x{int(coarse_level)}_to_x{int(coarse_level) - 1}"


def _select_plot_indices(
    states: np.ndarray,
    n_select: int,
    *,
    data_level: int,
    config: HierarchicalGaussianBenchmarkConfig,
) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    count = min(int(n_select), arr.shape[0])
    del data_level, config
    return np.arange(count, dtype=np.int64)


def _field_grid(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(length), dtype=np.float32)


def _sample_line_indices(n_samples: int, max_lines: int = 64) -> np.ndarray:
    if n_samples <= max_lines:
        return np.arange(n_samples, dtype=np.int64)
    return np.linspace(0, n_samples - 1, max_lines, dtype=np.int64)


def _gaussian_mixture_pdf(
    x_grid: np.ndarray,
    component_means: np.ndarray,
    sigma: float,
) -> np.ndarray:
    grid = np.asarray(x_grid, dtype=np.float64).reshape(-1, 1)
    means = np.asarray(component_means, dtype=np.float64).reshape(1, -1)
    scale = float(max(sigma, 1e-12))
    return np.mean(_norm.pdf((grid - means) / scale) / scale, axis=1)


def _scalar_density_limits(
    component_means: np.ndarray,
    sigma: float,
    *sample_sets: np.ndarray,
) -> tuple[float, float]:
    means = np.asarray(component_means, dtype=np.float64).reshape(-1)
    lower = float(np.min(means) - 4.0 * float(sigma))
    upper = float(np.max(means) + 4.0 * float(sigma))
    for values in sample_sets:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lower = min(lower, float(np.min(finite)))
            upper = max(upper, float(np.max(finite)))
    span = upper - lower
    if not np.isfinite(span) or span <= 1e-8:
        center = 0.5 * (lower + upper) if np.isfinite(lower + upper) else 0.0
        span = max(8.0 * float(sigma), 1.0)
        lower = center - 0.5 * span
        upper = center + 0.5 * span
    pad = 0.05 * span
    return float(lower - pad), float(upper + pad)


def _density_histogram_edges(
    x_min: float,
    x_max: float,
    *sample_sets: np.ndarray,
) -> np.ndarray:
    values = [
        np.asarray(arr, dtype=np.float64).reshape(-1)
        for arr in sample_sets
        if np.asarray(arr).size
    ]
    if not values:
        return np.linspace(float(x_min), float(x_max), 25, dtype=np.float64)
    pooled = np.concatenate(values, axis=0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 2:
        return np.linspace(float(x_min), float(x_max), 25, dtype=np.float64)

    edges = np.histogram_bin_edges(pooled, bins="fd", range=(float(x_min), float(x_max)))
    n_bins = max(int(edges.size) - 1, 1)
    if n_bins < 16:
        return np.linspace(float(x_min), float(x_max), 17, dtype=np.float64)
    if n_bins > 48:
        return np.linspace(float(x_min), float(x_max), 49, dtype=np.float64)
    return np.asarray(edges, dtype=np.float64)


def plot_conditioned_field_profiles(
    *,
    condition_states: np.ndarray,
    oracle_samples: np.ndarray,
    generated_samples: np.ndarray,
    coarse_level: int,
    config: HierarchicalGaussianBenchmarkConfig,
    output_stem: Path,
    title: str,
) -> dict[str, str]:
    format_for_paper()
    conditions = np.asarray(condition_states, dtype=np.float32)
    oracle = np.asarray(oracle_samples, dtype=np.float32)
    generated = np.asarray(generated_samples, dtype=np.float32)
    if oracle.shape != generated.shape:
        raise ValueError(f"oracle and generated samples must match, got {oracle.shape} versus {generated.shape}.")

    target_level = int(coarse_level) - 1
    reference_mean = hierarchical_gaussian_interval_mean(conditions, coarse_level, config=config)
    mean_fields = _extract_current_field(reference_mean, config, data_level=target_level)
    interval_idx = _interval_index_from_coarse_level(config, coarse_level)
    sigma = float(config.sigma_levels[interval_idx])
    n_rows = conditions.shape[0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.1, max(2.8, 2.35 * n_rows)), squeeze=False)

    for row in range(n_rows):
        condition_norm = float(np.linalg.norm(_extract_current_field(conditions[row], config, data_level=int(coarse_level))))
        component_means = np.asarray(mean_fields[row], dtype=np.float64).reshape(-1)
        oracle_values = np.asarray(
            _extract_current_field(oracle[row], config, data_level=target_level),
            dtype=np.float64,
        ).reshape(-1)
        generated_values = np.asarray(
            _extract_current_field(generated[row], config, data_level=target_level),
            dtype=np.float64,
        ).reshape(-1)
        x_min, x_max = _scalar_density_limits(component_means, sigma, oracle_values, generated_values)
        x_grid = np.linspace(x_min, x_max, 512, dtype=np.float64)
        true_density = _gaussian_mixture_pdf(x_grid, component_means, sigma)
        bin_edges = _density_histogram_edges(x_min, x_max, oracle_values, generated_values)
        row_ymax = float(np.max(true_density))

        for col, (label, samples, color) in enumerate((("Oracle", oracle, C_OBS), ("CSP", generated, C_GEN))):
            ax = axes[row, col]
            sample_values = np.asarray(
                _extract_current_field(samples[row], config, data_level=target_level),
                dtype=np.float64,
            ).reshape(-1)
            hist_density, _ = np.histogram(sample_values, bins=bin_edges, density=True)
            if hist_density.size:
                row_ymax = max(row_ymax, float(np.max(hist_density)))
            ax.hist(
                sample_values,
                bins=bin_edges,
                density=True,
                color=color,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.45,
                label=f"{label} samples",
            )
            ax.plot(
                x_grid,
                true_density,
                color=C_MEAN,
                linewidth=1.8,
                linestyle="-",
                label="True conditional density",
            )
            if row == 0:
                ax.set_title(f"{label} samples", fontsize=FONT_TITLE)
            ax.text(
                0.02,
                0.98,
                f"cond {row + 1}\n||state||={condition_norm:.3f}\nσ={sigma:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FONT_TICK,
                color=C_TEXT,
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.0},
            )
            ax.grid(alpha=0.20, color=C_GRID, linewidth=0.6)
            ax.set_xlabel(f"X{target_level} component value", fontsize=FONT_LABEL)
            ax.set_ylabel("Coordinate density", fontsize=FONT_LABEL)
            ax.tick_params(axis="both", labelsize=FONT_TICK)
            ax.set_xlim(x_min, x_max)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0 and col == 1:
                ax.legend(loc="upper right", frameon=False, fontsize=FONT_TICK)

        for col in range(2):
            axes[row, col].set_ylim(0.0, row_ymax * 1.12 if row_ymax > 0.0 else 1.0)

    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_rollout_field_profiles(
    *,
    coarse_states: np.ndarray,
    oracle_rollouts: np.ndarray,
    generated_rollouts: np.ndarray,
    config: HierarchicalGaussianBenchmarkConfig,
    output_stem: Path,
    title: str,
) -> dict[str, str]:
    format_for_paper()
    coarse = np.asarray(coarse_states, dtype=np.float32)
    oracle = np.asarray(oracle_rollouts, dtype=np.float32)
    generated = np.asarray(generated_rollouts, dtype=np.float32)
    if oracle.shape != generated.shape:
        raise ValueError(f"oracle and generated rollouts must match, got {oracle.shape} versus {generated.shape}.")

    n_rows = coarse.shape[0]
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.0, max(2.4, 2.2 * n_rows)), squeeze=False)
    level_colors = plt.cm.cividis(np.linspace(0.18, 0.88, config.num_levels))

    for row in range(n_rows):
        coarse_norm = float(np.linalg.norm(_extract_root_field(coarse[row], config)))
        for col, (label, rollouts) in enumerate((("Oracle", oracle), ("CSP", generated))):
            ax = axes[row, col]
            for data_level in range(config.num_levels):
                mean_field = np.mean(
                    _extract_current_field(rollouts[row, :, data_level, :], config, data_level=data_level),
                    axis=0,
                )
                x_grid = _field_grid(mean_field.shape[-1])
                ax.plot(
                    x_grid,
                    mean_field,
                    linewidth=1.3,
                    color=level_colors[data_level],
                    label=f"X{data_level}",
                )
            if row == 0:
                ax.set_title(label, fontsize=FONT_TITLE)
            ax.text(
                0.02,
                0.98,
                f"seed {row + 1}\n||X{config.num_levels - 1}||={coarse_norm:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FONT_TICK,
                color=C_TEXT,
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.0},
            )
            ax.grid(alpha=0.20, color=C_GRID, linewidth=0.6)
            ax.set_xlabel("Normalized component index", fontsize=FONT_LABEL)
            ax.set_ylabel("Mean component value", fontsize=FONT_LABEL)
            ax.tick_params(axis="both", labelsize=FONT_TICK)

    handles = [
        plt.Line2D([0], [0], color=level_colors[data_level], lw=1.3, label=f"X{data_level}")
        for data_level in range(config.num_levels)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=config.num_levels, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.03)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _flatten_prefix_current_fields(
    rollouts: np.ndarray,
    *,
    start_level: int,
    config: HierarchicalGaussianBenchmarkConfig,
) -> np.ndarray:
    pieces = [
        _extract_current_field(rollouts[:, :, level, :], config, data_level=level)
        for level in range(int(start_level), config.num_levels)
    ]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def build_hierarchical_gaussian_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Hierarchical Gaussian Path Benchmark",
        "=" * 40,
        f"benchmark_name: {summary['benchmark_name']}",
        f"n_eval_conditions: {summary['n_eval_conditions']}",
        f"n_realizations: {summary['n_realizations']}",
        f"plot_samples: {summary['plot_samples']}",
        f"tau_knots: {summary['tau_knots']}",
        f"coarse_to_fine_dims: {summary['benchmark_config']['coarse_to_fine_dims']}",
        "",
        "Teacher-forced one-step",
    ]

    for key in summary["interval_keys"]:
        metrics = summary["teacher_forced"][key]
        lines.append(
            f"  {key}: W1={metrics['w1']['mean']:.4f} +/- {metrics['w1']['std']:.4f}, "
            f"W2={metrics['w2']['mean']:.4f} +/- {metrics['w2']['std']:.4f}, "
            f"logpdf(gen)={metrics['generated_logpdf']['mean']:.4f}, "
            f"root-copy={metrics['root_copy_error']['mean']:.4e}"
        )
        for k_key, k_metrics in metrics["ecmmd"].get("k_values", {}).items():
            lines.append(
                f"    ECMMD K={k_key}: single={k_metrics['single_draw']['score']:.4e}, "
                f"D_n={k_metrics['derandomized']['score']:.4e}"
            )

    path_logpdf = summary["free_rollout"]["path_logpdf"]
    lines.append("")
    lines.append("Free-rollout path log density")
    lines.append(
        f"  oracle={path_logpdf['oracle']['mean']:.4f} +/- {path_logpdf['oracle']['std']:.4f}, "
        f"generated={path_logpdf['generated']['mean']:.4f} +/- {path_logpdf['generated']['std']:.4f}, "
        f"gap={path_logpdf['generated_minus_oracle_mean']:.4f}"
    )

    lines.append("")
    lines.append("Free-rollout marginals")
    for key, metrics in summary["free_rollout"]["marginals"].items():
        lines.append(
            f"  {key}: W1={metrics['w1']['mean']:.4f} +/- {metrics['w1']['std']:.4f}, "
            f"W2={metrics['w2']['mean']:.4f} +/- {metrics['w2']['std']:.4f}, "
            f"unstable={metrics['stability']['unstable_fraction']:.3f}"
        )

    lines.append("")
    lines.append("Free-rollout prefixes")
    for key, metrics in summary["free_rollout"]["prefixes"].items():
        lines.append(
            f"  {key}: W1={metrics['w1']['mean']:.4f} +/- {metrics['w1']['std']:.4f}, "
            f"W2={metrics['w2']['mean']:.4f} +/- {metrics['w2']['std']:.4f}"
        )

    lines.append("")
    lines.append("Figures")
    for figure_key, figure_path in summary["figures"].items():
        lines.append(f"  {figure_key}: {figure_path}")

    return "\n".join(lines)


def evaluate_hierarchical_gaussian_sampler_benchmark(
    *,
    sample_conditionals_fn: Callable[[np.ndarray, int, int, int], np.ndarray],
    sample_rollouts_fn: Callable[[np.ndarray, int, int], np.ndarray],
    latent_test: np.ndarray,
    tau_knots: np.ndarray,
    output_dir: str | Path,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
    problem: HierarchicalGaussianPathProblem | None = None,
    ecmmd_k_values: list[int] | tuple[int, ...] = (10, 20, 30),
    n_eval_conditions: int = 32,
    n_realizations: int = 128,
    plot_samples: int = 256,
    n_plot_conditions: int = 4,
    norm_threshold: float = 10.0,
    seed: int = 0,
) -> dict[str, Any]:
    problem_obj = _resolve_problem(config=config, problem=problem)
    cfg = problem_obj.config
    latent_test_np = np.asarray(latent_test, dtype=np.float32)
    tau_knots_np = np.asarray(tau_knots, dtype=np.float32).reshape(-1)
    if latent_test_np.shape != (cfg.num_levels, latent_test_np.shape[1], cfg.latent_dim):
        raise ValueError(
            "latent_test must have shape (num_levels, N, latent_dim); "
            f"expected ({cfg.num_levels}, N, {cfg.latent_dim}), got {latent_test_np.shape}."
        )
    if tau_knots_np.shape[0] != cfg.num_levels:
        raise ValueError(f"Expected {cfg.num_levels} tau knots, got {tau_knots_np.shape[0]}.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    n_test = int(latent_test_np.shape[1])
    eval_count = min(int(n_eval_conditions), n_test)
    condition_indices = np.sort(rng.choice(n_test, size=eval_count, replace=False)).astype(np.int64)
    seed_indices = np.sort(rng.choice(n_test, size=eval_count, replace=False)).astype(np.int64)

    interval_keys = [problem_obj.interval_key(coarse_level) for coarse_level in range(cfg.num_levels - 1, 0, -1)]
    teacher_forced: dict[str, dict[str, object]] = {}
    npz_payload: dict[str, Any] = {
        "tau_knots": tau_knots_np.astype(np.float32),
        "zt": (1.0 - tau_knots_np).astype(np.float32),
        "teacher_forced_condition_indices": condition_indices,
        "free_rollout_seed_indices": seed_indices,
        "benchmark_coarse_to_fine_dims": np.asarray(cfg.coarse_to_fine_dims, dtype=np.int64),
    }

    for coarse_level in range(cfg.num_levels - 1, 0, -1):
        key = problem_obj.interval_key(coarse_level)
        conditions = latent_test_np[coarse_level, condition_indices]
        oracle = problem_obj.sample_interval(
            conditions,
            coarse_level,
            n_realizations=int(n_realizations),
            seed=int(seed) + 10_000 * int(coarse_level),
        )
        generated = sample_conditionals_fn(
            conditions,
            int(coarse_level),
            int(n_realizations),
            int(seed) + 20_000 * int(coarse_level),
        )
        oracle_eval = problem_obj.extract_current_field(oracle, data_level=int(coarse_level) - 1)
        generated_eval = problem_obj.extract_current_field(generated, data_level=int(coarse_level) - 1)
        metrics, w1_arr, w2_arr = _evaluate_conditional_clouds(
            conditions,
            oracle_eval,
            generated_eval,
            ecmmd_k_values=list(int(k) for k in ecmmd_k_values),
            norm_threshold=float(norm_threshold),
        )
        oracle_logpdf = problem_obj.interval_logpdf(oracle, conditions, coarse_level)
        generated_logpdf = problem_obj.interval_logpdf(generated, conditions, coarse_level)
        root_ref = problem_obj.extract_root_field(conditions)
        metrics["oracle_logpdf"] = _metric_summary(oracle_logpdf)
        metrics["generated_logpdf"] = _metric_summary(generated_logpdf)
        metrics["generated_minus_oracle_logpdf_mean"] = float(np.mean(generated_logpdf) - np.mean(oracle_logpdf))
        metrics["root_copy_error"] = _summarize_root_copy_error(generated, root_ref, cfg)
        teacher_forced[key] = metrics
        npz_payload[f"teacher_forced_{key}_conditions"] = conditions.astype(np.float32)
        npz_payload[f"teacher_forced_{key}_oracle"] = oracle.astype(np.float32)
        npz_payload[f"teacher_forced_{key}_generated"] = generated.astype(np.float32)
        npz_payload[f"teacher_forced_{key}_w1_values"] = w1_arr.astype(np.float32)
        npz_payload[f"teacher_forced_{key}_w2_values"] = w2_arr.astype(np.float32)
        npz_payload[f"teacher_forced_{key}_oracle_logpdf"] = np.asarray(oracle_logpdf, dtype=np.float32)
        npz_payload[f"teacher_forced_{key}_generated_logpdf"] = np.asarray(generated_logpdf, dtype=np.float32)

    coarse_conditions = latent_test_np[-1, seed_indices]
    oracle_rollouts = problem_obj.sample_rollouts(
        coarse_conditions,
        n_realizations=int(n_realizations),
        seed=int(seed) + 100_000,
    )
    generated_rollouts = sample_rollouts_fn(
        coarse_conditions,
        int(n_realizations),
        int(seed) + 110_000,
    )

    oracle_path_logpdf = problem_obj.path_logpdf(oracle_rollouts, coarse_conditions)
    generated_path_logpdf = problem_obj.path_logpdf(generated_rollouts, coarse_conditions)
    free_rollout: dict[str, Any] = {
        "marginals": {},
        "prefixes": {},
        "stability": _summarize_stability(generated_rollouts[:, :, :, : cfg.max_current_dim], float(norm_threshold)),
        "root_copy_error": _summarize_root_copy_error(
            generated_rollouts,
            problem_obj.extract_root_field(coarse_conditions),
            cfg,
        ),
        "path_logpdf": {
            "oracle": _metric_summary(oracle_path_logpdf),
            "generated": _metric_summary(generated_path_logpdf),
            "generated_minus_oracle_mean": float(np.mean(generated_path_logpdf) - np.mean(oracle_path_logpdf)),
        },
    }
    npz_payload["free_rollout_coarse_conditions"] = coarse_conditions.astype(np.float32)
    npz_payload["free_rollout_oracle_rollouts"] = oracle_rollouts.astype(np.float32)
    npz_payload["free_rollout_generated_rollouts"] = generated_rollouts.astype(np.float32)
    npz_payload["free_rollout_oracle_path_logpdf"] = np.asarray(oracle_path_logpdf, dtype=np.float32)
    npz_payload["free_rollout_generated_path_logpdf"] = np.asarray(generated_path_logpdf, dtype=np.float32)

    coarsest_level = cfg.num_levels - 1
    for level_idx in range(cfg.num_levels - 1):
        marginal_key = f"x{coarsest_level}_to_x{int(level_idx)}"
        oracle_marginal = problem_obj.extract_current_field(oracle_rollouts[:, :, level_idx, :], data_level=level_idx)
        generated_marginal = problem_obj.extract_current_field(
            generated_rollouts[:, :, level_idx, :],
            data_level=level_idx,
        )
        marginal_metrics, marginal_w1, marginal_w2 = _evaluate_conditional_clouds(
            coarse_conditions,
            oracle_marginal,
            generated_marginal,
            ecmmd_k_values=list(int(k) for k in ecmmd_k_values),
            norm_threshold=float(norm_threshold),
        )
        free_rollout["marginals"][marginal_key] = marginal_metrics
        npz_payload[f"free_rollout_{marginal_key}_w1_values"] = marginal_w1.astype(np.float32)
        npz_payload[f"free_rollout_{marginal_key}_w2_values"] = marginal_w2.astype(np.float32)

        prefix_key = f"x{coarsest_level}_to_x{int(level_idx)}_prefix"
        oracle_prefix = _flatten_prefix_current_fields(oracle_rollouts, start_level=level_idx, config=cfg)
        generated_prefix = _flatten_prefix_current_fields(generated_rollouts, start_level=level_idx, config=cfg)
        prefix_metrics, prefix_w1, prefix_w2 = _evaluate_conditional_clouds(
            coarse_conditions,
            oracle_prefix,
            generated_prefix,
            ecmmd_k_values=list(int(k) for k in ecmmd_k_values),
            norm_threshold=float(norm_threshold),
        )
        free_rollout["prefixes"][prefix_key] = prefix_metrics
        npz_payload[f"free_rollout_{prefix_key}_w1_values"] = prefix_w1.astype(np.float32)
        npz_payload[f"free_rollout_{prefix_key}_w2_values"] = prefix_w2.astype(np.float32)

    finest_interval_coarse_level = 1
    teacher_figures_by_interval: dict[str, dict[str, str]] = {}
    for coarse_level in range(cfg.num_levels - 1, 0, -1):
        interval_key = problem_obj.interval_key(coarse_level)
        teacher_plot_indices = _select_plot_indices(
            latent_test_np[coarse_level],
            int(n_plot_conditions),
            data_level=coarse_level,
            config=cfg,
        )
        teacher_plot_conditions = latent_test_np[coarse_level, teacher_plot_indices]
        teacher_plot_oracle = problem_obj.sample_interval(
            teacher_plot_conditions,
            coarse_level,
            n_realizations=int(plot_samples),
            seed=int(seed) + 200_000 + 10_000 * int(coarse_level),
        )
        teacher_plot_generated = sample_conditionals_fn(
            teacher_plot_conditions,
            coarse_level,
            int(plot_samples),
            int(seed) + 210_000 + 10_000 * int(coarse_level),
        )
        teacher_figures = plot_conditioned_field_profiles(
            condition_states=teacher_plot_conditions,
            oracle_samples=teacher_plot_oracle,
            generated_samples=teacher_plot_generated,
            coarse_level=coarse_level,
            config=cfg,
            output_stem=output_path / f"fig_teacher_forced_{interval_key}_field_profiles",
            title=f"Teacher-forced conditional coordinate densities (pooled): {interval_key}",
        )
        teacher_figures_by_interval[interval_key] = teacher_figures
        npz_payload[f"teacher_plot_{interval_key}_indices"] = teacher_plot_indices.astype(np.int64)
        npz_payload[f"teacher_plot_{interval_key}_conditions"] = teacher_plot_conditions.astype(np.float32)
        npz_payload[f"teacher_plot_{interval_key}_oracle"] = teacher_plot_oracle.astype(np.float32)
        npz_payload[f"teacher_plot_{interval_key}_generated"] = teacher_plot_generated.astype(np.float32)

        if coarse_level == finest_interval_coarse_level:
            npz_payload["teacher_plot_indices"] = teacher_plot_indices.astype(np.int64)
            npz_payload["teacher_plot_conditions"] = teacher_plot_conditions.astype(np.float32)
            npz_payload["teacher_plot_oracle"] = teacher_plot_oracle.astype(np.float32)
            npz_payload["teacher_plot_generated"] = teacher_plot_generated.astype(np.float32)

    rollout_plot_indices = _select_plot_indices(
        latent_test_np[-1],
        int(n_plot_conditions),
        data_level=cfg.num_levels - 1,
        config=cfg,
    )
    rollout_plot_conditions = latent_test_np[-1, rollout_plot_indices]
    rollout_plot_oracle = problem_obj.sample_rollouts(
        rollout_plot_conditions,
        n_realizations=int(plot_samples),
        seed=int(seed) + 220_000,
    )
    rollout_plot_generated = sample_rollouts_fn(
        rollout_plot_conditions,
        int(plot_samples),
        int(seed) + 230_000,
    )
    rollout_figures = plot_rollout_field_profiles(
        coarse_states=rollout_plot_conditions,
        oracle_rollouts=rollout_plot_oracle,
        generated_rollouts=rollout_plot_generated,
        config=cfg,
        output_stem=output_path / f"fig_free_rollout_x{coarsest_level}_to_x0_field_profiles",
        title=f"Free rollout from X{coarsest_level} seed",
    )
    npz_payload["rollout_plot_indices"] = rollout_plot_indices.astype(np.int64)
    npz_payload["rollout_plot_conditions"] = rollout_plot_conditions.astype(np.float32)
    npz_payload["rollout_plot_oracle"] = rollout_plot_oracle.astype(np.float32)
    npz_payload["rollout_plot_generated"] = rollout_plot_generated.astype(np.float32)

    summary: dict[str, Any] = {
        "benchmark_name": HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
        "benchmark_config": cfg.to_dict(),
        "interval_keys": interval_keys,
        "n_eval_conditions": int(eval_count),
        "n_realizations": int(n_realizations),
        "plot_samples": int(plot_samples),
        "tau_knots": tau_knots_np.astype(float).tolist(),
        "zt": (1.0 - tau_knots_np).astype(float).tolist(),
        "teacher_forced_condition_indices": condition_indices.astype(int).tolist(),
        "free_rollout_seed_indices": seed_indices.astype(int).tolist(),
        "teacher_forced": teacher_forced,
        "free_rollout": free_rollout,
        "figures": {
            **{
                f"teacher_forced_{interval_key}_{ext}": path
                for interval_key, figure_paths in teacher_figures_by_interval.items()
                for ext, path in figure_paths.items()
            },
            f"free_rollout_x{coarsest_level}_to_x0_png": rollout_figures["png"],
            f"free_rollout_x{coarsest_level}_to_x0_pdf": rollout_figures["pdf"],
        },
    }

    np.savez_compressed(output_path / "benchmark_metrics.npz", **npz_payload)
    summary_text = build_hierarchical_gaussian_summary_text(summary)
    (output_path / "benchmark_summary.txt").write_text(summary_text, encoding="utf-8")
    (output_path / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def evaluate_hierarchical_gaussian_benchmark(
    *,
    drift_net: DriftNet,
    latent_test: np.ndarray,
    tau_knots: np.ndarray,
    sigma_fn: SigmaFn,
    dt0: float,
    output_dir: str | Path,
    config: HierarchicalGaussianBenchmarkConfig | None = None,
    problem: HierarchicalGaussianPathProblem | None = None,
    ecmmd_k_values: list[int] | tuple[int, ...] = (10, 20, 30),
    n_eval_conditions: int = 32,
    n_realizations: int = 128,
    plot_samples: int = 256,
    n_plot_conditions: int = 4,
    norm_threshold: float = 10.0,
    seed: int = 0,
) -> dict[str, Any]:
    return evaluate_hierarchical_gaussian_sampler_benchmark(
        sample_conditionals_fn=lambda conditions, coarse_level, n_realizations_value, seed_value: sample_model_conditionals(
            drift_net,
            conditions,
            tau_start=float(np.asarray(tau_knots, dtype=np.float32).reshape(-1)[int(coarse_level)]),
            tau_end=float(np.asarray(tau_knots, dtype=np.float32).reshape(-1)[int(coarse_level - 1)]),
            dt0=float(dt0),
            sigma_fn=sigma_fn,
            n_realizations=int(n_realizations_value),
            seed=int(seed_value),
        ),
        sample_rollouts_fn=lambda coarse_states, n_realizations_value, seed_value: sample_model_rollouts(
            drift_net,
            coarse_states,
            tau_knots=np.asarray(tau_knots, dtype=np.float32).reshape(-1),
            dt0=float(dt0),
            sigma_fn=sigma_fn,
            n_realizations=int(n_realizations_value),
            seed=int(seed_value),
        ),
        latent_test=latent_test,
        tau_knots=tau_knots,
        output_dir=output_dir,
        config=config,
        problem=problem,
        ecmmd_k_values=ecmmd_k_values,
        n_eval_conditions=n_eval_conditions,
        n_realizations=n_realizations,
        plot_samples=plot_samples,
        n_plot_conditions=n_plot_conditions,
        norm_threshold=norm_threshold,
        seed=seed,
    )
