"""Notebook-style helper to prototype decoding/lifting strategies.

Loads trained MMSFM models, cached time-coupled diffusion map lifters, and PCA
statistics so we can focus on the decoding bottleneck:
    latent (diffusion map) -> PCA coefficients -> ambient fields.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

# Ensure local imports work whether this is run as a script or imported.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))
notebook_root = repo_root / "notebooks"
if notebook_root.exists() and str(notebook_root) not in sys.path:
    sys.path.append(str(notebook_root))

from scripts.modelagent import build_agent
from scripts.pca_precomputed_main import (  # type: ignore
    PCADataAgent,
    _array_checksum,
    _load_cached_result,
    _meta_hash,
    _resolve_cache_base,
    _save_cached_result,
    build_lift_times,
    lift_latent_trajectory,
    load_pca_data,
    prepare_timecoupled_latents,
)
from scripts.time_local_lifting import ensure_time_coupled_lifter, fetch_training_times
from tran_inclusions.interpolation import build_dense_latent_trajectories

from mmsfm.data_utils import pca_decode
from mmsfm.diffusion_map_sampler import DiffusionMapTrajectorySampler
from mmsfm.deep_kernel import AttentionWeightNet, DeepKernelEncoder
from mmsfm.preimage_decoder import PreimageEnergyDecoder, TimeConditionedInitializer
from mmsfm.psi_provider import PsiProvider
from utils import (
    IndexableMinMaxScaler,
    IndexableNoOpScaler,
    IndexableStandardScaler,
    build_zt,
    get_device,
)


def _coerce_value(text: str) -> Any:
    """Parse values stored in args.txt."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return stripped


def load_args_file(args_path: Path) -> argparse.Namespace:
    """Load the experiment args.txt file into a Namespace."""
    values = {}
    with args_path.open("r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, raw_val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _coerce_value(raw_val)
    return argparse.Namespace(**values)


def _ensure_defaults(run_args: argparse.Namespace) -> argparse.Namespace:
    """Backfill defaults for fields that may be missing in args.txt."""
    defaults = {
        "agent_type": "precomputed",
        "flowmatcher": "sb",
        "scaler_type": "standard",
        "diff_ref": "miniflow",
        "precomputed_ratio_max": 100.0,
        "precomputed_t_clip_eps": 1e-4,
        "n_dense": 200,
        "frechet_mode": "triplet",
        "flow_param": "velocity",
        "min_sigma_ratio": 1e-4,
        "flow_loss_weight": 1.0,
        "score_loss_weight": 1.0,
        "t_dim": 32,
        "tc_k": 80,
        "tc_alpha": 1.0,
        "tc_beta": -0.2,
        "tc_epsilon_scales_min": 0.01,
        "tc_epsilon_scales_max": 0.2,
        "tc_epsilon_scales_num": 32,
        "tc_power_iter_tol": 1e-12,
        "tc_power_iter_maxiter": 10_000,
        "tc_neighbor_k": 16,
        "tc_batch_lift": 32,
        "eval_method": "nearest",
        "nogpu": False,
        "no_cache": False,
        "refresh_cache": False,
    }
    for key, val in defaults.items():
        if getattr(run_args, key, None) is None:
            setattr(run_args, key, val)

    if getattr(run_args, "flow_lr", None) is None:
        run_args.flow_lr = run_args.lr
    if getattr(run_args, "score_lr", None) is None:
        run_args.score_lr = run_args.lr
    if getattr(run_args, "zt", None) is None:
        run_args.zt = []
    if getattr(run_args, "eval_zt_idx", None) == "None":
        run_args.eval_zt_idx = None
    return run_args


def _build_scaler(name: str):
    name = (name or "standard").lower()
    if name == "minmax":
        return IndexableMinMaxScaler()
    if name == "standard":
        return IndexableStandardScaler()
    if name == "noop":
        return IndexableNoOpScaler()
    raise ValueError(f"Unknown scaler_type '{name}'.")


def _to_list_time_major(arr: np.ndarray) -> list[np.ndarray]:
    """Convert (T, N, D) arrays to a list of length T."""
    return [np.asarray(arr[t]) for t in range(arr.shape[0])]


@dataclass
class DecodingArtifacts:
    args: argparse.Namespace
    agent: PCADataAgent
    scaler: Any
    lifter: Any
    pca_info: dict
    training_times: Optional[np.ndarray]
    zt: np.ndarray
    norm_test_latents: Optional[list[np.ndarray]]
    cache_base: Path
    tc_info: dict
    decoder: "AbstractDecoder"


class AbstractDecoder:
    """Interface for latent→PCA→ambient decoding strategies."""

    def latent_to_pca(
        self,
        latent_traj: np.ndarray,
        *,
        target_times: Optional[np.ndarray] = None,
        normalized: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def pca_to_fields(self, coeffs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(
        self,
        latent_traj: np.ndarray,
        *,
        target_times: Optional[np.ndarray] = None,
        normalized: bool = False,
        decode_fields: bool = False,
    ):
        coeffs = self.latent_to_pca(
            latent_traj,
            target_times=target_times,
            normalized=normalized,
        )
        if not decode_fields:
            return coeffs, None
        fields = self.pca_to_fields(coeffs)
        return coeffs, fields


class ConvexHullDecoder(AbstractDecoder):
    """Default decoder that lifts via ConvexHullInterpolator then inverts PCA."""

    def __init__(
        self,
        lifter,
        training_times: Optional[np.ndarray],
        scaler,
        pca_info: dict,
        *,
        neighbor_k: int = 16,
        batch_size: int = 32,
        whitening_epsilon: float = 1e-12,
    ):
        self.lifter = lifter
        self.training_times = training_times
        self.scaler = scaler
        self.pca_info = pca_info
        self.neighbor_k = int(neighbor_k)
        self.batch_size = int(batch_size)
        self.whitening_epsilon = float(whitening_epsilon)

    def _denorm(self, latent_traj: np.ndarray) -> np.ndarray:
        traj_list = _to_list_time_major(latent_traj)
        denorm_list = self.scaler.inverse_transform(traj_list)
        return np.stack(denorm_list, axis=0)

    def latent_to_pca(
        self,
        latent_traj: np.ndarray,
        *,
        target_times: Optional[np.ndarray] = None,
        normalized: bool = False,
    ) -> np.ndarray:
        traj = np.asarray(latent_traj, dtype=np.float64)
        if normalized:
            if self.scaler is None:
                raise ValueError("Scaler required to denormalize normalized latents.")
            traj = self._denorm(traj)

        return lift_latent_trajectory(
            traj,
            self.lifter,
            neighbor_k=self.neighbor_k,
            batch_size=self.batch_size,
            target_times=target_times,
            training_times=self.training_times,
        )

    def pca_to_fields(self, coeffs: np.ndarray) -> np.ndarray:
        return pca_decode(
            coeffs,
            self.pca_info["components"],
            self.pca_info["mean"],
            self.pca_info.get("explained_variance"),
            self.pca_info.get("is_whitened", True),
            self.whitening_epsilon,
        )


def _normalize_psi_mode(mode: str) -> str:
    mode = (mode or "nearest").lower().strip()
    if mode in {"nearest"}:
        return "nearest"
    if mode in {"interpolation", "linear"}:
        return "interpolation"
    raise ValueError("psi_mode must be one of {'nearest', 'interpolation'}.")


class PreimageSolverDecoder(AbstractDecoder):
    """Decoder that inverts the deep-kernel encoder via an iterative pre-image solver."""

    def __init__(
        self,
        solver: PreimageEnergyDecoder,
        scaler,
        pca_info: dict,
        *,
        device: torch.device,
        psi_mode: str = "nearest",
        batch_size: int = 128,
        whitening_epsilon: float = 1e-12,
    ):
        self.solver = solver.to(device)
        self.solver.eval()
        self.scaler = scaler
        self.pca_info = pca_info
        self.device = device
        self.psi_mode = _normalize_psi_mode(psi_mode)
        self.batch_size = int(batch_size)
        self.whitening_epsilon = float(whitening_epsilon)

    def _norm(self, latent_traj: np.ndarray) -> np.ndarray:
        traj_list = _to_list_time_major(latent_traj)
        norm_list = self.scaler.transform(traj_list)
        return np.stack(norm_list, axis=0)

    def latent_to_pca(
        self,
        latent_traj: np.ndarray,
        *,
        target_times: Optional[np.ndarray] = None,
        normalized: bool = False,
    ) -> np.ndarray:
        traj = np.asarray(latent_traj, dtype=np.float32)
        if traj.ndim != 3:
            raise ValueError("Expected latent_traj with shape (T, N, latent_dim).")
        if target_times is None:
            raise ValueError("target_times is required for PreimageSolverDecoder.")
        times = np.asarray(target_times, dtype=np.float32).reshape(-1)
        if times.shape[0] != traj.shape[0]:
            raise ValueError("target_times length must match latent_traj time dimension.")

        if not normalized:
            if self.scaler is None:
                raise ValueError("Scaler required to normalize unnormalized latents.")
            traj = self._norm(traj)

        T, N, _ = traj.shape
        x_dim = self.solver.x_dim
        out = np.zeros((T, N, x_dim), dtype=np.float32)

        for ti in range(T):
            t_scalar = float(times[ti])
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                y_batch = torch.from_numpy(traj[ti, start:end]).to(self.device)
                x_hat = self.solver.decode(y_batch, t_scalar, psi_mode=self.psi_mode, differentiable=False)
                out[ti, start:end] = x_hat.detach().cpu().numpy()
        return out

    def pca_to_fields(self, coeffs: np.ndarray) -> np.ndarray:
        return pca_decode(
            coeffs,
            self.pca_info["components"],
            self.pca_info["mean"],
            self.pca_info.get("explained_variance"),
            self.pca_info.get("is_whitened", True),
            self.whitening_epsilon,
        )


def _build_psi_provider(
    cache_base: Path,
    run_args: argparse.Namespace,
    tc_info: dict,
    tc_cache_meta: dict,
    scaler,
    zt_train_times: np.ndarray,
    refresh_cache: bool,
    allow_recompute: bool,
    *,
    psi_mode: str,
) -> PsiProvider:
    psi_mode = _normalize_psi_mode(psi_mode)
    interp_cache_meta = {
        "version": 1,
        "tc_cache_hash": _meta_hash(tc_cache_meta),
        "n_dense": run_args.n_dense,
        "frechet_mode": run_args.frechet_mode,
        "times_train": np.round(zt_train_times, 8).tolist(),
        "latent_train_shape": tuple(tc_info["latent_train_tensor"].shape),
        "metric_alpha": 0.0,
        "compute_global": True,
        "compute_triplet": run_args.frechet_mode == "triplet",
    }
    interp_cache_path = cache_base / "interpolation.pkl"
    interp_result = _load_cached_result(
        interp_cache_path,
        interp_cache_meta,
        "latent interpolation",
        refresh=refresh_cache,
    )

    if interp_result is None:
        if not allow_recompute:
            raise FileNotFoundError(
                f"Interpolation cache missing at {interp_cache_path}. "
                "Re-run with --allow-recompute-cache to build dense trajectories."
            )
        interp_result = build_dense_latent_trajectories(
            tc_info["traj_result"],
            times_train=zt_train_times,
            tc_embeddings_time=tc_info["latent_train_tensor"],
            n_dense=run_args.n_dense,
            frechet_mode=run_args.frechet_mode,
            compute_global=True,
            compute_triplet=(run_args.frechet_mode == "triplet"),
            compute_naive=False,
        )
        if not getattr(run_args, "no_cache", False):
            _save_cached_result(interp_cache_path, interp_result, interp_cache_meta, "latent interpolation")

    if run_args.frechet_mode == "triplet":
        dense_trajs = interp_result.phi_frechet_triplet_dense
    else:
        dense_trajs = interp_result.phi_frechet_global_dense
    if dense_trajs is None:
        dense_trajs = interp_result.phi_frechet_dense
    if dense_trajs is None:
        raise ValueError("Dense trajectories are missing from interpolation cache.")

    dense_list = [dense_trajs[t] for t in range(dense_trajs.shape[0])]
    norm_dense_trajs = np.stack(scaler.transform(dense_list), axis=0).astype(np.float32)
    return PsiProvider(interp_result.t_dense.astype(np.float32), norm_dense_trajs, mode=psi_mode)


def _load_preimage_solver(
    ckpt_path: Path,
    *,
    psi_provider: PsiProvider,
    device: torch.device,
) -> PreimageEnergyDecoder:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    else:
        state_dict = payload
        config = {}

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint payload at {ckpt_path}.")

    alpha_raw = state_dict.get("step_size.alpha_raw")
    d_raw = state_dict.get("preconditioner.d_raw")
    U = state_dict.get("preconditioner.U")
    keys = state_dict.get("encoder.weight_net.keys")
    if alpha_raw is None or d_raw is None or U is None or keys is None:
        raise ValueError(
            "Checkpoint missing required keys; expected at least "
            "{'step_size.alpha_raw','preconditioner.d_raw','preconditioner.U','encoder.weight_net.keys'}."
        )

    t_len = int(alpha_raw.shape[0])
    x_dim = int(d_raw.shape[1])
    rank = int(U.shape[2])
    n_train = int(keys.shape[0])
    key_dim = int(keys.shape[1])

    # Infer weight-net dimensions.
    time_dim = int(state_dict["encoder.weight_net.time_emb.mlp.0.weight"].shape[0])
    hidden_dim = int(state_dict["encoder.weight_net.q_net.0.weight"].shape[0])
    q_linear_keys = [k for k in state_dict if k.startswith("encoder.weight_net.q_net.") and k.endswith(".weight")]
    depth = max(len(q_linear_keys) - 2, 0)

    weight_net = AttentionWeightNet(
        x_dim=x_dim,
        n_train=n_train,
        key_dim=key_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        depth=depth,
        topk=None,
    )
    encoder = DeepKernelEncoder(psi_provider, weight_net)

    # Infer initializer dimensions.
    init_time_dim = int(state_dict["init_net.time_emb.mlp.0.weight"].shape[0])
    init_hidden_dim = int(state_dict["init_net.net.0.weight"].shape[0])
    init_in_dim = int(state_dict["init_net.net.0.weight"].shape[1])
    y_dim = int(init_in_dim - init_time_dim)
    init_linear_keys = [k for k in state_dict if k.startswith("init_net.net.") and k.endswith(".weight")]
    init_depth = max(len(init_linear_keys) - 2, 0)
    init_net = TimeConditionedInitializer(
        y_dim=y_dim,
        x_dim=x_dim,
        hidden_dim=init_hidden_dim,
        depth=init_depth,
        time_dim=init_time_dim,
    )

    t_knots = state_dict.get("t_knots")
    if t_knots is None:
        t_knots = torch.linspace(0.0, 1.0, steps=t_len, dtype=torch.float32)

    use_prior = any(k.startswith("prior_mu.") for k in state_dict) and any(k.startswith("prior_sigma.") for k in state_dict)
    prior_mu = np.zeros((t_len, x_dim), dtype=np.float32) if use_prior else None
    prior_sigma = np.ones((t_len, x_dim), dtype=np.float32) if use_prior else None

    solver = PreimageEnergyDecoder(
        encoder=encoder,
        x_dim=x_dim,
        y_dim=y_dim,
        t_knots=t_knots.detach().cpu().numpy(),
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        lambda_reg=float(config.get("lambda_reg", 0.0)),
        preconditioner_rank=rank,
        preconditioner_eps=float(config.get("preconditioner_eps", 1e-4)),
        n_steps=int(config.get("n_steps", 8)),
        init_net=init_net,
        sigma_min=float(config.get("sigma_min", 1e-6)),
    )
    solver.load_state_dict(state_dict, strict=False)
    solver.to(device)
    solver.eval()
    return solver


def _load_or_compute_tc_info(
    cache_base: Path,
    run_args: argparse.Namespace,
    zt_rem_idxs: np.ndarray,
    *,
    full_marginals: list[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    marginal_times: np.ndarray,
    refresh_cache: bool,
    allow_recompute: bool,
) -> tuple[dict, dict]:
    """Load cached time-coupled diffusion map artifacts or compute them."""
    tc_cache_meta = {
        "version": 1,
        "data_path": str(Path(run_args.data_path).resolve()),
        "test_size": run_args.test_size,
        "seed": run_args.seed,
        "hold_one_out": getattr(run_args, "hold_one_out", None),
        "tc_k": run_args.tc_k,
        "tc_alpha": run_args.tc_alpha,
        "tc_beta": run_args.tc_beta,
        "tc_epsilon_scales_min": run_args.tc_epsilon_scales_min,
        "tc_epsilon_scales_max": run_args.tc_epsilon_scales_max,
        "tc_epsilon_scales_num": run_args.tc_epsilon_scales_num,
        "tc_power_iter_tol": run_args.tc_power_iter_tol,
        "tc_power_iter_maxiter": run_args.tc_power_iter_maxiter,
        "zt": np.round(run_args.zt, 8).tolist(),
        "zt_rem_idxs": np.asarray(zt_rem_idxs, dtype=int).tolist(),
        "marginal_times": np.round(np.asarray(marginal_times, dtype=float), 8).tolist()
        if marginal_times is not None
        else None,
        "train_idx_checksum": _array_checksum(train_idx),
        "test_idx_checksum": _array_checksum(test_idx),
    }
    tc_cache_path = cache_base / "tc_embeddings.pkl"
    tc_info = _load_cached_result(
        tc_cache_path,
        tc_cache_meta,
        "time-coupled embeddings",
        refresh=refresh_cache,
    )
    if tc_info is not None:
        return tc_info, tc_cache_meta

    if not allow_recompute:
        raise FileNotFoundError(
            f"Cached embeddings not found at {tc_cache_path}. "
            "Re-run with --allow-recompute-cache or precompute the cache."
        )
    tc_info = prepare_timecoupled_latents(
        full_marginals,
        train_idx=train_idx,
        test_idx=test_idx,
        zt_rem_idxs=np.arange(len(full_marginals)),
        times_raw=np.array(marginal_times, dtype=float),
        tc_k=run_args.tc_k,
        tc_alpha=run_args.tc_alpha,
        tc_beta=run_args.tc_beta,
        tc_epsilon_scales_min=run_args.tc_epsilon_scales_min,
        tc_epsilon_scales_max=run_args.tc_epsilon_scales_max,
        tc_epsilon_scales_num=run_args.tc_epsilon_scales_num,
        tc_power_iter_tol=run_args.tc_power_iter_tol,
        tc_power_iter_maxiter=run_args.tc_power_iter_maxiter,
    )
    if not getattr(run_args, "no_cache", False):
        _save_cached_result(tc_cache_path, tc_info, tc_cache_meta, "time-coupled embeddings")
    return tc_info, tc_cache_meta


def _build_precomputed_sampler(
    cache_base: Path,
    run_args: argparse.Namespace,
    tc_info: dict,
    tc_cache_meta: dict,
    scaler,
    zt_train_times: np.ndarray,
    refresh_cache: bool,
    allow_recompute: bool,
) -> DiffusionMapTrajectorySampler:
    """Build the dense-trajectory sampler used by precomputed agents."""
    interp_cache_meta = {
        "version": 1,
        "tc_cache_hash": _meta_hash(tc_cache_meta),
        "n_dense": run_args.n_dense,
        "frechet_mode": run_args.frechet_mode,
        "times_train": np.round(zt_train_times, 8).tolist(),
        "latent_train_shape": tuple(tc_info["latent_train_tensor"].shape),
        "metric_alpha": 0.0,
        "compute_global": True,
        "compute_triplet": run_args.frechet_mode == "triplet",
    }
    interp_cache_path = cache_base / "interpolation.pkl"
    interp_result = _load_cached_result(
        interp_cache_path,
        interp_cache_meta,
        "latent interpolation",
        refresh=refresh_cache,
    )

    if interp_result is None:
        if not allow_recompute:
            raise FileNotFoundError(
                f"Interpolation cache missing at {interp_cache_path}. "
                "Re-run with --allow-recompute-cache to build dense trajectories."
            )
        interp_result = build_dense_latent_trajectories(
            tc_info["traj_result"],
            times_train=zt_train_times,
            tc_embeddings_time=tc_info["latent_train_tensor"],
            n_dense=run_args.n_dense,
            frechet_mode=run_args.frechet_mode,
            compute_global=True,
            compute_triplet=(run_args.frechet_mode == "triplet"),
            compute_naive=False,
        )
        if not getattr(run_args, "no_cache", False):
            _save_cached_result(interp_cache_path, interp_result, interp_cache_meta, "latent interpolation")

    if run_args.frechet_mode == "triplet":
        dense_trajs = interp_result.phi_frechet_triplet_dense
    else:
        dense_trajs = interp_result.phi_frechet_global_dense
    if dense_trajs is None:
        dense_trajs = interp_result.phi_frechet_dense

    dense_list = [dense_trajs[t] for t in range(dense_trajs.shape[0])]
    norm_dense_trajs = np.stack(scaler.transform(dense_list), axis=0)
    return DiffusionMapTrajectorySampler(interp_result.t_dense, norm_dense_trajs, interpolate=False)


def load_decoding_artifacts(
    run_dir: Path,
    *,
    data_path: Optional[str],
    cache_dir: Optional[str],
    refresh_cache: bool,
    allow_recompute_cache: bool,
    neighbor_k: int,
    batch_size: int,
    decoder_type: str = "convex",
    psi_mode: str = "nearest",
    preimage_ckpt: Optional[str] = None,
    preimage_batch_size: int = 128,
    device_override: Optional[str] = None,
    whitening_epsilon: float = 1e-12,
) -> DecodingArtifacts:
    """Load trained models, lifter, scaler, and decoder from a run directory."""
    args_path = run_dir / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"args.txt not found in {run_dir}")

    run_args = load_args_file(args_path)
    run_args = _ensure_defaults(run_args)
    if data_path is not None:
        run_args.data_path = data_path
    if cache_dir is not None:
        run_args.cache_dir = cache_dir
    run_args.refresh_cache = refresh_cache
    run_args.no_cache = getattr(run_args, "no_cache", False)

    # Load PCA data and align with training setup (drop first marginal).
    data_tuple = load_pca_data(
        run_args.data_path,
        run_args.test_size,
        run_args.seed,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple
    if len(data) > 0:
        data = data[1:]
        testdata = testdata[1:]
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]

    marginals = list(range(len(data)))
    zt_full = build_zt(run_args.zt, marginals)
    run_args.zt = zt_full
    zt_rem_idxs = np.arange(zt_full.shape[0], dtype=int)
    if getattr(run_args, "hold_one_out", None) is not None:
        zt_rem_idxs = np.delete(zt_rem_idxs, int(run_args.hold_one_out))
        data = [data[i] for i in zt_rem_idxs]
        testdata = [testdata[i] for i in zt_rem_idxs]
        full_marginals = [full_marginals[i] for i in zt_rem_idxs]
        marginal_times = marginal_times[zt_rem_idxs]
    zt_train_times = zt_full[zt_rem_idxs]

    cache_base = _resolve_cache_base(run_args.cache_dir, run_args.data_path)

    tc_info, tc_cache_meta = _load_or_compute_tc_info(
        cache_base,
        run_args,
        zt_rem_idxs,
        full_marginals=full_marginals,
        train_idx=train_idx,
        test_idx=test_idx,
        marginal_times=marginal_times,
        refresh_cache=refresh_cache,
        allow_recompute=allow_recompute_cache,
    )

    scaler = _build_scaler(run_args.scaler_type)
    scaler.fit(tc_info["latent_train"])
    norm_test_latents = scaler.transform(tc_info["latent_test"])

    sampler = None
    if run_args.agent_type == "precomputed":
        sampler = _build_precomputed_sampler(
            cache_base,
            run_args,
            tc_info,
            tc_cache_meta,
            scaler,
            zt_train_times,
            refresh_cache,
            allow_recompute_cache,
        )

    device = device_override or get_device(run_args.nogpu)
    dim = run_args.tc_k
    if sampler is not None:
        base_agent = build_agent(run_args.agent_type, run_args, zt_rem_idxs, dim, sampler, device=device)
    else:
        base_agent = build_agent(run_args.agent_type, run_args, zt_rem_idxs, dim, device=device)
    agent: PCADataAgent = PCADataAgent(base_agent, pca_info)

    flow_ckpt = run_dir / "flow_model.pth"
    if not flow_ckpt.exists():
        raise FileNotFoundError(f"flow_model.pth not found in {run_dir}")
    agent.model.load_state_dict(torch.load(flow_ckpt, map_location=device))  # type: ignore[arg-type]
    if agent.is_sb:
        score_ckpt = run_dir / "score_model.pth"
        if not score_ckpt.exists():
            raise FileNotFoundError(
                "Score model missing; SB decoding requires score_model.pth for backward sampling."
            )
        agent.score_model.load_state_dict(torch.load(score_ckpt, map_location=device))  # type: ignore[arg-type]
        agent.score_model.eval()  # type: ignore[union-attr]
    agent.model.eval()

    lifter = ensure_time_coupled_lifter(tc_info)
    training_times = fetch_training_times(tc_info, lifter)
    decoder_kind = (decoder_type or "convex").lower().strip()
    if decoder_kind == "convex":
        decoder: AbstractDecoder = ConvexHullDecoder(
            lifter,
            training_times,
            scaler,
            pca_info,
            neighbor_k=neighbor_k,
            batch_size=batch_size,
            whitening_epsilon=whitening_epsilon,
        )
    elif decoder_kind == "preimage":
        if preimage_ckpt is None:
            raise ValueError("--preimage-ckpt is required when --decoder-type=preimage.")
        psi_provider = _build_psi_provider(
            cache_base,
            run_args,
            tc_info,
            tc_cache_meta,
            scaler,
            zt_train_times,
            refresh_cache,
            allow_recompute_cache,
            psi_mode=psi_mode,
        )
        solver = _load_preimage_solver(
            Path(preimage_ckpt),
            psi_provider=psi_provider,
            device=torch.device(device),
        )
        decoder = PreimageSolverDecoder(
            solver,
            scaler,
            pca_info,
            device=torch.device(device),
            psi_mode=psi_mode,
            batch_size=preimage_batch_size,
            whitening_epsilon=whitening_epsilon,
        )
    else:
        raise ValueError("--decoder-type must be one of {'convex','preimage'}.")

    return DecodingArtifacts(
        args=run_args,
        agent=agent,
        scaler=scaler,
        lifter=lifter,
        pca_info=pca_info,
        training_times=training_times,
        zt=zt_train_times,
        norm_test_latents=norm_test_latents,
        cache_base=cache_base,
        tc_info=tc_info,
        decoder=decoder,
    )


def _load_latent_file(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    return np.asarray(data)


def _maybe_limit_samples(traj: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None:
        return traj
    return traj[:, :limit]


def main():
    parser = argparse.ArgumentParser(description="Prototype decoding/lifting strategies.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training output directory containing flow_model.pth.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional override of the PCA npz path.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache base used for tc embeddings.")
    parser.add_argument("--latent-path", type=str, default=None, help="Optional latent trajectory .npy to decode.")
    parser.add_argument("--latent-times", type=str, default=None, help="Optional npy/npz of times for the latent trajectory slices.")
    parser.add_argument("--latent-normalized", action="store_true", help="Set if latent-path is already in normalized model space.")
    parser.add_argument("--generate-backward", action="store_true", help="Generate backward SDE trajectory when no latent-path is provided.")
    parser.add_argument("--neighbor-k", type=int, default=16, help="Nearest neighbors for convex-hull lifting.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for lifting.")
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="convex",
        choices=["convex", "preimage"],
        help="Decoder backend: 'convex' uses ConvexHullInterpolator; 'preimage' uses the iterative pre-image solver.",
    )
    parser.add_argument(
        "--psi-mode",
        type=str,
        default="nearest",
        choices=["nearest", "interpolation"],
        help="How to sample precomputed Psi(t): 'nearest' or 'interpolation' (linear in time).",
    )
    parser.add_argument(
        "--preimage-ckpt",
        type=str,
        default=None,
        help="Path to a pre-image solver checkpoint (.pth). Required for --decoder-type=preimage.",
    )
    parser.add_argument(
        "--preimage-batch-size",
        type=int,
        default=128,
        help="Batch size for pre-image decoding (independent from --batch-size).",
    )
    parser.add_argument("--limit-samples", type=int, default=None, help="Restrict to the first N samples for quick experiments.")
    parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for saved outputs (default: <run_dir>/decoded).")
    parser.add_argument("--decode-ambient", action="store_true", help="Decode PCA coefficients to ambient fields.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached tc/interpolation artifacts.")
    parser.add_argument("--allow-recompute-cache", action="store_true", help="Recompute caches when they are missing.")
    parser.add_argument(
        "--lift-times",
        type=float,
        nargs="+",
        default=None,
        help="Times in [0,1] to lift; defaults to training zt plus midpoints.",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device.")
    parser.add_argument("--whitening-epsilon", type=float, default=1e-12, help="Stability epsilon for PCA decode.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    artifacts = load_decoding_artifacts(
        run_dir,
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        allow_recompute_cache=args.allow_recompute_cache,
        neighbor_k=args.neighbor_k,
        batch_size=args.batch_size,
        decoder_type=args.decoder_type,
        psi_mode=args.psi_mode,
        preimage_ckpt=args.preimage_ckpt,
        preimage_batch_size=args.preimage_batch_size,
        device_override=args.device,
        whitening_epsilon=args.whitening_epsilon,
    )

    latent_times = None
    normalized_latents = args.latent_normalized
    if args.latent_times:
        latent_times = _load_latent_file(Path(args.latent_times))

    if args.latent_path:
        latent_traj = _load_latent_file(Path(args.latent_path))
    else:
        if artifacts.norm_test_latents is None:
            raise RuntimeError("Test latents unavailable; cannot generate trajectories without latent-path.")
        norm_x0 = artifacts.norm_test_latents[0]
        norm_xT = artifacts.norm_test_latents[-1]
        if args.generate_backward:
            _, latent_traj = artifacts.agent.traj_gen(norm_xT, generate_backward=True)
        else:
            latent_traj, _ = artifacts.agent.traj_gen(norm_x0, generate_backward=False)
        normalized_latents = True

    latent_traj = _maybe_limit_samples(latent_traj, args.limit_samples)

    lift_times = (
        np.asarray(args.lift_times, dtype=float) if args.lift_times is not None else build_lift_times(artifacts.zt)
    )
    traj_for_lift = artifacts.agent._get_traj_at_zt(latent_traj, lift_times)

    coeffs, fields = artifacts.decoder.decode(
        traj_for_lift,
        target_times=lift_times if latent_times is None else latent_times,
        normalized=normalized_latents,
        decode_fields=args.decode_ambient,
    )

    out_prefix = Path(args.out_prefix) if args.out_prefix is not None else run_dir / "decoded"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out_prefix}_lift_times.npy", lift_times)
    np.save(f"{out_prefix}_pca_coeffs.npy", coeffs)
    if fields is not None:
        np.save(f"{out_prefix}_fields.npy", fields)

    print(f"Decoded PCA coefficients saved to {out_prefix}_pca_coeffs.npy")
    if fields is not None:
        print(f"Decoded ambient fields saved to {out_prefix}_fields.npy")


if __name__ == "__main__":
    main()
