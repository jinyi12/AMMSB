"""Utilities for analysing PCA-coefficient MMSFM runs inside notebooks."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmsfm.models.models import MLP, ResNet

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _format_for_paper() -> None:
    """Best-effort Matplotlib style update without hard dependency."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    plt.rcParams.update({"image.cmap": "viridis"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "Utopia",
                "ITC Bookman",
                "Bookman",
                "Nimbus Roman No9 L",
                "Palatino",
                "Charter",
                "serif",
            ]
        }
    )
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"mathtext.fontset": "custom"})
    plt.rcParams.update({"mathtext.rm": "serif"})
    plt.rcParams.update({"mathtext.it": "serif:italic"})
    plt.rcParams.update({"mathtext.bf": "serif:bold"})


class IndexableScaler:
    """Minimal indexable scaler interface used for PCA analysis."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._fitted = False

    def fit(self, arrays: Sequence[np.ndarray]) -> None:
        self._fit([np.asarray(arr) for arr in arrays])
        self._fitted = True

    def transform(self, arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self._check_fitted()
        return [self._transform(np.asarray(arr)) for arr in arrays]

    def inverse_transform(self, arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self._check_fitted()
        return [self._inverse(np.asarray(arr)) for arr in arrays]

    def _fit(self, arrays: Sequence[np.ndarray]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transform(self, array: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _inverse(self, array: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} has not been fitted.")


class IndexableNoOpScaler(IndexableScaler):
    def __init__(self) -> None:
        super().__init__('IndexableNoOpScaler')

    def _fit(self, arrays: Sequence[np.ndarray]) -> None:
        return None

    def _transform(self, array: np.ndarray) -> np.ndarray:
        return array.copy()

    def _inverse(self, array: np.ndarray) -> np.ndarray:
        return array.copy()


class IndexableStandardScaler(IndexableScaler):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__('IndexableStandardScaler')
        self.eps = eps

    def _fit(self, arrays: Sequence[np.ndarray]) -> None:
        stacked = np.vstack(arrays)
        self.mean_ = stacked.mean(axis=0)
        self.scale_ = stacked.std(axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)

    def _transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean_) / self.scale_

    def _inverse(self, array: np.ndarray) -> np.ndarray:
        return array * self.scale_ + self.mean_


class IndexableMinMaxScaler(IndexableScaler):
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__('IndexableMinMaxScaler')
        self.eps = eps

    def _fit(self, arrays: Sequence[np.ndarray]) -> None:
        stacked = np.vstack(arrays)
        self.data_min_ = stacked.min(axis=0)
        self.data_max_ = stacked.max(axis=0)
        self.range_ = self.data_max_ - self.data_min_
        self.range_ = np.where(self.range_ < self.eps, 1.0, self.range_)

    def _transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.data_min_) / self.range_

    def _inverse(self, array: np.ndarray) -> np.ndarray:
        return array * self.range_ + self.data_min_


def build_zt(
    zt: Optional[Sequence[float]],
    marginals: Sequence[int],
) -> np.ndarray:
    """Recreate the zt construction used during training without external deps."""
    if zt is None:
        count = len(marginals)
        if count == 0:
            raise ValueError('Cannot infer zt without marginals.')
        values = np.linspace(0.0, 1.0, count)
    else:
        values = np.asarray(list(zt), dtype=float)
        if values.ndim != 1:
            raise ValueError('zt must be one-dimensional.')
        if not np.all(np.diff(values) > 0):
            raise ValueError('zt values must be strictly increasing.')
        if len(values) != len(marginals):
            raise ValueError('Length of zt must match number of marginals.')
    a = values[0]
    b = values[-1]
    if np.isclose(b, a):
        return np.zeros_like(values)
    return (values - a) / (b - a)


def parse_args_file(args_path: Path) -> Dict[str, Any]:
    """Parse the key=value args.txt artifact written by MMSFM training scripts."""
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")

    parsed: Dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


def _resolve_path(base: Path, maybe_path: Any) -> Path:
    """Resolve dataset paths stored in config relative to project root."""
    path = Path(str(maybe_path))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_pca_dataset(
    data_path: Path,
    test_size: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """Load PCA coefficient marginals and split into train/test partitions."""
    npz = np.load(data_path)

    marginal_keys = sorted(
        [k for k in npz.keys() if k.startswith("marginal_")],
        key=lambda name: float(name.split("_")[1]),
    )

    all_marginals = [np.asarray(npz[k]) for k in marginal_keys]
    if not all_marginals:
        raise ValueError(f"No marginals found in {data_path}")

    n_samples = all_marginals[0].shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    if 0 < test_size < 1:
        test_count = int(round(test_size * n_samples))
    else:
        test_count = int(test_size)
    test_count = max(1, min(test_count, n_samples - 1))
    test_idx = np.sort(perm[:test_count])
    train_idx = np.sort(perm[test_count:])

    train_data = [marg[train_idx] for marg in all_marginals]
    test_data = [marg[test_idx] for marg in all_marginals]

    pca_info: Dict[str, Any] = {
        "components": np.asarray(npz["pca_components"]),
        "mean": np.asarray(npz["pca_mean"]),
        "explained_variance": np.asarray(npz["pca_explained_variance"]),
        "data_dim": int(npz["data_dim"]),
        "is_whitened": bool(npz["is_whitened"]) if "is_whitened" in npz else True,
    }
    return train_data, test_data, pca_info


def _build_scaler(kind: str) -> IndexableScaler:
    kind_lower = (kind or "standard").lower()
    if kind_lower == "minmax":
        return IndexableMinMaxScaler()
    if kind_lower == "standard":
        return IndexableStandardScaler()
    if kind_lower == "noop":
        return IndexableNoOpScaler()
    raise ValueError(f"Unknown scaler_type '{kind}'.")


def _compute_resolution(data_dim: int) -> Optional[int]:
    if data_dim <= 0:
        return None
    root = int(round(math.sqrt(data_dim)))
    if root * root == data_dim:
        return root
    return None


def _build_time_conditioned_model(
    modelname: str,
    dim: int,
    width: int,
    depth: int,
    device: torch.device,
) -> nn.Module:
    name = modelname.lower()
    if name == "mlp":
        model = MLP(dim=dim, w=width, depth=depth, time_varying=True)
    elif name == "resnet":
        model = ResNet(dim=dim, w=width, depth=depth, time_varying=True)
    else:
        raise ValueError(f"Unsupported modelname '{modelname}'.")
    return model.to(device)


class TimeConditionedFlow(nn.Module):
    """Wrap a time-conditioned MLP/ResNet to expose (t, x) -> v_t(x) signature."""

    def __init__(self, base_model: nn.Module, feature_shape: Tuple[int, ...]):
        super().__init__()
        self.base_model = base_model
        self.feature_shape = feature_shape
        flat_dim = int(np.prod(feature_shape))
        if flat_dim <= 0:
            raise ValueError(f"Invalid feature shape {feature_shape}")
        self.flat_dim = flat_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_flat = x.view(x.shape[0], -1)
        if x_flat.shape[1] != self.flat_dim:
            raise ValueError(
                f"Expected feature dimension {self.flat_dim}, got {x_flat.shape[1]}"
            )
        t_flat = t.view(-1, 1).to(device=x_flat.device, dtype=x_flat.dtype)
        xt = torch.cat([x_flat, t_flat], dim=-1)
        velocities = self.base_model(xt)
        return velocities.view(*original_shape)


@dataclass
class PCARunArtifacts:
    """Container bundling together the artefacts required for notebook analysis."""

    config: Dict[str, Any]
    result_dir: Path
    base_dir: Path
    zt: np.ndarray
    zt_rem_idxs: np.ndarray
    scaler: IndexableScaler
    flow_model: nn.Module
    score_model: Optional[nn.Module]
    pca_info: Dict[str, Any]
    train_data: List[np.ndarray]
    test_data: List[np.ndarray]
    coeff_dim: int
    data_dim: int
    resolution: Optional[int]
    n_infer: int
    t_infer: int
    sigma: float
    device: torch.device
    is_sb: bool
    scaler_type: str

    def _transform_list(
        self,
        arrays: Sequence[np.ndarray],
        *,
        inverse: bool = False,
    ) -> List[np.ndarray]:
        if inverse:
            transformed = self.scaler.inverse_transform(arrays)
        else:
            transformed = self.scaler.transform(arrays)
        if isinstance(transformed, np.ndarray):
            transformed = list(transformed)
        return [np.asarray(arr) for arr in transformed]  # type: ignore[list-item]

    def transform_single(
        self,
        array: np.ndarray,
        *,
        inverse: bool = False,
    ) -> np.ndarray:
        result = (
            self.scaler.inverse_transform([array])[0]
            if inverse
            else self.scaler.transform([array])[0]
        )
        return np.asarray(result)

    def test_marginals(
        self,
        *,
        normalized: bool = False,
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        arrays = (
            self._transform_list(self.test_data) if normalized else self.test_data
        )
        target_device = device or torch.device("cpu")
        return [
            torch.from_numpy(np.asarray(arr)).float().to(target_device) for arr in arrays
        ]

    def train_marginals(
        self,
        *,
        normalized: bool = False,
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        arrays = (
            self._transform_list(self.train_data) if normalized else self.train_data
        )
        target_device = device or torch.device("cpu")
        return [
            torch.from_numpy(np.asarray(arr)).float().to(target_device) for arr in arrays
        ]

    def endpoints(
        self,
        *,
        normalized: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        test_tensors = self.test_marginals(
            normalized=normalized, device=device or torch.device("cpu")
        )
        return test_tensors[0], test_tensors[-1]

    def stacked_test_tensor(
        self,
        *,
        normalized: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        tensors = self.test_marginals(normalized=normalized, device=device)
        return torch.stack(tensors, dim=0)

    def flow_wrapper(self) -> TimeConditionedFlow:
        feature_shape = (self.coeff_dim,)
        return TimeConditionedFlow(self.flow_model, feature_shape)


def load_pca_run_artifacts(
    result_dir: Path,
    *,
    base_dir: Path,
    device: Optional[torch.device] = None,
) -> PCARunArtifacts:
    """Load trained PCA MMSFM checkpoint, dataset splits, and scalers."""
    device = device or torch.device("cpu")
    args_path = result_dir / "args.txt"
    config = parse_args_file(args_path)

    test_size = float(config.get("test_size", 0.2))
    seed = int(config.get("seed", 42))
    data_path = _resolve_path(base_dir, config.get("data_path", "data/mm_data.npz"))

    train_data_all, test_data, pca_info = _load_pca_dataset(data_path, test_size, seed)

    marginals = list(range(len(train_data_all)))
    zt_config = config.get("zt", None)
    if isinstance(zt_config, (list, tuple)) and len(zt_config) == 0:
        zt_config = None
    zt = build_zt(zt_config, marginals)

    zt_rem_idxs = np.arange(len(zt), dtype=int)
    hold_one_out = config.get("hold_one_out", None)
    if hold_one_out is not None and hold_one_out != "None":
        hold_idx = int(hold_one_out)
        if hold_idx < 0 or hold_idx >= len(zt_rem_idxs):
            raise ValueError(f"hold_one_out index {hold_idx} out of range")
        zt_rem_idxs = np.delete(zt_rem_idxs, hold_idx)

    train_data = [train_data_all[i] for i in zt_rem_idxs]

    scaler_type = str(config.get("scaler_type", "standard"))
    scaler = _build_scaler(scaler_type)
    scaler.fit(train_data)

    coeff_dim = int(train_data[0].shape[1])
    data_dim = int(pca_info["data_dim"])
    resolution = _compute_resolution(data_dim)

    modelname = str(config.get("modelname", "mlp"))
    width = int(config.get("w_len", 64))
    depth = int(config.get("modeldepth", 2))

    flow_model = _build_time_conditioned_model(
        modelname,
        coeff_dim,
        width,
        depth,
        device,
    )
    flow_state_path = result_dir / "flow_model.pth"
    flow_state = torch.load(flow_state_path, map_location=device, weights_only=True)
    flow_model.load_state_dict(flow_state)

    is_sb = str(config.get("flowmatcher", "ot")).lower() == "sb"
    if is_sb:
        score_model = _build_time_conditioned_model(
            modelname,
            coeff_dim,
            width,
            depth,
            device,
        )
        score_state_path = result_dir / "score_model.pth"
        if not score_state_path.exists():
            raise FileNotFoundError(
                "Score model required by config but score_model.pth not found."
            )
        score_state = torch.load(score_state_path, map_location=device, weights_only=True)
        score_model.load_state_dict(score_state)
    else:
        score_model = None

    n_infer = min(int(config.get("n_infer", test_data[0].shape[0])), test_data[0].shape[0])
    t_infer = int(config.get("t_infer", 400))
    sigma = float(config.get("sigma", 0.3))

    return PCARunArtifacts(
        config=config,
        result_dir=result_dir,
        base_dir=base_dir,
        zt=zt,
        zt_rem_idxs=zt_rem_idxs,
        scaler=scaler,
        flow_model=flow_model,
        score_model=score_model,
        pca_info=pca_info,
        train_data=train_data,
        test_data=test_data,
        coeff_dim=coeff_dim,
        data_dim=data_dim,
        resolution=resolution,
        n_infer=n_infer,
        t_infer=t_infer,
        sigma=sigma,
        device=device,
        is_sb=is_sb,
        scaler_type=scaler_type,
    )


_TRAJ_VARIANTS = {
    ("ode", False, False): "ode_traj_epoch{epoch}.npy",
    ("ode", True, False): "ode_traj_at_zt_epoch{epoch}.npy",
    ("sde", False, False): "sde_traj_epoch{epoch}.npy",
    ("sde", True, False): "sde_traj_at_zt_epoch{epoch}.npy",
    ("sde", False, True): "sde_traj_backward_epoch{epoch}.npy",
}


def list_trajectory_epochs(
    result_dir: Path,
    kind: str,
    *,
    at_zt: bool = False,
    backward: bool = False,
) -> List[int]:
    """Return sorted list of epochs with saved trajectories."""
    key = (kind.lower(), at_zt, backward)
    if key not in _TRAJ_VARIANTS:
        raise ValueError(f"Unsupported trajectory kind {key}")
    pattern = _TRAJ_VARIANTS[key].replace("{epoch}", "*")
    epochs: List[int] = []
    for path in result_dir.glob(pattern):
        match = re.search(r"epoch(\d+)", path.stem)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def load_trajectory_array(
    result_dir: Path,
    kind: str,
    *,
    epoch: Optional[int] = None,
    at_zt: bool = False,
    backward: bool = False,
) -> np.ndarray:
    """Load ODE/SDE trajectory array saved during training."""
    available = list_trajectory_epochs(result_dir, kind, at_zt=at_zt, backward=backward)
    if not available:
        raise FileNotFoundError(
            f"No trajectories found for kind='{kind}' (at_zt={at_zt}, backward={backward})"
        )
    chosen_epoch = epoch or available[-1]
    if chosen_epoch not in available:
        raise FileNotFoundError(
            f"Trajectory for epoch {chosen_epoch} not found (available: {available})"
        )
    key = (kind.lower(), at_zt, backward)
    pattern = _TRAJ_VARIANTS[key].format(epoch=chosen_epoch)
    path = result_dir / pattern
    return np.load(path)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_coefficient_scatter(
    marginals: Sequence[torch.Tensor | np.ndarray],
    time_points: Sequence[float],
    *,
    dims: Tuple[int, int] = (0, 1),
    n_samples: int = 512,
    out_path: Optional[Path] = None,
    close: bool = True,
) -> Optional[Figure]:
    """Scatter plot of PCA coefficients across marginals for quick inspection."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _format_for_paper()
    arrays = [
        np.asarray(m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m)
        for m in marginals
    ]

    num_marginals = len(arrays)
    if num_marginals == 0:
        raise ValueError("No marginals provided to plot_coefficient_scatter.")

    ncols = min(4, num_marginals)
    nrows = (num_marginals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes_arr = np.atleast_1d(axes).flatten()

    t_min = float(np.min(time_points)) if len(time_points) > 0 else 0.0
    t_max = float(np.max(time_points)) if len(time_points) > 0 else 1.0
    denom = (t_max - t_min) or 1.0

    global_x: List[float] = []
    global_y: List[float] = []

    for idx, (ax, arr) in enumerate(zip(axes_arr, arrays)):
        sample = arr
        if sample.shape[0] > n_samples:
            indices = np.linspace(0, sample.shape[0] - 1, n_samples).astype(int)
            sample = sample[indices]
        x = sample[:, dims[0]]
        y = sample[:, dims[1]]
        global_x.extend(x.tolist())
        global_y.extend(y.tolist())

        t_val = float(time_points[idx]) if idx < len(time_points) else idx
        colour = cm.viridis((t_val - t_min) / denom)
        ax.scatter(x, y, s=6, alpha=0.35, color=colour, edgecolors="none")
        ax.set_title(f"t = {t_val:.3f}")
        ax.set_xlabel(f"coeff[{dims[0]}]")
        ax.set_ylabel(f"coeff[{dims[1]}]")

    if global_x and global_y:
        x_min, x_max = min(global_x), max(global_x)
        y_min, y_max = min(global_y), max(global_y)
        x_margin = (x_max - x_min) * 0.1 or 1e-3
        y_margin = (y_max - y_min) * 0.1 or 1e-3
        for ax in axes_arr[:num_marginals]:
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for ax in axes_arr[num_marginals:]:
        ax.axis("off")

    fig.tight_layout()

    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if close:
        plt.close(fig)
        return None
    return fig


def plot_coefficient_trajectories(
    trajectory: torch.Tensor | np.ndarray,
    *,
    dims: Tuple[int, int] = (0, 1),
    sample_indices: Optional[Sequence[int]] = None,
    n_samples: int = 6,
    time_points: Optional[Sequence[float]] = None,
    out_path: Optional[Path] = None,
    close: bool = True,
) -> Optional[Figure]:
    """Plot selected coefficient trajectories in a lower-dimensional projection."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _format_for_paper()
    traj = (
        trajectory.detach().cpu().numpy()
        if isinstance(trajectory, torch.Tensor)
        else np.asarray(trajectory)
    )
    if traj.ndim != 3:
        raise ValueError("Trajectory must have shape (T, N, D)")

    T, N, _ = traj.shape
    if sample_indices is None:
        if n_samples >= N:
            sample_indices = list(range(N))
        else:
            sample_indices = np.linspace(0, N - 1, n_samples).astype(int).tolist()
    else:
        sample_indices = [int(i) for i in sample_indices]

    if time_points is None or len(time_points) != T:
        time_points = np.linspace(0.0, 1.0, T)
    time_points = np.asarray(time_points)

    norm = plt.Normalize(vmin=time_points.min(), vmax=time_points.max())
    cmap = cm.get_cmap("viridis")

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    for idx in sample_indices:
        coords = traj[:, idx, :]
        xy = coords[:, [dims[0], dims[1]]]
        colours = cmap(norm(time_points))
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            linewidth=1.5,
            alpha=0.85,
            color=colours[-1],
        )
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=time_points,
            cmap=cmap,
            s=8,
            alpha=0.8,
        )
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            marker="o",
            color="black",
            s=30,
            label="start" if idx == sample_indices[0] else None,
        )
        ax.scatter(
            xy[-1, 0],
            xy[-1, 1],
            marker="X",
            color="crimson",
            s=40,
            label="end" if idx == sample_indices[0] else None,
        )

    ax.set_xlabel(f"coeff[{dims[0]}]")
    ax.set_ylabel(f"coeff[{dims[1]}]")
    ax.set_title("Sample coefficient trajectories")
    ax.grid(alpha=0.15)
    if sample_indices:
        ax.legend(loc="best", frameon=False)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("time")

    fig.tight_layout()

    if out_path is not None:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if close:
        plt.close(fig)
        return None
    return fig
