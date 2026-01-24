
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scripts.wandb_compat import wandb
from scripts.noise_schedules import ExponentialContractingSchedule

def plot_latent_trajectories(
    traj: np.ndarray,       # (T, N, K)
    reference: np.ndarray,  # (T_ref, N_ref, K) - reference marginals
    zt: np.ndarray,         # (T_ref,) - reference time points
    save_path: Path,
    title: str = "Latent Trajectories",
    n_highlight: int = 10,
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Plot trajectories in latent space."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(8, 8))

    d0, d1 = dims
    T, N, K = traj.shape

    # Plot reference marginals
    T_ref = reference.shape[0]
    colors = cm.viridis(np.linspace(0, 1, T_ref))
    for t_idx in range(T_ref):
        ax.scatter(
            reference[t_idx, :, d0],
            reference[t_idx, :, d1],
            c=[colors[t_idx]],
            alpha=0.3,
            s=5,
            label=f"t={zt[t_idx]:.2f}" if t_idx % 2 == 0 else None,
        )

    # Plot trajectories
    n_plot = min(n_highlight, N)
    for i in range(n_plot):
        ax.plot(
            traj[:, i, d0],
            traj[:, i, d1],
            c="black",
            alpha=0.5,
            linewidth=0.5,
        )

    ax.set_xlabel(f"Latent dim {d0}")
    ax.set_ylabel(f"Latent dim {d1}")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_latent_vector_field(
    velocity_model: nn.Module,
    latent_data: np.ndarray,  # (T, N, K) - for determining bounds
    zt: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    grid_size: int = 20,
    t_values: list[float] = None,
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Plot learned velocity field as quiver plot."""
    import matplotlib.pyplot as plt

    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    d0, d1 = dims
    K = latent_data.shape[2]

    # Compute bounds from data
    all_data = latent_data.reshape(-1, K)
    x_min, x_max = all_data[:, d0].min(), all_data[:, d0].max()
    y_min, y_max = all_data[:, d1].min(), all_data[:, d1].max()
    margin = 0.1
    x_min -= margin * (x_max - x_min)
    x_max += margin * (x_max - x_min)
    y_min -= margin * (y_max - y_min)
    y_max += margin * (y_max - y_min)

    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 5))
    if len(t_values) == 1:
        axes = [axes]

    velocity_model.eval()

    for ax, t_val in zip(axes, t_values):
        # Create grid
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        XX, YY = np.meshgrid(xx, yy)

        # Build full latent vectors (set other dims to mean)
        mean_other = all_data.mean(axis=0)
        grid_points = np.zeros((grid_size * grid_size, K), dtype=np.float32)
        grid_points[:, d0] = XX.ravel()
        grid_points[:, d1] = YY.ravel()
        for d in range(K):
            if d not in dims:
                grid_points[:, d] = mean_other[d]

        # Evaluate velocity
        with torch.no_grad():
            y = torch.from_numpy(grid_points).float().to(device)
            t = torch.full((len(grid_points),), t_val, device=device)
            v = velocity_model(y, t=t).cpu().numpy()

        U = v[:, d0].reshape(grid_size, grid_size)
        V = v[:, d1].reshape(grid_size, grid_size)

        ax.quiver(XX, YY, U, V, alpha=0.7)
        ax.set_xlabel(f"Latent dim {d0}")
        ax.set_ylabel(f"Latent dim {d1}")
        ax.set_title(f"t = {t_val:.2f}")

    plt.suptitle("Learned Velocity Field")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/vector_field": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_score_vector_field(
    score_model: nn.Module,
    schedule: ExponentialContractingSchedule,
    latent_data: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    grid_size: int = 20,
    t_values: list[float] = None,
    dims: tuple[int, int] = (0, 1),
    score_parameterization: Literal["scaled", "raw"] = "scaled",
    run=None,
) -> None:
    """Plot learned score field as quiver plot with color-coded magnitude.

    Args:
        score_model: Trained score model.
        schedule: Noise schedule for computing g(t).
        latent_data: Training data for determining bounds, shape (T, N, K).
        save_path: Where to save the figure.
        device: Torch device.
        grid_size: Grid resolution.
        t_values: Time points to visualize.
        dims: Which latent dimensions to plot.
        score_parameterization: Whether model outputs "scaled" or "raw" score.
        run: Optional wandb run for logging.
    """
    import matplotlib.pyplot as plt

    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    d0, d1 = dims
    K = latent_data.shape[2]

    # Compute bounds from data
    all_data = latent_data.reshape(-1, K)
    x_min, x_max = all_data[:, d0].min(), all_data[:, d0].max()
    y_min, y_max = all_data[:, d1].min(), all_data[:, d1].max()
    margin = 0.1
    x_min -= margin * (x_max - x_min)
    x_max += margin * (x_max - x_min)
    y_min -= margin * (y_max - y_min)
    y_max += margin * (y_max - y_min)

    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 5))
    if len(t_values) == 1:
        axes = [axes]

    score_model.eval()

    for ax, t_val in zip(axes, t_values):
        # Create grid
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        XX, YY = np.meshgrid(xx, yy)

        # Build full latent vectors
        mean_other = all_data.mean(axis=0)
        grid_points = np.zeros((grid_size * grid_size, K), dtype=np.float32)
        grid_points[:, d0] = XX.ravel()
        grid_points[:, d1] = YY.ravel()
        for d in range(K):
            if d not in dims:
                grid_points[:, d] = mean_other[d]

        # Evaluate score
        with torch.no_grad():
            y = torch.from_numpy(grid_points).float().to(device)
            t = torch.full((len(grid_points),), t_val, device=device)
            s_theta = score_model(y, t=t)

            # Convert to raw score if needed for visualization
            if score_parameterization == "scaled":
                # s_scaled = (g^2/2) * s_raw, so s_raw = s_scaled / (g^2/2)
                g_t = schedule.sigma_t(t).unsqueeze(-1)
                score_raw = s_theta / (g_t ** 2 / 2.0 + 1e-8)
            else:
                score_raw = s_theta

            score_np = score_raw.cpu().numpy()

        U = score_np[:, d0].reshape(grid_size, grid_size)
        V = score_np[:, d1].reshape(grid_size, grid_size)
        magnitude = np.sqrt(U**2 + V**2)

        # Color by magnitude
        quiv = ax.quiver(XX, YY, U, V, magnitude, alpha=0.7, cmap='viridis')
        plt.colorbar(quiv, ax=ax, label='Score Magnitude')

        # Overlay data distribution
        T_ref = latent_data.shape[0]
        t_idx = np.argmin(np.abs(np.array([0.0, 1.0]) - t_val))  # nearest marginal
        if T_ref > t_idx:
            ax.scatter(
                latent_data[t_idx, :, d0],
                latent_data[t_idx, :, d1],
                c='red',
                alpha=0.2,
                s=5,
                label='Data'
            )

        ax.set_xlabel(f"Latent dim {d0}")
        ax.set_ylabel(f"Latent dim {d1}")
        ax.set_title(f"Score Field (raw) at t={t_val:.2f}")
        ax.legend()

    plt.suptitle("Learned Score Field")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/score_field": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_velocity_vs_score_magnitude(
    velocity_model: nn.Module,
    score_model: nn.Module,
    schedule: ExponentialContractingSchedule,
    latent_data: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    n_samples: int = 1000,
    score_parameterization: Literal["scaled", "raw"] = "scaled",
    run=None,
) -> None:
    """Compare magnitudes of velocity and score terms along trajectories.

    This helps diagnose if the score term dominates the velocity term,
    causing backward SDE to diverge.
    """
    import matplotlib.pyplot as plt

    velocity_model.eval()
    score_model.eval()

    T, N, K = latent_data.shape

    # Sample random points from training data
    t_indices = np.random.randint(0, T, size=n_samples)
    sample_indices = np.random.randint(0, N, size=n_samples)

    y_samples = torch.from_numpy(
        latent_data[t_indices, sample_indices]
    ).float().to(device)

    # Sample times uniformly
    t_samples = torch.rand(n_samples, device=device)

    with torch.no_grad():
        # Evaluate velocity
        v = velocity_model(y_samples, t=t_samples)
        v_mag = torch.norm(v, dim=1).cpu().numpy()

        # Evaluate score
        s_theta = score_model(y_samples, t=t_samples)

        # Convert to score contribution in backward SDE drift
        if score_parameterization == "scaled":
            # In backward SDE: score_term = s_scaled directly
            score_term = s_theta
        else:
            # In backward SDE: score_term = (g^2/2) * s_raw
            g_t = schedule.sigma_t(t_samples).unsqueeze(-1)
            score_term = (g_t ** 2 / 2.0) * s_theta

        score_mag = torch.norm(score_term, dim=1).cpu().numpy()

    t_np = t_samples.cpu().numpy()

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Magnitude vs time
    axes[0].scatter(t_np, v_mag, alpha=0.3, s=5, label='Velocity', c='blue')
    axes[0].scatter(t_np, score_mag, alpha=0.3, s=5, label='Score term', c='red')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_yscale('log')
    axes[0].set_title('Velocity vs Score Magnitude over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio
    ratio = score_mag / (v_mag + 1e-8)
    axes[1].scatter(t_np, ratio, alpha=0.3, s=5, c='purple')
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal magnitude')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Score / Velocity Ratio')
    axes[1].set_yscale('log')
    axes[1].set_title('Score-to-Velocity Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Histogram of ratio
    axes[2].hist(ratio, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(x=1.0, color='black', linestyle='--', label='Equal magnitude')
    axes[2].set_xlabel('Score / Velocity Ratio')
    axes[2].set_ylabel('Count')
    axes[2].set_xscale('log')
    axes[2].set_title('Distribution of Ratio')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/velocity_vs_score": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)

    # Print statistics
    print("\nVelocity vs Score Magnitude Statistics:")
    print(f"  Velocity magnitude: mean={v_mag.mean():.4f}, std={v_mag.std():.4f}")
    print(f"  Score term magnitude: mean={score_mag.mean():.4f}, std={score_mag.std():.4f}")
    print(f"  Score/Velocity ratio: mean={ratio.mean():.4f}, median={np.median(ratio):.4f}")
    print(f"  Ratio > 1 (score dominates): {(ratio > 1).mean() * 100:.1f}% of samples")


def plot_marginal_comparison(
    generated: np.ndarray,   # (T, N, D) or (T, N, K)
    reference: np.ndarray,   # (T_ref, N_ref, D) or (T_ref, N_ref, K)
    zt: np.ndarray,
    t_indices: list[int],
    save_path: Path,
    title: str = "Marginal Comparison",
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Compare generated vs reference marginals at specific times."""
    import matplotlib.pyplot as plt

    d0, d1 = dims
    n_plots = len(t_indices)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, t_indices):
        # Reference
        ax.scatter(
            reference[t_idx, :, d0],
            reference[t_idx, :, d1],
            c="blue",
            alpha=0.3,
            s=10,
            label="Reference",
        )
        # Generated
        ax.scatter(
            generated[t_idx, :, d0],
            generated[t_idx, :, d1],
            c="red",
            alpha=0.3,
            s=10,
            label="Generated",
        )
        ax.set_title(f"t = {zt[t_idx]:.2f}")
        ax.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_training_curves(
    flow_losses: np.ndarray,
    score_losses: np.ndarray,
    save_path: Path,
    run=None,
) -> None:
    """Plot training loss curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(flow_losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Flow Loss")
    ax1.set_title("Flow Loss")
    ax1.set_yscale("log")

    ax2.plot(score_losses)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score Loss")
    ax2.set_title("Score Loss")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/training_curves": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)
