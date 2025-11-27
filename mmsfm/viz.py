from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def format_for_paper() -> None:
    """Apply publication-style defaults for matplotlib figures."""
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
    plt.close("all")


def _global_vrange(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([a.ravel() for a in arrays if a is not None])
    return float(np.min(vals)), float(np.max(vals))


def plot_field_comparisons(
    imgs_true: np.ndarray,
    imgs_gh: np.ndarray,
    imgs_gh_local: Optional[np.ndarray],
    imgs_convex: np.ndarray,
    sample_indices: Sequence[int],
    imgs_krr: Optional[np.ndarray] = None,
    vmax_mode: Literal["global", "per_sample"] = "global",
) -> None:
    """Plot side-by-side comparisons of reconstructed fields."""
    methods: list[tuple[str, np.ndarray]] = [("GH", imgs_gh)]
    if imgs_gh_local is not None:
        methods.append(("GH-local", imgs_gh_local))
    methods.append(("Convex", imgs_convex))
    if imgs_krr is not None:
        methods.append(("KRR (time-local)", imgs_krr))

    n_rows = len(sample_indices)
    n_cols = 1 + 2 * len(methods)  # truth + prediction + error for each method

    if vmax_mode == "global":
        vmin, vmax = _global_vrange([imgs_true] + [arr for _, arr in methods])
        err_vmax = float(
            np.max(
                np.concatenate(
                    [np.abs(arr - imgs_true).ravel() for _, arr in methods]
                )
            )
        )
    else:
        vmin = vmax = err_vmax = None  # computed per-sample

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(sample_indices):
        if vmax_mode == "per_sample":
            vmin, vmax = _global_vrange(
                [imgs_true[idx : idx + 1]] + [arr[idx : idx + 1] for _, arr in methods]
            )
            err_vmax = float(
                np.max(
                    np.concatenate(
                        [np.abs(arr[idx] - imgs_true[idx]).ravel() for _, arr in methods]
                    )
                )
            )
        col = 0
        ax = axes[row, col]
        im = ax.imshow(imgs_true[idx], vmin=vmin, vmax=vmax)
        ax.set_title(f"Truth (idx={idx})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        col += 1

        for name, arr in methods:
            pred_img = arr[idx]
            rmse = float(np.sqrt(np.mean((pred_img - imgs_true[idx]) ** 2)))

            ax_pred = axes[row, col]
            im_pred = ax_pred.imshow(pred_img, vmin=vmin, vmax=vmax)
            ax_pred.set_title(f"{name} (RMSE={rmse:.2e})")
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            ax_pred.figure.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.01)
            col += 1

            ax_err = axes[row, col]
            im_err = ax_err.imshow(np.abs(pred_img - imgs_true[idx]), vmin=0.0, vmax=err_vmax)
            ax_err.set_title(f"{name} |error|")
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            ax_err.figure.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.01)
            col += 1

    plt.suptitle("Holdout reconstruction comparison", y=1.02)
    plt.show()


def plot_error_statistics(
    metrics: dict[str, dict[str, Any]],
    g_star: np.ndarray,
) -> None:
    """Plot distributions of errors across methods."""
    if not metrics:
        print("No metrics to plot.")
        return

    method_names = list(metrics.keys())
    rel_errors = [metrics[name]["rel_error"] for name in method_names]
    rmses = [metrics[name]["rmse"] for name in method_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    parts = axes[0].violinplot(rel_errors, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightgray")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    axes[0].set_xticks(np.arange(1, len(method_names) + 1))
    axes[0].set_xticklabels(method_names)
    axes[0].set_ylabel("Relative error")
    axes[0].set_title("Relative error distribution")
    axes[0].grid(alpha=0.3)

    for name, errs in zip(method_names, rmses):
        axes[1].scatter(g_star[:, 0], errs, s=10, alpha=0.6, label=name)
    axes[1].set_xlabel("$g_1$ (held-out latent coord)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Error vs latent coordinate")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.show()
