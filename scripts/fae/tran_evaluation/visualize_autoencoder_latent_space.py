#!/usr/bin/env python
"""Visualize FAE latent spaces for one or more trained autoencoder runs.

For each run directory, this script:
1) loads the FAE checkpoint + dataset split,
2) encodes fields into latent codes,
3) fits a PCA embedding,
4) writes per-run latent scatter plots and cached latent arrays.

It also writes a combined multi-run comparison figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.analyze_latent_noise_sweep import compute_latent_codes  # noqa: E402
from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: E402
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
)
from scripts.fae.fae_naive.train_attention_components import (  # noqa: E402
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)
from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from scripts.images.field_visualization import format_for_paper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize latent space of trained FAE autoencoders.")
    p.add_argument(
        "--run_dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Run directory containing args.json and checkpoints. Repeat for multiple runs.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/latent_space_autoencoder_viz",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
    )
    p.add_argument(
        "--max_samples_per_time",
        type=int,
        default=256,
        help="Max loaded samples per time marginal (0 means all).",
    )
    p.add_argument(
        "--points_per_time_plot",
        type=int,
        default=600,
        help="Max points shown per time marginal in scatter plots.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _resolve_existing_path(raw_path: str | Path | None, roots: list[Path]) -> Optional[Path]:
    if raw_path is None:
        return None
    raw = Path(str(raw_path))
    candidates = [raw, REPO_ROOT / raw]
    for root in roots:
        candidates.append(root / raw)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _resolve_checkpoint(run_dir: Path) -> Path:
    args_json = _load_json(run_dir / "args.json")
    if "fae_checkpoint" in args_json:
        external = _resolve_existing_path(args_json["fae_checkpoint"], [run_dir, Path.cwd()])
        if external is not None and external.exists():
            return external
    candidates = [
        run_dir / "checkpoints" / "best_state.pkl",
        run_dir / "checkpoints" / "state.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"No FAE checkpoint found in {run_dir}")


def _normalise_raw_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def _load_time_data(
    run_dir: Path,
    *,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args_json = _load_json(run_dir / "args.json")

    data_path = _resolve_existing_path(args_json.get("data_path"), [run_dir, Path.cwd()])
    if data_path is None:
        raise FileNotFoundError(f"Could not resolve data_path from {run_dir / 'args.json'}")

    train_ratio_raw = args_json.get("train_ratio", 0.8)
    train_ratio = 0.8 if train_ratio_raw is None else float(train_ratio_raw)

    raw_indices = _normalise_raw_list(args_json.get("held_out_indices", "")).strip()
    raw_times = _normalise_raw_list(args_json.get("held_out_times", "")).strip()

    held_out_indices: Optional[list[int]] = None
    if raw_indices and raw_indices.lower() not in {"none", "null", "false", "no"}:
        held_out_indices = parse_held_out_indices_arg(raw_indices)
    elif raw_times and raw_times.lower() not in {"none", "null", "false", "no"}:
        meta = load_dataset_metadata(str(data_path))
        times_norm = meta.get("times_normalized")
        if times_norm is None:
            raise ValueError(f"Dataset missing times_normalized for held_out_times in {run_dir}")
        held_out_indices = parse_held_out_times_arg(raw_times, np.asarray(times_norm, dtype=np.float32))

    time_data = load_training_time_data_naive(
        str(data_path),
        held_out_indices=held_out_indices,
        train_ratio=train_ratio,
        split=split,  # type: ignore[arg-type]
        max_samples=max_samples_per_time,
        seed=seed,
    )
    if not time_data:
        raise ValueError(f"No time marginals available for {run_dir}")

    coords = np.asarray(time_data[0]["x"], dtype=np.float32)
    n_common = min(int(d["u"].shape[0]) for d in time_data)
    if n_common < 1:
        raise ValueError(f"No usable samples for {run_dir}")

    fields_per_time = np.stack(
        [np.asarray(d["u"][:n_common], dtype=np.float32) for d in time_data],
        axis=0,
    )
    dataset_time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)
    dataset_times_norm = np.asarray([float(d["t_norm"]) for d in time_data], dtype=np.float32)
    return fields_per_time, coords, dataset_time_indices, dataset_times_norm


def _fit_pca(
    z: np.ndarray,  # (M, K)
    *,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_components = int(min(max(1, n_components), z.shape[1]))
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=seed)
        z_proj = pca.fit_transform(z)
        mean = pca.mean_.astype(np.float32)
        components = pca.components_.astype(np.float32)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        return z_proj.astype(np.float32), mean, components, explained.astype(np.float32)
    except Exception:
        mean = z.mean(axis=0, keepdims=True)
        z_center = z - mean
        _, _, vt = np.linalg.svd(z_center, full_matrices=False)
        components = vt[:n_components]
        z_proj = z_center @ components.T
        var = np.var(z_proj, axis=0)
        var_ratio = var / max(float(var.sum()), 1e-12)
        return (
            z_proj.astype(np.float32),
            mean.astype(np.float32).ravel(),
            components.astype(np.float32),
            var_ratio.astype(np.float32),
        )


def _model_label(args_json: dict[str, Any]) -> str:
    optimizer = str(args_json.get("optimizer", "unknown")).upper()
    loss = str(args_json.get("loss_type", "unknown")).lower()
    use_prior = bool(args_json.get("use_prior", False))

    if loss == "ntk_scaled":
        loss_name = "NTK"
    elif loss == "l2":
        loss_name = "L2"
    else:
        loss_name = loss.upper()

    if use_prior:
        return f"{optimizer}+{loss_name}+Prior"
    return f"{optimizer}+{loss_name}"


def _safe_name(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    name = "".join(out).strip("_").lower()
    return name or "run"


def _plot_single_run(
    *,
    title: str,
    z_proj: np.ndarray,  # (T, N, 2)
    time_indices: np.ndarray,  # (T,)
    times_norm: np.ndarray,  # (T,)
    out_prefix: Path,
    points_per_time_plot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_count, n_count, _ = z_proj.shape

    keep_points: list[np.ndarray] = []
    keep_time: list[np.ndarray] = []
    for t in range(t_count):
        n_take = min(points_per_time_plot, n_count)
        if n_take < n_count:
            idx = rng.choice(n_count, size=n_take, replace=False)
        else:
            idx = np.arange(n_count)
        keep_points.append(z_proj[t, idx])
        keep_time.append(np.full((n_take,), t, dtype=np.int64))

    points = np.concatenate(keep_points, axis=0)
    t_labels = np.concatenate(keep_time, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.4))
    cmap = plt.get_cmap("cividis", t_count)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=t_labels,
        cmap=cmap,
        s=8,
        alpha=0.75,
        linewidths=0.0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("time-marginal index")
    cbar.set_ticks(np.arange(t_count))
    cbar.set_ticklabels([f"{int(i)}@{float(tn):.3f}" for i, tn in zip(time_indices, times_norm)])

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=250 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    return points, t_labels


def _plot_combined_grid(
    *,
    run_titles: list[str],
    sampled_points: list[np.ndarray],  # each (M, 2)
    sampled_tlabels: list[np.ndarray],  # each (M,)
    n_times: int,
    out_path: Path,
) -> None:
    n_runs = len(run_titles)
    n_cols = 3
    n_rows = int(np.ceil(n_runs / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows), squeeze=False)
    cmap = plt.get_cmap("cividis", n_times)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i >= n_runs:
            ax.axis("off")
            continue
        pts = sampled_points[i]
        tl = sampled_tlabels[i]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=tl,
            cmap=cmap,
            s=6,
            alpha=0.70,
            linewidths=0.0,
        )
        ax.set_title(run_titles[i], fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=250 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    format_for_paper()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    grid_titles: list[str] = []
    grid_points: list[np.ndarray] = []
    grid_tlabels: list[np.ndarray] = []

    for i, run_raw in enumerate(args.run_dirs, start=1):
        run_dir = Path(run_raw).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir does not exist: {run_dir}")

        args_json = _load_json(run_dir / "args.json")
        label = _model_label(args_json)
        run_tag = _safe_name(f"{label}_{run_dir.name}")
        run_out = out_dir / run_tag
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(args.run_dirs)}] {run_dir}")
        print(f"  model: {label}")

        ckpt_path = _resolve_checkpoint(run_dir)
        print(f"  checkpoint: {ckpt_path}")

        fields_per_time, coords, time_indices, times_norm = _load_time_data(
            run_dir,
            split=args.split,
            max_samples_per_time=int(args.max_samples_per_time),
            seed=int(args.seed),
        )
        print(
            "  data:",
            f"T={fields_per_time.shape[0]}",
            f"N={fields_per_time.shape[1]}",
            f"P={fields_per_time.shape[2]}",
            f"K?=pending",
        )

        ckpt = load_fae_checkpoint(ckpt_path)
        autoencoder, params, batch_stats, _meta = build_attention_fae_from_checkpoint(ckpt)
        latent_codes = compute_latent_codes(
            autoencoder,
            params,
            batch_stats,
            fields_per_time,
            coords,
            batch_size=64,
        )
        t_count, n_count, latent_dim = latent_codes.shape
        print(f"  latent shape: T={t_count}, N={n_count}, K={latent_dim}")

        flat = latent_codes.reshape(-1, latent_dim).astype(np.float32)
        z_proj_flat, mean_vec, comp_mat, explained = _fit_pca(flat, n_components=2, seed=int(args.seed))
        z_proj = z_proj_flat.reshape(t_count, n_count, 2)

        np.savez_compressed(
            run_out / "latent_projection_data.npz",
            latent_codes=latent_codes.astype(np.float32),
            latent_pca2=z_proj.astype(np.float32),
            dataset_time_indices=time_indices.astype(np.int64),
            dataset_times_norm=times_norm.astype(np.float32),
            pca_mean=mean_vec.astype(np.float32),
            pca_components=comp_mat.astype(np.float32),
            pca_explained_variance_ratio=explained.astype(np.float32),
            run_dir=str(run_dir),
            label=label,
        )

        pts, tlabels = _plot_single_run(
            title=label,
            z_proj=z_proj,
            time_indices=time_indices,
            times_norm=times_norm,
            out_prefix=run_out / "latent_pca2_scatter",
            points_per_time_plot=int(args.points_per_time_plot),
            seed=int(args.seed),
        )

        grid_titles.append(label)
        grid_points.append(pts)
        grid_tlabels.append(tlabels)
        summaries.append(
            {
                "run_dir": str(run_dir),
                "label": label,
                "checkpoint": str(ckpt_path),
                "split": args.split,
                "T": int(t_count),
                "N_per_time": int(n_count),
                "latent_dim": int(latent_dim),
                "dataset_time_indices": time_indices.tolist(),
                "dataset_times_norm": times_norm.tolist(),
                "artifacts": {
                    "plot_png": str(run_out / "latent_pca2_scatter.png"),
                    "plot_pdf": str(run_out / "latent_pca2_scatter.pdf"),
                    "npz": str(run_out / "latent_projection_data.npz"),
                },
            }
        )

    if grid_titles:
        n_times = int(max(int(np.max(tl)) for tl in grid_tlabels) + 1)
        _plot_combined_grid(
            run_titles=grid_titles,
            sampled_points=grid_points,
            sampled_tlabels=grid_tlabels,
            n_times=n_times,
            out_path=out_dir / "latent_pca2_scatter_grid",
        )

    (out_dir / "latent_space_viz_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved latent-space visualizations to {out_dir}")
    print(f"- {out_dir / 'latent_space_viz_summary.json'}")
    print(f"- {out_dir / 'latent_pca2_scatter_grid.png'} / .pdf")


if __name__ == "__main__":
    main()
