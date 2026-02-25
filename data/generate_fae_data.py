"""Generate raw 2D field data for the Functional Autoencoder (FAE) pipeline.

Wraps existing generators from ``multimarginal_generation.py`` and saves raw
fields with grid coordinates in an npz archive.  No PCA is applied – fields
are stored at full spatial resolution so the FAE can operate directly on
``(x, y, t)`` coordinate queries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.multimarginal_generation import (
    affine_standardize_marginals,
    generate_multiscale_grf_data,
    generate_tran_multiscale_inclusion_data,
    log_standardize_marginals,
    minmax_scale_marginals,
    parse_held_out_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_grid_coords(resolution: int) -> np.ndarray:
    """Return pixel-centre coordinates on ``[0, 1]^2``.

    Parameters
    ----------
    resolution : int
        Number of pixels along each axis.

    Returns
    -------
    coords : np.ndarray, shape ``[resolution**2, 2]``
        Each row is ``(x_i, y_i)`` with values in ``(0, 1)``.
    """
    half = 0.5 / resolution
    ticks = np.linspace(half, 1.0 - half, resolution)
    X, Y = np.meshgrid(ticks, ticks, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw 2D field data for the FAE pipeline."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the .npz file."
    )
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--n_constraints", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")

    # Generator selection
    parser.add_argument(
        "--data_generator",
        type=str,
        default="grf",
        choices=["grf", "tran_inclusion"],
    )

    # GRF-specific
    parser.add_argument("--generation_method", type=str, default="fft")
    parser.add_argument("--covariance_type", type=str, default="exponential")
    parser.add_argument("--schedule_type", type=str, default="geometric")
    parser.add_argument("--L_domain", type=float, default=1.0)
    parser.add_argument("--micro_corr_length", type=float, default=0.1)
    parser.add_argument("--H_max_factor", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--H_divisor", type=float, default=6.0)
    parser.add_argument("--mean_val", type=float, default=10.0)
    parser.add_argument("--std_val", type=float, default=2.0)
    parser.add_argument("--kl_error_threshold", type=float, default=1e-3)
    parser.add_argument("--concentration", type=float, default=2.0)

    # Tran-inclusion-specific
    parser.add_argument("--tran_D_large", type=float, default=1.0)
    parser.add_argument("--tran_vol_frac_large", type=float, default=0.2)
    parser.add_argument("--tran_vol_frac_small", type=float, default=0.1)
    parser.add_argument("--tran_matrix_value", type=float, default=1.0)
    parser.add_argument("--tran_inclusion_value", type=float, default=1000.0)
    parser.add_argument("--tran_H_macro", type=float, default=None)
    parser.add_argument("--tran_H_meso_list", type=str, default="")

    # Scaling / normalisation
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="log_standardize",
        choices=["none", "minmax", "log_standardize", "affine_standardize"],
    )
    parser.add_argument("--scaling_epsilon", type=float, default=1e-6)

    # Held-out times
    parser.add_argument(
        "--held_out_indices",
        type=str,
        default="",
        help="Comma-separated 0-based indices of times to hold out.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Parse Tran meso list
    # ------------------------------------------------------------------
    tran_H_meso_list = None
    if args.tran_H_meso_list:
        tran_H_meso_list = []
        for h_str in args.tran_H_meso_list.split(","):
            h_str = h_str.strip()
            if not h_str:
                continue
            if h_str.upper().endswith("D"):
                factor = float(h_str[:-1])
                tran_H_meso_list.append(factor * args.tran_D_large)
            else:
                tran_H_meso_list.append(float(h_str))

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    if args.data_generator == "tran_inclusion":
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

    # Convert torch tensors → numpy
    marginal_arrays: Dict[float, np.ndarray] = {
        t: samples.cpu().numpy() for t, samples in marginal_data.items()
    }
    sorted_times = sorted(marginal_arrays.keys())

    # ------------------------------------------------------------------
    # Held-out handling
    # ------------------------------------------------------------------
    held_out_indices_list, held_out_times_list = parse_held_out_indices(
        args.held_out_indices, sorted_times
    )
    if held_out_indices_list:
        print(
            f"Held-out time indices: {held_out_indices_list}  "
            f"(times: {held_out_times_list})"
        )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    save_dict: Dict[str, object] = {
        "resolution": np.int32(args.resolution),
        "data_dim": np.int32(data_dim),
        "scale_mode": args.scale_mode,
        "scaling_epsilon": np.float32(args.scaling_epsilon),
        "data_generator": args.data_generator,
        "held_out_indices": np.array(held_out_indices_list, dtype=np.int32),
        "held_out_times": np.array(held_out_times_list, dtype=np.float32),
    }

    if args.scale_mode == "minmax":
        print(f"Applying global min-max scaling (eps={args.scaling_epsilon}).")
        marginal_arrays, data_min, data_scale = minmax_scale_marginals(
            marginal_arrays, eps=float(args.scaling_epsilon)
        )
        save_dict["minmax_data_min"] = np.float32(data_min)
        save_dict["minmax_data_scale"] = np.float32(data_scale)
        print(f"  Global data_min={data_min:.4f}, scale={data_scale:.4f}")

    elif args.scale_mode == "log_standardize":
        print(f"Applying log+global-standardisation (eps={args.scaling_epsilon}).")
        marginal_arrays, log_eps, log_mean, log_std = log_standardize_marginals(
            marginal_arrays, eps=float(args.scaling_epsilon)
        )
        save_dict["log_epsilon"] = np.float32(log_eps)
        save_dict["log_mean"] = np.float32(log_mean)
        save_dict["log_std"] = np.float32(log_std)
        print(f"  Global log-space mean={log_mean:.4f}, std={log_std:.4f}")
    elif args.scale_mode == "affine_standardize":
        if args.data_generator != "tran_inclusion":
            raise ValueError(
                "scale_mode='affine_standardize' is intended for bounded two-phase fields. "
                "Use it with --data_generator tran_inclusion."
            )
        delta = float(args.scaling_epsilon)
        data_min = float(min(args.tran_matrix_value, args.tran_inclusion_value))
        data_max = float(max(args.tran_matrix_value, args.tran_inclusion_value))
        print(
            "Applying affine→global-standardisation "
            f"(bounds=[{data_min}, {data_max}], delta={delta})."
        )
        marginal_arrays, affine_delta, affine_mean, affine_std, affine_min, affine_max = affine_standardize_marginals(
            marginal_arrays,
            data_min=data_min,
            data_max=data_max,
            delta=delta,
        )
        save_dict["affine_delta"] = np.float32(affine_delta)
        save_dict["affine_mean"] = np.float32(affine_mean)
        save_dict["affine_std"] = np.float32(affine_std)
        save_dict["affine_min"] = np.float32(affine_min)
        save_dict["affine_max"] = np.float32(affine_max)
        print(f"  Global affine-space mean={affine_mean:.4f}, std={affine_std:.4f}")

    # ------------------------------------------------------------------
    # Build grid coordinates
    # ------------------------------------------------------------------
    grid_coords = build_grid_coords(args.resolution)
    save_dict["grid_coords"] = grid_coords

    # ------------------------------------------------------------------
    # Normalise times to [0, 1]
    # ------------------------------------------------------------------
    times_arr = np.array(sorted_times, dtype=np.float32)
    if times_arr.max() - times_arr.min() > 1e-9:
        times_normalized = (times_arr - times_arr.min()) / (
            times_arr.max() - times_arr.min()
        )
    else:
        times_normalized = np.zeros_like(times_arr)
    save_dict["times"] = times_arr
    save_dict["times_normalized"] = times_normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # Save raw marginals
    # ------------------------------------------------------------------
    for t in sorted_times:
        arr = marginal_arrays[t]
        save_dict[f"raw_marginal_{t}"] = arr.astype(np.float32)

    np.savez(args.output_path, **save_dict)
    print(f"Saved FAE data to {args.output_path}")
    print(f"  Times: {sorted_times}")
    print(f"  Samples per time: {next(iter(marginal_arrays.values())).shape[0]}")
    print(f"  Spatial dim: {data_dim}  (resolution={args.resolution})")


if __name__ == "__main__":
    main()
