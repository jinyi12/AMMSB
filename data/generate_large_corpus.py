"""Generate a large corpus of microstructure fields for conditional evaluation.

Generates N_corpus samples using the same data-generation procedure and
transform parameters as an existing reference dataset, so that latent codes
are directly comparable.

Usage
-----
python data/generate_large_corpus.py \
    --reference_dataset data/fae_tran_inclusions.npz \
    --n_samples 50000 \
    --batch_size 1000 \
    --output_path data/fae_tran_inclusions_corpus.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.multimarginal_generation import (
    generate_tran_multiscale_inclusion_data,
)
from data.generate_fae_data import build_grid_coords


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a large corpus of microstructure fields for conditional evaluation."
    )
    parser.add_argument(
        "--reference_dataset",
        type=str,
        required=True,
        help="Path to the reference dataset npz (e.g., data/fae_tran_inclusions.npz). "
        "Transform parameters and generation settings are read from this file.",
    )
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Generate in batches to manage memory.",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    # Allow overriding generation parameters (defaults read from reference)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--L_domain", type=float, default=None)
    parser.add_argument("--tran_D_large", type=float, default=None)
    parser.add_argument("--tran_vol_frac_large", type=float, default=None)
    parser.add_argument("--tran_vol_frac_small", type=float, default=None)
    parser.add_argument("--tran_matrix_value", type=float, default=None)
    parser.add_argument("--tran_inclusion_value", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load reference dataset metadata
    # ------------------------------------------------------------------
    ref_path = Path(args.reference_dataset)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {ref_path}")

    ref = np.load(ref_path, allow_pickle=True)

    resolution = args.resolution or int(ref["resolution"])
    scale_mode = str(ref["scale_mode"])
    data_generator = str(ref["data_generator"])

    if data_generator != "tran_inclusion":
        raise ValueError(
            f"This script only supports tran_inclusion datasets, got: {data_generator}"
        )

    # Read transform parameters from reference (we'll reuse them exactly)
    if scale_mode == "log_standardize":
        ref_log_epsilon = float(ref["log_epsilon"])
        ref_log_mean = float(ref["log_mean"])
        ref_log_std = float(ref["log_std"])
        ref_minmax_data_min = None
        ref_minmax_data_scale = None
        ref_affine_min = None
        ref_affine_max = None
        ref_affine_delta = None
        ref_affine_mean = None
        ref_affine_std = None
    elif scale_mode == "minmax":
        ref_minmax_data_min = float(ref["minmax_data_min"])
        ref_minmax_data_scale = float(ref["minmax_data_scale"])
        ref_log_epsilon = None
        ref_log_mean = None
        ref_log_std = None
        ref_affine_min = None
        ref_affine_max = None
        ref_affine_delta = None
        ref_affine_mean = None
        ref_affine_std = None
    elif scale_mode == "affine_standardize":
        ref_minmax_data_min = None
        ref_minmax_data_scale = None
        ref_log_epsilon = None
        ref_log_mean = None
        ref_log_std = None
        ref_affine_delta = float(ref.get("affine_delta", 0.0))
        ref_affine_mean = float(ref["affine_mean"])
        ref_affine_std = float(ref["affine_std"])
        ref_affine_min = float(ref["affine_min"])
        ref_affine_max = float(ref["affine_max"])
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

    # Read generation parameters from reference times to infer H schedule
    ref_times = ref["times"].astype(np.float32)
    ref_held_out = [int(i) for i in ref["held_out_indices"]]
    ref.close()

    # Infer H schedule from reference dataset args (if available) or use defaults
    # We need the same H_meso_list and H_macro as the reference
    # Try to parse from an args.txt next to the dataset, or use defaults
    L_domain = args.L_domain or 6.0
    D_large = args.tran_D_large or 1.0
    vol_frac_large = args.tran_vol_frac_large or 0.2
    vol_frac_small = args.tran_vol_frac_small or 0.1
    if scale_mode == "affine_standardize" and ref_affine_min is not None and ref_affine_max is not None:
        matrix_default = ref_affine_min
        inclusion_default = ref_affine_max
    else:
        matrix_default = 1.0
        inclusion_default = 1000.0

    matrix_value = float(args.tran_matrix_value) if args.tran_matrix_value is not None else float(matrix_default)
    inclusion_value = float(args.tran_inclusion_value) if args.tran_inclusion_value is not None else float(inclusion_default)

    # Infer H_meso_list from the number of intermediate times
    # Reference times: [0.0, t1, t2, ..., 1.0] where intermediates are meso scales
    n_meso = len(ref_times) - 2  # exclude t=0 (micro) and t=1 (macro)
    # Default meso schedule from generate_fae_data.py
    default_meso = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    if n_meso <= len(default_meso):
        H_meso_list = [h * D_large for h in default_meso[:n_meso]]
    else:
        H_meso_list = [h * D_large for h in default_meso]

    H_macro = L_domain  # consistent with default

    print(f"Reference dataset: {ref_path}")
    print(f"  scale_mode={scale_mode}, resolution={resolution}")
    if scale_mode == "log_standardize":
        print(
            f"  ref_log_epsilon={ref_log_epsilon}, "
            f"ref_log_mean={ref_log_mean:.4f}, ref_log_std={ref_log_std:.4f}"
        )
    elif scale_mode == "minmax":
        print(
            f"  ref_minmax_data_min={ref_minmax_data_min}, "
            f"ref_minmax_data_scale={ref_minmax_data_scale}"
        )
    else:
        print(
            f"  ref_affine_min={ref_affine_min}, ref_affine_max={ref_affine_max}, "
            f"ref_affine_delta={ref_affine_delta}, ref_affine_mean={ref_affine_mean:.4f}, ref_affine_std={ref_affine_std:.4f}"
        )
    print(f"  H_meso_list={H_meso_list}, H_macro={H_macro}")
    print(f"  Generating {args.n_samples} samples in batches of {args.batch_size}")

    # ------------------------------------------------------------------
    # Generate in batches
    # ------------------------------------------------------------------
    n_total = args.n_samples
    batch_size = args.batch_size

    # Collect all marginal arrays per time key
    all_marginals: dict[float, list[np.ndarray]] = {}

    n_generated = 0
    batch_idx = 0
    while n_generated < n_total:
        n_batch = min(batch_size, n_total - n_generated)
        print(f"\nBatch {batch_idx}: generating {n_batch} samples ({n_generated}/{n_total})...")

        marginal_data, data_dim = generate_tran_multiscale_inclusion_data(
            N_samples=n_batch,
            resolution=resolution,
            L_domain=L_domain,
            D_large=D_large,
            vol_frac_large=vol_frac_large,
            vol_frac_small=vol_frac_small,
            matrix_value=matrix_value,
            inclusion_value=inclusion_value,
            H_meso_list=H_meso_list,
            H_macro=H_macro,
            device=args.device,
        )

        for t, tensor in marginal_data.items():
            arr = tensor.cpu().numpy().astype(np.float32)
            if t not in all_marginals:
                all_marginals[t] = []
            all_marginals[t].append(arr)

        n_generated += n_batch
        batch_idx += 1

    # Concatenate batches
    marginal_arrays: dict[float, np.ndarray] = {
        t: np.concatenate(arrs, axis=0) for t, arrs in all_marginals.items()
    }
    sorted_times = sorted(marginal_arrays.keys())
    print(f"\nGenerated {n_generated} samples across {len(sorted_times)} time steps.")

    # ------------------------------------------------------------------
    # Apply transform with REFERENCE parameters
    # ------------------------------------------------------------------
    print(f"Applying {scale_mode} with reference transform parameters...")
    scaled: dict[float, np.ndarray] = {}
    for t, arr in marginal_arrays.items():
        if scale_mode == "log_standardize":
            log_arr = np.log(arr + float(ref_log_epsilon))
            standardized = (log_arr - float(ref_log_mean)) / float(ref_log_std)
            scaled[t] = standardized.astype(np.float32)
        elif scale_mode == "minmax":
            standardized = ((arr - float(ref_minmax_data_min)) / float(ref_minmax_data_scale))
            scaled[t] = standardized.astype(np.float32)
        else:
            scale = float(ref_affine_max - ref_affine_min)
            p = (arr - float(ref_affine_min)) / scale
            if float(ref_affine_delta) > 0.0:
                p = np.clip(p, float(ref_affine_delta), 1.0 - float(ref_affine_delta))
            standardized = (p - float(ref_affine_mean)) / float(ref_affine_std)
            scaled[t] = standardized.astype(np.float32)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    grid_coords = build_grid_coords(resolution)

    times_arr = np.array(sorted_times, dtype=np.float32)
    if times_arr.max() - times_arr.min() > 1e-9:
        times_normalized = (times_arr - times_arr.min()) / (times_arr.max() - times_arr.min())
    else:
        times_normalized = np.zeros_like(times_arr)

    save_dict: dict[str, object] = {
        "resolution": np.int32(resolution),
        "data_dim": np.int32(data_dim),
        "scale_mode": scale_mode,
        "data_generator": data_generator,
        "held_out_indices": np.array(ref_held_out, dtype=np.int32),
        "held_out_times": np.array([], dtype=np.float32),
        "grid_coords": grid_coords,
        "times": times_arr,
        "times_normalized": times_normalized.astype(np.float32),
    }
    if scale_mode == "log_standardize":
        save_dict.update({
            "log_epsilon": np.float32(ref_log_epsilon),
            "log_mean": np.float32(ref_log_mean),
            "log_std": np.float32(ref_log_std),
        })
    elif scale_mode == "minmax":
        save_dict.update({
            "minmax_data_min": np.float32(ref_minmax_data_min),
            "minmax_data_scale": np.float32(ref_minmax_data_scale),
        })
    else:
        save_dict.update({
            "affine_delta": np.float32(ref_affine_delta),
            "affine_mean": np.float32(ref_affine_mean),
            "affine_std": np.float32(ref_affine_std),
            "affine_min": np.float32(ref_affine_min),
            "affine_max": np.float32(ref_affine_max),
        })

    for t in sorted_times:
        save_dict[f"raw_marginal_{t}"] = scaled[t]

    output_path = args.output_path
    if output_path is None:
        stem = ref_path.stem
        output_path = str(ref_path.parent / f"{stem}_corpus.npz")

    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved corpus to {output_path}")
    print(f"  Times: {sorted_times}")
    print(f"  Samples per time: {n_generated}")
    print(f"  Spatial dim: {data_dim} (resolution={resolution})")


if __name__ == "__main__":
    main()
