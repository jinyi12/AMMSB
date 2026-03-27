"""Encode a large corpus of microstructure fields into FAE latent codes.

Reads the corpus npz (produced by ``data/generate_large_corpus.py``) and a
pretrained FAE checkpoint, then encodes every sample at every training time
into latent codes.

The run contract may be:

- a legacy latent-MSBM run with ``args.txt`` and ``fae_latents.npz``
- a CSP run with ``config/args.json`` plus ``fae_latents.npz`` or
  ``fae_token_latents.npz``

Transformer-token FAE checkpoints are encoded through the maintained flattened
downstream transport surface, so the output archive stores ``(N, L*D)`` latents
for compatibility metrics.

Usage
-----
python scripts/fae/tran_evaluation/encode_corpus.py \
    --corpus_path data/fae_tran_inclusions_corpus.npz \
    --run_dir results/2026-02-01T23-00-12-38 \
    --output_path data/corpus_latents.npz \
    --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from mmsfm.fae.multiscale_dataset_naive import load_training_time_data_naive
from scripts.fae.tran_evaluation.run_support import parse_key_value_args_file as parse_args_file


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _parse_time_indices_arg(raw: str) -> np.ndarray | None:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return None
    return np.asarray([int(item) for item in values], dtype=np.int64)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    json_cfg_path = run_dir / "config" / "args.json"
    if json_cfg_path.exists():
        return json.loads(json_cfg_path.read_text())

    legacy_cfg_path = run_dir / "args.txt"
    if legacy_cfg_path.exists():
        return parse_args_file(legacy_cfg_path)

    raise FileNotFoundError(
        "Could not resolve a run contract from "
        f"{run_dir}. Expected either config/args.json or args.txt."
    )


def _load_time_indices_from_archive(archive_path: Path) -> np.ndarray:
    with np.load(archive_path, allow_pickle=True) as payload:
        if "time_indices" not in payload:
            raise KeyError(f"Missing 'time_indices' in {archive_path}")
        return np.asarray(payload["time_indices"], dtype=np.int64)


def _resolve_fae_checkpoint_path(
    *,
    run_dir: Path,
    run_cfg: dict[str, Any],
    override_path: str | None,
) -> Path:
    raw_path = override_path or run_cfg.get("fae_checkpoint")
    if raw_path in (None, "", "None"):
        raise ValueError(
            "Could not resolve an FAE checkpoint. Pass --fae_checkpoint or use a run_dir "
            "whose contract records fae_checkpoint."
        )
    checkpoint_path = _resolve_repo_path(str(raw_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_time_indices(
    *,
    run_dir: Path,
    run_cfg: dict[str, Any],
    time_indices_path_override: str | None,
    time_indices_override: str,
) -> np.ndarray:
    parsed_override = _parse_time_indices_arg(time_indices_override)
    if parsed_override is not None:
        return parsed_override

    if time_indices_path_override not in (None, "", "None"):
        archive_path = _resolve_repo_path(str(time_indices_path_override))
        if not archive_path.exists():
            raise FileNotFoundError(f"Time-index archive not found: {archive_path}")
        return _load_time_indices_from_archive(archive_path)

    candidate_paths: list[Path] = []
    for key in ("resolved_latents_path", "latents_path"):
        raw_path = run_cfg.get(key)
        if raw_path not in (None, "", "None"):
            candidate_paths.append(_resolve_repo_path(str(raw_path)))
    candidate_paths.extend([
        run_dir / "fae_latents.npz",
        run_dir / "fae_token_latents.npz",
    ])

    seen: set[Path] = set()
    for archive_path in candidate_paths:
        resolved = archive_path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            return _load_time_indices_from_archive(resolved)
        except KeyError:
            continue

    raise FileNotFoundError(
        "Could not resolve time indices for corpus encoding. Pass --time_indices, "
        "--time_indices_path, or use a run_dir whose latent archive records time_indices."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode corpus fields into FAE latent codes. Supports both legacy "
            "latent-MSBM runs and CSP runs, including transformer-token FAEs "
            "through flattened downstream transport latents."
        ),
    )
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus npz.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a latent-MSBM or CSP run directory that records the FAE contract.",
    )
    parser.add_argument(
        "--fae_checkpoint",
        type=str,
        default=None,
        help="Optional override for the FAE checkpoint path.",
    )
    parser.add_argument(
        "--time_indices_path",
        type=str,
        default=None,
        help="Optional override archive path that contains a time_indices array.",
    )
    parser.add_argument(
        "--time_indices",
        type=str,
        default="",
        help="Optional comma-separated dataset time indices to encode.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output path for latent codes.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    corpus_path = Path(args.corpus_path).expanduser().resolve()
    run_cfg = _load_run_config(run_dir)

    # ------------------------------------------------------------------
    # Load FAE checkpoint
    # ------------------------------------------------------------------
    fae_checkpoint_path = _resolve_fae_checkpoint_path(
        run_dir=run_dir,
        run_cfg=run_cfg,
        override_path=args.fae_checkpoint,
    )

    print(f"Loading FAE from {fae_checkpoint_path}...")
    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, fae_meta = build_fae_from_checkpoint(ckpt)
    encode_fn, _decode_fn = make_fae_apply_fns(autoencoder, fae_params, fae_batch_stats)
    print(f"  Latent dim: {fae_meta['latent_dim']}")

    # ------------------------------------------------------------------
    # Load time index mapping from MSBM training
    # ------------------------------------------------------------------
    time_indices = _resolve_time_indices(
        run_dir=run_dir,
        run_cfg=run_cfg,
        time_indices_path_override=args.time_indices_path,
        time_indices_override=args.time_indices,
    )
    print(f"Run time indices (dataset): {time_indices.tolist()}")

    # ------------------------------------------------------------------
    # Load corpus data at training times
    # ------------------------------------------------------------------
    print(f"Loading corpus from {corpus_path}...")
    corpus_time_data = load_training_time_data_naive(
        str(corpus_path),
        held_out_indices=None,  # uses held_out_indices from file
    )

    # Filter to only the time indices used by MSBM
    time_idx_set = set(time_indices.tolist())
    corpus_time_data = [d for d in corpus_time_data if d["idx"] in time_idx_set]
    corpus_time_data = sorted(corpus_time_data, key=lambda d: float(d["t_norm"]))

    print(f"  Loaded {len(corpus_time_data)} time marginals")
    for d in corpus_time_data:
        print(f"    idx={d['idx']}, t_norm={d['t_norm']:.4f}, n_samples={d['u'].shape[0]}")

    # ------------------------------------------------------------------
    # Encode all samples at each time
    # ------------------------------------------------------------------
    save_dict: dict[str, object] = {
        "time_indices": time_indices,
    }

    for d in corpus_time_data:
        idx = d["idx"]
        u_all = np.asarray(d["u"], dtype=np.float32)  # (N, P, 1)
        x = np.asarray(d["x"], dtype=np.float32)       # (P, 2)
        N = u_all.shape[0]

        print(f"Encoding time idx={idx} ({N} samples)...")
        parts: list[np.ndarray] = []
        for i in range(0, N, args.batch_size):
            u_b = u_all[i : i + args.batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            parts.append(encode_fn(u_b, x_b))
            if (i // args.batch_size) % 100 == 0 and i > 0:
                print(f"  {i}/{N}")

        z = np.concatenate(parts, axis=0)  # (N, K)
        save_dict[f"latents_{idx}"] = z.astype(np.float32)
        print(f"  -> shape {z.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else corpus_path.parent / "corpus_latents.npz"
    )

    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved corpus latents to {output_path}")


if __name__ == "__main__":
    main()
