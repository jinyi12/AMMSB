"""Encode a large corpus of microstructure fields into FAE latent codes.

Reads the corpus npz (produced by ``data/generate_large_corpus.py``) and a
pretrained FAE checkpoint, then encodes every sample at every training time
into latent codes.

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
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.fae_latent_utils import (
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive
from scripts.fae.tran_evaluation.run_support import parse_key_value_args_file as parse_args_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode corpus fields into FAE latent codes.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus npz.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to MSBM training run directory.")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for latent codes.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    corpus_path = Path(args.corpus_path)

    # ------------------------------------------------------------------
    # Load FAE checkpoint
    # ------------------------------------------------------------------
    train_cfg = parse_args_file(run_dir / "args.txt")
    fae_checkpoint_path = Path(train_cfg["fae_checkpoint"])
    if not fae_checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {fae_checkpoint_path}")

    print(f"Loading FAE from {fae_checkpoint_path}...")
    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, fae_meta = build_attention_fae_from_checkpoint(ckpt)
    encode_fn, decode_fn = make_fae_apply_fns(autoencoder, fae_params, fae_batch_stats)
    print(f"  Latent dim: {fae_meta['latent_dim']}")

    # ------------------------------------------------------------------
    # Load time index mapping from MSBM training
    # ------------------------------------------------------------------
    lat_npz = np.load(run_dir / "fae_latents.npz", allow_pickle=True)
    time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64)
    lat_npz.close()
    print(f"MSBM time indices (dataset): {time_indices.tolist()}")

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
    output_path = args.output_path
    if output_path is None:
        output_path = str(corpus_path.parent / "corpus_latents.npz")

    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved corpus latents to {output_path}")


if __name__ == "__main__":
    main()
