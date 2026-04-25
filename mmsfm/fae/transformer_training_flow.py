"""Transformer-token FAE training flow with dedicated data ingestion."""

from __future__ import annotations

import argparse
from typing import Callable, Optional, Sequence

from mmsfm.fae.fae_training_components import (
    load_dataset_metadata,
    parse_float_list_arg,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
    parse_int_list_arg,
)
from mmsfm.fae.fae_training_runtime import (
    BuildAutoencoderFn,
    SetupFn,
    run_training_from_datasets,
)
from mmsfm.fae.multiscale_dataset_naive import MultiscaleFieldDatasetNaive
from mmsfm.fae.single_scale_dataset import (
    SingleScaleFieldDataset,
    load_single_scale_metadata,
)


def _uses_patchified_transformer(args: argparse.Namespace) -> bool:
    return getattr(args, "transformer_tokenization", "points") == "patches"


def run_transformer_training(
    args: argparse.Namespace,
    *,
    build_autoencoder_fn: BuildAutoencoderFn,
    architecture_name: str,
    wandb_name_prefix: str,
    wandb_tags: Sequence[str] = (),
    reconstruct_fn: Optional[Callable] = None,
    setup_fn: Optional[SetupFn] = None,
) -> dict:
    """Run the transformer-token FAE path with dedicated dataset handling."""
    print("\nLoading transformer dataset path ...")
    held_out_indices: Optional[list[int]] = None
    dataset_resolution: Optional[int] = None
    transformer_patch_mode = _uses_patchified_transformer(args)
    encoder_full_grid = bool(transformer_patch_mode)

    encoder_n_points = args.encoder_n_points if args.encoder_n_points > 0 else None
    decoder_n_points = args.decoder_n_points if args.decoder_n_points > 0 else None
    encoder_n_points_by_time = (
        parse_int_list_arg(args.encoder_n_points_by_time)
        if args.encoder_n_points_by_time
        else None
    )
    decoder_n_points_by_time = (
        parse_int_list_arg(args.decoder_n_points_by_time)
        if args.decoder_n_points_by_time
        else None
    )
    encoder_point_ratio_by_time = (
        parse_float_list_arg(args.encoder_point_ratio_by_time)
        if args.encoder_point_ratio_by_time
        else None
    )
    decoder_point_ratio_by_time = (
        parse_float_list_arg(args.decoder_point_ratio_by_time)
        if args.decoder_point_ratio_by_time
        else None
    )
    train_masking_strategy = args.masking_strategy
    eval_masking_strategy = (
        args.masking_strategy
        if args.eval_masking_strategy == "same"
        else args.eval_masking_strategy
    )
    return_decoder_gradients = getattr(args, "loss_type", "l2") == "sobolev_h1"

    if transformer_patch_mode:
        encoder_n_points = None
        encoder_n_points_by_time = None
        encoder_point_ratio_by_time = None
        print(
            "  Transformer tokenization: patches "
            "(full-grid encoder inputs with coordinate-query decoding; "
            "masking controls decoder query selection only)"
        )
    else:
        print("  Transformer tokenization: points (point-token encoder + coordinate-query decoder)")

    if args.training_mode == "single_scale":
        meta = load_single_scale_metadata(args.data_path)
        dataset_resolution = (
            int(meta["resolution"]) if meta.get("resolution") is not None else None
        )
        print(f"Dataset: {args.data_path}")
        print(f"  Resolution: {meta.get('resolution')}")
        print(f"  Available times: {meta.get('n_times')}")
        print(f"  Single-scale index: {args.single_scale_index}")

        train_dataset = SingleScaleFieldDataset(
            npz_path=args.data_path,
            time_index=args.single_scale_index,
            train=True,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            masking_strategy=train_masking_strategy,
            return_decoder_gradients=return_decoder_gradients,
            encoder_full_grid=encoder_full_grid,
        )
        test_dataset = SingleScaleFieldDataset(
            npz_path=args.data_path,
            time_index=args.single_scale_index,
            train=False,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            masking_strategy=eval_masking_strategy,
            return_decoder_gradients=return_decoder_gradients,
            encoder_full_grid=encoder_full_grid,
        )
    else:
        dataset_meta = load_dataset_metadata(args.data_path)
        dataset_resolution = (
            int(dataset_meta["resolution"])
            if dataset_meta.get("resolution") is not None
            else None
        )
        print(f"Dataset: {args.data_path}")
        print(f"  Resolution: {dataset_meta.get('resolution')}")
        print(f"  Samples: {dataset_meta.get('n_samples')}")
        print(f"  Times: {dataset_meta.get('n_times')}")

        if args.held_out_indices:
            held_out_indices = parse_held_out_indices_arg(args.held_out_indices)
        elif args.held_out_times:
            if dataset_meta.get("times_normalized") is None:
                raise ValueError("--held-out-times requires times_normalized in dataset.")
            held_out_indices = parse_held_out_times_arg(
                args.held_out_times,
                dataset_meta["times_normalized"],
            )

        train_dataset = MultiscaleFieldDatasetNaive(
            npz_path=args.data_path,
            train=True,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            encoder_point_ratio_by_time=encoder_point_ratio_by_time,
            decoder_point_ratio_by_time=decoder_point_ratio_by_time,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            encoder_n_points_by_time=encoder_n_points_by_time,
            decoder_n_points_by_time=decoder_n_points_by_time,
            masking_strategy=train_masking_strategy,
            held_out_indices=held_out_indices,
            return_decoder_gradients=return_decoder_gradients,
            encoder_full_grid=encoder_full_grid,
        )
        test_dataset = MultiscaleFieldDatasetNaive(
            npz_path=args.data_path,
            train=False,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            encoder_point_ratio_by_time=encoder_point_ratio_by_time,
            decoder_point_ratio_by_time=decoder_point_ratio_by_time,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            encoder_n_points_by_time=encoder_n_points_by_time,
            decoder_n_points_by_time=decoder_n_points_by_time,
            masking_strategy=eval_masking_strategy,
            held_out_indices=held_out_indices,
            return_decoder_gradients=return_decoder_gradients,
            encoder_full_grid=encoder_full_grid,
        )

    if transformer_patch_mode:
        if dataset_resolution is None:
            raise ValueError(
                "Transformer patch tokenization requires dataset resolution metadata."
            )
        setattr(
            args,
            "transformer_grid_size",
            (int(dataset_resolution), int(dataset_resolution)),
        )

    use_time_grouped_batches = (
        args.training_mode == "multi_scale"
        and (
            getattr(args, "loss_type", "l2") == "ntk_scaled"
            or bool(args.decoder_n_points_by_time)
            or bool(args.decoder_point_ratio_by_time)
            or (
                not transformer_patch_mode
                and (
                    bool(args.encoder_n_points_by_time)
                    or bool(args.encoder_point_ratio_by_time)
                )
            )
        )
    )

    return run_training_from_datasets(
        args,
        build_autoencoder_fn=build_autoencoder_fn,
        architecture_name=architecture_name,
        wandb_name_prefix=wandb_name_prefix,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        dataset_resolution=dataset_resolution,
        held_out_indices=held_out_indices,
        use_time_grouped_batches=use_time_grouped_batches,
        wandb_tags=wandb_tags,
        reconstruct_fn=reconstruct_fn,
        setup_fn=setup_fn,
    )
