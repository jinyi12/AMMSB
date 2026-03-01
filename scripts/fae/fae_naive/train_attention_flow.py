"""Shared training/evaluation flow for attention-style FAE scripts."""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from functional_autoencoders.datasets import NumpyLoader
from functional_autoencoders.domains.off_grid import RandomlySampledEuclidean
from functional_autoencoders.losses.fae import get_loss_fae_fn
from scripts.fae.fae_naive.single_scale_dataset import (
    SingleScaleFieldDataset,
    load_single_scale_metadata,
)
from scripts.fae.fae_naive.sobolev_losses import get_sobolev_h1_loss_fn
from scripts.fae.fae_naive.train_attention_components import (
    MSEMetricNaive,
    evaluate_at_times,
    evaluate_train_reconstruction,
    load_dataset_metadata,
    parse_float_list_arg,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
    parse_int_list_arg,
    save_model_artifact,
    save_model_info,
    setup_output_directory,
    visualize_reconstructions_all_times,
    visualize_sample_reconstructions,
)
from scripts.fae.multiscale_dataset_naive import (
    MultiscaleFieldDatasetNaive,
    TimeGroupedBatchSampler,
    load_held_out_data_naive,
    load_training_time_data_naive,
)
from scripts.fae.wandb_trainer import WandbAutoencoderTrainer

BuildAutoencoderFn = Callable[
    [jax.Array, argparse.Namespace, tuple[int, ...]],
    tuple[object, dict],
]
SetupFn = Callable[
    [object, argparse.Namespace],
    tuple[Callable, list, Optional[Callable]],
]
"""Return (loss_fn, metrics, reconstruct_fn) or
(loss_fn, metrics, reconstruct_fn, extra_init_params_fn)."""


def _estimate_sobolev_balance_from_dataset(
    dataset,
    *,
    n_items: int = 16,
    seed: int = 0,
):
    """Estimate E[u^2] / E[|du|^2] from sampled dataset items."""
    if n_items <= 0:
        return None

    rng = np.random.default_rng(int(seed))
    n_total = len(dataset)
    if n_total <= 0:
        return None

    indices = rng.integers(0, n_total, size=min(int(n_items), n_total)).tolist()
    u2_vals: list[float] = []
    g2_vals: list[float] = []
    for idx in indices:
        item = dataset[int(idx)]
        if not (isinstance(item, tuple) or isinstance(item, list)) or len(item) < 5:
            continue
        u_dec, _x_dec, _u_enc, _x_enc, du_dec = item[:5]
        u_dec = np.asarray(u_dec, dtype=np.float32)
        du_dec = np.asarray(du_dec, dtype=np.float32)
        if u_dec.size == 0 or du_dec.size == 0:
            continue
        u2_vals.append(float(np.mean(u_dec**2)))
        g2_vals.append(float(np.mean(np.sum(du_dec**2, axis=-1))))

    if not u2_vals or not g2_vals:
        return None

    u2 = float(np.mean(u2_vals))
    g2 = float(np.mean(g2_vals))
    ratio = u2 / (g2 + 1e-12)
    return {"u2": u2, "g2": g2, "ratio": ratio}


def _compute_collapse_diagnostics(
    autoencoder,
    state,
    dataloader,
    *,
    n_batches: int,
    seed: int = 0,
) -> dict:
    key = jax.random.PRNGKey(int(seed))
    full_mse_vals: list[float] = []
    zero_mse_vals: list[float] = []
    sens_vals: list[float] = []
    zvar_vals: list[float] = []

    for i, batch in enumerate(dataloader):
        if i >= int(n_batches):
            break
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_dec = jnp.array(u_dec)
        x_dec = jnp.array(x_dec)
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)

        z = autoencoder.encode(state, u_enc, x_enc, train=False)
        u_hat = autoencoder.decode(state, z, x_dec, train=False)
        full_mse_vals.append(float(jnp.mean((u_hat - u_dec) ** 2)))

        z0 = jnp.zeros_like(z)
        u_hat0 = autoencoder.decode(state, z0, x_dec, train=False)
        zero_mse_vals.append(float(jnp.mean((u_hat0 - u_dec) ** 2)))

        key, subkey = jax.random.split(key)
        z_shuf = jax.random.permutation(subkey, z, axis=0)
        u_hat_shuf = autoencoder.decode(state, z_shuf, x_dec, train=False)
        sens_vals.append(float(jnp.mean((u_hat - u_hat_shuf) ** 2)))

        zvar_vals.append(float(jnp.mean(jnp.var(z, axis=0))))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "full_mse": _mean(full_mse_vals),
        "zero_latent_mse": _mean(zero_mse_vals),
        "decode_sensitivity": _mean(sens_vals),
        "latent_var_mean": _mean(zvar_vals),
        "n_batches": int(min(int(n_batches), len(full_mse_vals))),
    }


def _build_wandb_tags(
    args: argparse.Namespace,
    extra_tags: Sequence[str],
) -> list[str]:
    tags = ["naive", getattr(args, "training_mode", "multi_scale")]
    if hasattr(args, "pooling_type"):
        tags.append(str(args.pooling_type))
    if hasattr(args, "decoder_type"):
        tags.append(str(args.decoder_type))
    if hasattr(args, "optimizer"):
        tags.append(str(args.optimizer))
    if getattr(args, "loss_type", "l2") != "l2":
        tags.append(str(args.loss_type))
    tags.extend(list(extra_tags))
    return list(dict.fromkeys(tags))


def run_training(
    args: argparse.Namespace,
    *,
    build_autoencoder_fn: BuildAutoencoderFn,
    architecture_name: str,
    wandb_name_prefix: str,
    wandb_tags: Sequence[str] = (),
    reconstruct_fn: Optional[Callable] = None,
    setup_fn: Optional[SetupFn] = None,
) -> dict:
    """Run the full train/eval/artifact flow shared by attention scripts."""
    decoder_features = tuple(int(x.strip()) for x in args.decoder_features.split(",") if x.strip())
    if not decoder_features:
        raise ValueError("--decoder-features must contain at least one layer width.")

    wandb_run = None
    wandb_run_id = None
    if not getattr(args, "wandb_disabled", False) and HAS_WANDB:
        default_wandb_name = (
            f"{wandb_name_prefix}_{args.training_mode}_{args.decoder_type}_{args.optimizer}"
        )
        wandb_name = args.wandb_name or default_wandb_name

        config = vars(args).copy()
        config["decoder_features_tuple"] = list(decoder_features)
        config["architecture"] = architecture_name

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            tags=_build_wandb_tags(args, wandb_tags),
        )
        wandb_run = wandb.run
        wandb_run_id = wandb_run.id if wandb_run else None

    paths = setup_output_directory(
        base_dir=args.output_dir,
        run_name=args.run_name,
        wandb_run_id=wandb_run_id,
    )
    print(f"Output directory: {paths['root']}")

    if wandb_run is not None:
        wandb_run.config.update({"output_dir": paths["root"]}, allow_val_change=True)

    with open(os.path.join(paths["root"], "args.json"), "w") as file:
        json.dump(vars(args), file, indent=2)

    key = jax.random.PRNGKey(args.seed)

    print("\n" + "=" * 70)
    print(f"ARCHITECTURE: {architecture_name}")
    print("=" * 70)
    print(f"  Training mode: {args.training_mode}")
    if hasattr(args, "encoder_multiscale_sigmas") and args.encoder_multiscale_sigmas:
        print(f"  Encoder multiscale sigmas: {args.encoder_multiscale_sigmas}")
    if hasattr(args, "decoder_type"):
        print(f"  Decoder type: {args.decoder_type}")
    if hasattr(args, "decoder_multiscale_sigmas") and args.decoder_multiscale_sigmas:
        print(f"  Decoder multiscale sigmas: {args.decoder_multiscale_sigmas}")
    if hasattr(args, "pooling_type"):
        print(f"  Pooling: {args.pooling_type}")
    print(f"  Optimizer: {args.optimizer}")
    if args.optimizer in {"adamw", "muon"}:
        print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(
            "  Muon beta: "
            f"{args.muon_beta}, ns_steps: {args.muon_ns_steps}, adaptive: {args.muon_adaptive}"
        )
    loss_type = getattr(args, "loss_type", "custom")
    print(f"  Loss type: {loss_type}")
    print("=" * 70 + "\n")

    print("\nLoading dataset ...")
    held_out_indices: Optional[list[int]] = None
    dataset_resolution: Optional[int] = None
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
    return_decoder_gradients = getattr(args, "loss_type", "l2") == "sobolev_h1"

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
            masking_strategy=args.masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            return_decoder_gradients=return_decoder_gradients,
        )
        eval_masking_strategy = (
            args.masking_strategy
            if args.eval_masking_strategy == "same"
            else args.eval_masking_strategy
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
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            return_decoder_gradients=return_decoder_gradients,
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
            masking_strategy=args.masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            held_out_indices=held_out_indices,
            return_decoder_gradients=return_decoder_gradients,
        )
        eval_masking_strategy = (
            args.masking_strategy
            if args.eval_masking_strategy == "same"
            else args.eval_masking_strategy
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
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            held_out_indices=held_out_indices,
            return_decoder_gradients=return_decoder_gradients,
        )

    use_time_grouped_batches = (
        args.training_mode == "multi_scale"
        and (
            getattr(args, "loss_type", "l2") == "ntk_scaled"
            or bool(args.encoder_n_points_by_time)
            or bool(args.decoder_n_points_by_time)
            or bool(args.encoder_point_ratio_by_time)
            or bool(args.decoder_point_ratio_by_time)
        )
    )
    if use_time_grouped_batches:
        train_loader = NumpyLoader(
            train_dataset,
            batch_sampler=TimeGroupedBatchSampler(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            ),
        )
        test_loader = NumpyLoader(
            test_dataset,
            batch_sampler=TimeGroupedBatchSampler(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                seed=args.seed,
            ),
        )
    else:
        train_loader = NumpyLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = NumpyLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )

    print(f"  Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

    n_loss_terms = 1
    if args.training_mode == "multi_scale" and hasattr(train_dataset, "n_times"):
        n_loss_terms = max(1, int(train_dataset.n_times))
    setattr(args, "ntk_n_loss_terms", int(n_loss_terms))

    if getattr(args, "loss_type", "l2") == "sobolev_h1":
        est = _estimate_sobolev_balance_from_dataset(
            train_dataset,
            n_items=16,
            seed=args.seed,
        )
        if est is not None:
            ratio = float(est["ratio"])
            if ratio > 0.0 and np.isfinite(ratio):
                dominance = float(args.lambda_grad) / ratio if ratio > 0 else float("inf")
                if dominance >= 100.0:
                    warnings.warn(
                        "Sobolev loss scale check: estimated E[u^2]/E[|∇u|^2] "
                        f"≈ {ratio:.3e} from dataset gradients, but --lambda-grad={args.lambda_grad:.3e}. "
                        f"This makes the gradient term dominate by ~{dominance:.1e}x and can stall training. "
                        f"Try --lambda-grad≈{ratio:.3e} (or start smaller and ramp up).",
                        UserWarning,
                    )

    key, subkey = jax.random.split(key)
    autoencoder, architecture_info = build_autoencoder_fn(subkey, args, decoder_features)
    save_model_info(paths, architecture_info, args)

    extra_init_params_fn = None
    if setup_fn is not None:
        setup_result = setup_fn(autoencoder, args)
        loss_fn, metrics, reconstruct_fn = setup_result[:3]
        if len(setup_result) >= 4:
            extra_init_params_fn = setup_result[3]
    else:
        latent_noise_scale = getattr(args, "latent_noise_scale", 0.0) or 0.0
        domain = RandomlySampledEuclidean(s=0.0)
        if args.loss_type == "l2":
            if latent_noise_scale > 0.0:
                from scripts.fae.fae_naive.noisy_fae_loss import get_loss_fae_with_noise_fn

                loss_fn = get_loss_fae_with_noise_fn(
                    autoencoder, domain, beta=args.beta,
                    latent_noise_scale=latent_noise_scale,
                )
            else:
                loss_fn = get_loss_fae_fn(autoencoder, domain, beta=args.beta)
        elif args.loss_type == "sobolev_h1":
            loss_fn = get_sobolev_h1_loss_fn(
                autoencoder=autoencoder,
                beta=args.beta,
                lambda_grad=args.lambda_grad,
                grad_method=args.sobolev_grad_method,
                fd_eps=args.sobolev_fd_eps,
                fd_periodic=args.sobolev_fd_periodic,
                resolution=dataset_resolution,
                subtract_data_norm=args.sobolev_subtract_data_norm,
                latent_noise_scale=latent_noise_scale,
            )
        elif args.loss_type == "ntk_scaled":
            from scripts.fae.fae_naive.ntk_losses import (
                NTKDiagnosticMetric,
                get_ntk_scaled_loss_fn,
            )

            loss_fn = get_ntk_scaled_loss_fn(
                autoencoder=autoencoder,
                beta=args.beta,
                scale_norm=args.ntk_scale_norm,
                epsilon=args.ntk_epsilon,
                estimate_total_trace=bool(args.ntk_estimate_total_trace),
                total_trace_ema_decay=float(args.ntk_total_trace_ema_decay),
                n_loss_terms=n_loss_terms,
                latent_noise_scale=latent_noise_scale,
                calibration_interval=int(args.ntk_calibration_interval),
                cv_threshold=float(args.ntk_cv_threshold),
                diag_subsample=int(args.ntk_diag_subsample),
            )
        else:
            raise ValueError(f"Unsupported --loss-type={args.loss_type}")

        metrics = [MSEMetricNaive(autoencoder, domain)]
        if args.loss_type == "ntk_scaled":
            metrics.append(
                NTKDiagnosticMetric(
                    autoencoder,
                    scale_norm=args.ntk_scale_norm,
                    epsilon=args.ntk_epsilon,
                    n_batches=1,
                    estimate_total_trace=bool(args.ntk_estimate_total_trace),
                    n_loss_terms=n_loss_terms,
                    cv_threshold=float(args.ntk_cv_threshold),
                    diag_subsample=int(args.ntk_diag_subsample),
                    latent_noise_scale=latent_noise_scale,
                )
            )

    vis_callback = None
    if wandb_run is not None:

        def vis_callback(state, epoch):
            return visualize_sample_reconstructions(
                autoencoder,
                state,
                test_loader,
                n_samples=args.n_vis_samples,
                n_batches=1,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + int(epoch) + 1000),
            )

    best_model_path = (
        os.path.join(paths["checkpoints"], "best_state.pkl")
        if args.save_best_model
        else None
    )
    optimizer_config = {
        "name": args.optimizer,
        "weight_decay": args.weight_decay,
        "muon_beta": args.muon_beta,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_adaptive": args.muon_adaptive,
    }

    trainer = WandbAutoencoderTrainer(
        autoencoder=autoencoder,
        loss_fn=loss_fn,
        metrics=metrics,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        wandb_run=wandb_run,
        vis_callback=vis_callback,
        vis_interval=args.vis_interval,
        save_best_model=args.save_best_model,
        best_model_path=best_model_path,
        optimizer_config=optimizer_config,
        extra_init_params_fn=extra_init_params_fn,
    )

    print("\nStarting training ...")
    key, subkey = jax.random.split(key)
    result = trainer.fit(
        key=subkey,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_factor=args.lr_decay_factor,
        max_step=args.max_steps,
        eval_interval=args.eval_interval,
        verbose="full",
    )

    state = result["state"]
    training_loss = result["training_loss_history"]
    np.save(os.path.join(paths["logs"], "training_loss.npy"), np.array(training_loss, dtype=np.float32))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"FAE Training Loss ({args.training_mode}, {args.loss_type}, {args.optimizer})"
    )
    fig.tight_layout()
    loss_plot_path = os.path.join(paths["figures"], "training_loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    if wandb_run is not None:
        wandb_run.log({"plots/training_loss": wandb.Image(loss_plot_path)})

    print("\nEvaluating reconstruction on test split ...")
    test_metrics = evaluate_train_reconstruction(
        autoencoder,
        state,
        test_loader,
        n_batches=args.eval_n_batches,
        reconstruct_fn=reconstruct_fn,
        key=jax.random.PRNGKey(args.seed + 2000),
        progress_every_batches=10,
    )
    print(f"  Test-split MSE: {test_metrics['mse']:.6f}")
    print(f"  Test-split Rel-MSE: {test_metrics['rel_mse']:.6f}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "final/test_mse": test_metrics["mse"],
                "final/test_rel_mse": test_metrics["rel_mse"],
            }
        )

    collapse_diag = None
    if setup_fn is None:
        print("\nComputing collapse diagnostics ...")
        collapse_diag = _compute_collapse_diagnostics(
            autoencoder,
            state,
            test_loader,
            n_batches=max(1, min(int(args.eval_n_batches), 10)),
            seed=args.seed + 123,
        )
        print(
            "  Diagnostics: "
            f"full_mse={collapse_diag['full_mse']:.6f}, "
            f"zero_latent_mse={collapse_diag['zero_latent_mse']:.6f}, "
            f"decode_sensitivity={collapse_diag['decode_sensitivity']:.6e}, "
            f"latent_var_mean={collapse_diag['latent_var_mean']:.6e}"
        )

    ho_results = {}
    train_time_results = {}
    if args.training_mode == "multi_scale":
        held_out_data = load_held_out_data_naive(
            args.data_path,
            held_out_indices=held_out_indices,
            train_ratio=args.train_ratio,
            split=getattr(args, "eval_time_split", "test"),
            max_samples=getattr(args, "eval_time_max_samples", 128),
            seed=args.seed + 6000,
        )
        if held_out_data:
            print("\nEvaluating on held-out times ...")
            ho_results = evaluate_at_times(
                autoencoder,
                state,
                held_out_data,
                batch_size=args.batch_size,
                label="Held-out",
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 3000),
                progress_every_batches=20,
            )
            if wandb_run is not None:
                for t_norm, metrics_for_t in ho_results.items():
                    wandb_run.log({f"final/held_out_mse_t{t_norm:.3f}": metrics_for_t["mse"]})

        training_time_data = load_training_time_data_naive(
            args.data_path,
            held_out_indices=held_out_indices,
            train_ratio=args.train_ratio,
            split=getattr(args, "eval_time_split", "test"),
            max_samples=getattr(args, "eval_time_max_samples", 128),
            seed=args.seed + 7000,
        )
        if training_time_data:
            print("\nEvaluating on training times ...")
            train_time_results = evaluate_at_times(
                autoencoder,
                state,
                training_time_data,
                batch_size=args.batch_size,
                label="Training",
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 4000),
                progress_every_batches=20,
            )
            if wandb_run is not None:
                for t_norm, metrics_for_t in train_time_results.items():
                    wandb_run.log(
                        {f"final/train_time_mse_t{t_norm:.3f}": metrics_for_t["mse"]}
                    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    if ho_results:
        avg_ho_mse = np.mean([v["mse"] for v in ho_results.values()])
        avg_ho_rel_mse = np.mean([v["rel_mse"] for v in ho_results.values()])
        print(f"  Held-out avg MSE: {avg_ho_mse:.6f}, Rel-MSE: {avg_ho_rel_mse:.6f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/held_out_avg_mse": avg_ho_mse,
                    "final/held_out_avg_rel_mse": avg_ho_rel_mse,
                }
            )
    if train_time_results:
        avg_train_mse = np.mean([v["mse"] for v in train_time_results.values()])
        avg_train_rel_mse = np.mean([v["rel_mse"] for v in train_time_results.values()])
        print(
            "  Training times avg MSE: "
            f"{avg_train_mse:.6f}, Rel-MSE: {avg_train_rel_mse:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/train_times_avg_mse": avg_train_mse,
                    "final/train_times_avg_rel_mse": avg_train_rel_mse,
                }
            )
    print("=" * 70)

    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "collapse_diagnostics": collapse_diag,
        "held_out_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in ho_results.items()
        },
        "training_time_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in train_time_results.items()
        },
        "architecture": architecture_info,
        "wandb_run_id": wandb_run_id,
        "training_mode": args.training_mode,
        "loss_type": getattr(args, "loss_type", "custom"),
        "optimizer": args.optimizer,
    }
    with open(os.path.join(paths["root"], "eval_results.json"), "w") as file:
        json.dump(eval_dict, file, indent=2)

    print("\nSaving model artifacts ...")
    final_ckpt_path = save_model_artifact(
        state,
        paths,
        architecture_info,
        args,
        is_best=False,
        wandb_run=wandb_run,
    )
    print(f"  Final model: {final_ckpt_path}")

    if args.save_best_model and trainer.best_state is not None:
        best_ckpt_path = save_model_artifact(
            trainer.best_state,
            paths,
            architecture_info,
            args,
            is_best=True,
            wandb_run=wandb_run,
        )
        print(f"  Best model: {best_ckpt_path}")

    print("\nGenerating visualizations ...")
    try:
        if getattr(args, "skip_final_viz", False):
            print("  Skipping final visualizations (--skip-final-viz).")
            raise RuntimeError("skipped")
        if args.training_mode == "multi_scale":
            visualize_reconstructions_all_times(
                autoencoder,
                state,
                args.data_path,
                paths["figures"],
                n_samples=args.n_vis_samples,
                held_out_indices=held_out_indices,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 5000),
            )
        else:
            fig = visualize_sample_reconstructions(
                autoencoder,
                state,
                test_loader,
                n_samples=args.n_vis_samples,
                n_batches=2,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 5000),
            )
            if fig is not None:
                fig.savefig(
                    os.path.join(paths["figures"], "reconstructions_single_scale.png"),
                    dpi=150,
                )
                plt.close(fig)
        if wandb_run is not None:
            for vis_file in glob.glob(os.path.join(paths["figures"], "*.png")):
                image_name = os.path.basename(vis_file).replace(".png", "")
                wandb_run.log({f"reconstructions/{image_name}": wandb.Image(vis_file)})
    except Exception as exc:
        print(f"Warning: Visualization failed: {exc}")

    if wandb_run is not None:
        wandb.finish()

    print(f"\nDone. Output saved to: {paths['root']}")
    return {
        "output_dir": paths["root"],
        "wandb_run_id": wandb_run_id,
        "test_metrics": test_metrics,
    }
