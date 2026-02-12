"""AutoencoderTrainer with real-time wandb logging."""

from typing import Sequence, Literal, Union, Callable, Optional
import jax
import jax.numpy as jnp
import optax

try:
    import wandb
except ImportError:
    wandb = None

from functional_autoencoders.train.autoencoder_trainer import AutoencoderTrainer


class WandbAutoencoderTrainer(AutoencoderTrainer):
    """AutoencoderTrainer with wandb logging callbacks.

    Parameters
    ----------
    wandb_run : wandb.Run or None
        Active wandb run for logging.
    vis_callback : callable or None
        Optional callback function to generate visualization figures.
        Called with (state, epoch) and should return a matplotlib figure or None.
    vis_interval : int
        How often (in epochs) to log visualizations.
        Visualizations can only be logged on evaluation epochs (see `eval_interval`
        in `fit`), so `vis_interval` is checked when `_evaluate` runs.
    save_best_model : bool
        If True, track and save the best model based on validation MSE.
    best_model_path : str or None
        Path to save the best model checkpoint. Required if save_best_model=True.
    """

    def __init__(
        self,
        autoencoder,
        loss_fn,
        metrics: Sequence,
        train_dataloader,
        test_dataloader,
        wandb_run=None,
        vis_callback: Optional[Callable] = None,
        vis_interval: int = 1,
        save_best_model: bool = False,
        best_model_path: Optional[str] = None,
        track_spectral: bool = False,
        optimizer_config: Optional[dict] = None,
    ):
        super().__init__(
            autoencoder=autoencoder,
            loss_fn=loss_fn,
            metrics=metrics,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )
        self.wandb_run = wandb_run
        self.vis_callback = vis_callback
        self.vis_interval = vis_interval
        self.save_best_model = save_best_model
        self.best_model_path = best_model_path
        self.best_metric_value = float("inf")
        self.best_state = None
        self.track_spectral = track_spectral
        self.spectral_history = []  # For storing full spectral metrics over training
        self.optimizer_config = dict(optimizer_config or {"name": "adam"})

        if save_best_model and best_model_path is None:
            raise ValueError("best_model_path must be provided if save_best_model=True")

    def _get_optimizer(self, lr, lr_decay_step, lr_decay_factor):
        """Build optimizer from configuration.

        Supported names:
        - adam
        - adamw
        - muon
        """
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=lr_decay_step,
            decay_rate=lr_decay_factor,
        )
        name = str(self.optimizer_config.get("name", "adam")).lower()
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.0))

        if name == "adam":
            return optax.adam(learning_rate=schedule)

        if name == "adamw":
            return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

        if name == "muon":
            # Muon applies Newton-Schulz orthogonalization on 2D params and
            # falls back to Adam updates for non-2D params.
            return optax.contrib.muon(
                learning_rate=schedule,
                ns_steps=int(self.optimizer_config.get("muon_ns_steps", 5)),
                beta=float(self.optimizer_config.get("muon_beta", 0.95)),
                adaptive=bool(self.optimizer_config.get("muon_adaptive", False)),
                weight_decay=weight_decay,
            )

        raise ValueError(f"Unknown optimizer '{name}'. Expected one of: adam, adamw, muon.")

    def _get_init_variables(self, key):
        """Override to handle separate encoder/decoder coordinate dimensions.

        This is necessary for time-invariant decoder architectures where:
        - Encoder uses 3D coordinates (x, y, t)
        - Decoder uses 2D coordinates (x, y)
        """
        batch = next(iter(self.train_dataloader))
        u_dec, x_dec, u_enc, x_enc = batch

        # Use a single sample for initialization
        init_u = jnp.array(u_enc[:1])
        init_x_enc = jnp.array(x_enc[:1])
        init_x_dec = jnp.array(x_dec[:1])

        # Initialize with separate encoder and decoder coordinates
        variables = self.autoencoder.init(key, init_u, init_x_enc, init_x_dec, train=False)
        return variables

    def _print_metrics(self, epoch, verbose):
        """Override to handle dict-valued metrics (e.g. SpectralMetric)."""
        if verbose != "none":
            parts = []
            for metric_name in self.metrics_history:
                value = self.metrics_history[metric_name][-1]
                if isinstance(value, dict):
                    # Format dict metrics as key=val pairs
                    dict_parts = [f"{k}={v:.3E}" for k, v in value.items()]
                    parts.append(f"{metric_name}: {{{', '.join(dict_parts)}}}")
                else:
                    parts.append(f"{metric_name}: {value:.3E}")
            metric_string = " | ".join(parts)
            import sys
            print(f"epoch {epoch:6} || {metric_string}")
            sys.stdout.flush()

    def _train_one_epoch(self, key, state, step, train_step_fn, epoch, verbose):
        """Override to add wandb logging."""
        state, step = super()._train_one_epoch(
            key, state, step, train_step_fn, epoch, verbose
        )

        # Log training loss to wandb
        if self.wandb_run is not None:
            epoch_loss = self.training_loss_history[-1]
            self.wandb_run.log({
                "train/loss": float(epoch_loss),
                "train/epoch": epoch,
                "train/step": step,
            })

        return state, step

    def _evaluate(self, key, state):
        """Override to add wandb logging and best model tracking."""
        super()._evaluate(key, state)

        # `training_loss_history` is appended at the end of `_train_one_epoch`, so
        # the 0-based epoch index is `len(...) - 1`.
        epoch = max(len(self.training_loss_history) - 1, 0)

        # Track best model if enabled
        if self.save_best_model:
            # Look for MSE metric (case insensitive)
            mse_value = None
            for metric in self.metrics:
                if "mse" in metric.name.lower():
                    mse_value = self.metrics_history[metric.name][-1]
                    break

            if mse_value is not None and mse_value < self.best_metric_value:
                self.best_metric_value = mse_value
                # Store a copy of the state for best model
                import copy
                self.best_state = copy.deepcopy(state)
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "eval/best_mse": float(mse_value),
                        "eval/best_epoch": epoch,
                    })

        # Log evaluation metrics to wandb
        if self.wandb_run is not None:
            log_dict = {"eval/epoch": epoch}
            for metric in self.metrics:
                metric_value = self.metrics_history[metric.name][-1]
                # Clean metric name for wandb
                clean_name = metric.name.replace(" ", "_").replace("(", "").replace(")", "")

                # Handle both scalar and dict-valued metrics
                if isinstance(metric_value, dict):
                    # Log each key in the dict
                    for key, val in metric_value.items():
                        log_dict[f"eval/{clean_name}/{key}"] = float(val)
                else:
                    log_dict[f"eval/{clean_name}"] = float(metric_value)

            self.wandb_run.log(log_dict)

            # Log visualizations if callback is provided and it's time
            should_log_vis = (
                self.vis_callback is not None and
                self.vis_interval > 0 and
                epoch % self.vis_interval == 0
            )

            if should_log_vis:
                try:
                    import matplotlib.pyplot as plt
                    fig = self.vis_callback(state, epoch)
                    if fig is not None and wandb is not None:
                        self.wandb_run.log({
                            "eval/reconstructions": wandb.Image(fig),
                            "eval/epoch_vis": epoch,
                        })
                        plt.close(fig)
                except Exception as e:
                    print(f"Warning: Visualization callback failed: {e}")

    def save_best_model_checkpoint(self):
        """Save the best model checkpoint to disk."""
        if not self.save_best_model or self.best_state is None:
            return

        import pickle
        import numpy as np
        import jax

        with open(self.best_model_path, "wb") as f:
            pickle.dump(
                {
                    "params": jax.tree.map(np.array, self.best_state.params),
                    "batch_stats": (
                        jax.tree.map(np.array, self.best_state.batch_stats)
                        if self.best_state.batch_stats
                        else None
                    ),
                    "best_metric_value": self.best_metric_value,
                },
                f,
            )
        print(f"Best model checkpoint saved to {self.best_model_path} (MSE={self.best_metric_value:.6f})")
