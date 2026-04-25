"""AutoencoderTrainer with real-time wandb logging."""

import inspect
from collections.abc import Mapping
from typing import Any, Callable, Literal, Optional, Sequence, Union

import flax
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from functional_autoencoders.train.autoencoder_trainer import AutoencoderTrainer
from functional_autoencoders.train import TrainNanError, TrainState as BaseTrainState


class ExtendedTrainState(BaseTrainState):
    """Local extension point for MMSFM-side auxiliary training state."""

    aux_state: flax.core.FrozenDict | dict[str, Any] | None = None


def _normalize_visualization_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return {
            str(name): figure
            for name, figure in payload.items()
            if figure is not None
        }
    return {"reconstructions": payload}


def _to_python_scalar(value: Any) -> float | int | bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    array_value = jnp.asarray(value)
    if array_value.dtype == jnp.bool_:
        return bool(array_value)
    if jnp.issubdtype(array_value.dtype, jnp.integer):
        return int(array_value)
    return float(array_value)


class WandbAutoencoderTrainer(AutoencoderTrainer):
    """AutoencoderTrainer with wandb logging callbacks.

    Parameters
    ----------
    wandb_run : wandb.Run or None
        Active wandb run for logging.
    vis_callback : callable or None
        Optional callback function to generate visualization figures.
        Called with (state, epoch) and should return either a matplotlib figure,
        a dict of named matplotlib figures, or None.
    vis_interval : int
        How often (in epochs) to log visualizations.
        Visualizations can only be logged on evaluation epochs (see `eval_interval`
        in `fit`), so `vis_interval` is checked when `_evaluate` runs.
    pre_step_aux_update_fn : callable or None
        Optional MMSFM-side callback executed before the jitted optimizer step.
        Called with ``(state, step, key, epoch, batch)``. Use this for adaptive
        state that must be refreshed from the current parameter iterate and
        applied in the same optimization step.
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
        optimizer_config: Optional[dict] = None,
        initial_variables: Optional[Mapping[str, Any]] = None,
        extra_init_params_fn: Optional[Callable] = None,
        extra_init_aux_state_fn: Optional[Callable] = None,
        pre_step_aux_update_fn: Optional[Callable] = None,
        aux_update_fn: Optional[Callable] = None,
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
        self.optimizer_config = dict(optimizer_config or {"name": "adam"})
        self.initial_variables = dict(initial_variables) if initial_variables is not None else None
        self.extra_init_params_fn = extra_init_params_fn
        self.extra_init_aux_state_fn = extra_init_aux_state_fn
        self.pre_step_aux_update_fn = pre_step_aux_update_fn
        self.aux_update_fn = aux_update_fn
        self.aux_history: list[dict[str, float | int | bool]] = []
        try:
            self._loss_supports_aux_state = "aux_state" in inspect.signature(loss_fn).parameters
        except (TypeError, ValueError):
            self._loss_supports_aux_state = False

        if save_best_model and best_model_path is None:
            raise ValueError("best_model_path must be provided if save_best_model=True")

    def _get_optimizer(self, lr, lr_decay_step, lr_decay_factor):
        """Build optimizer from configuration.

        Supported names:
        - adam
        - adamw
        - muon
        """
        warmup_steps = int(self.optimizer_config.get("lr_warmup_steps", 0))
        decay_schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=lr_decay_step,
            decay_rate=lr_decay_factor,
        )
        if warmup_steps > 0:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=lr,
                        transition_steps=warmup_steps,
                    ),
                    decay_schedule,
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = decay_schedule
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

        When ``extra_init_params_fn`` is set, its output is merged into
        ``variables["params"]`` (e.g. to initialise a latent prior network).
        """
        batch = next(iter(self.train_dataloader))
        u_dec, x_dec, u_enc, x_enc = batch[:4]

        # Use a single sample for initialization
        init_u = jnp.array(u_enc[:1])
        init_x_enc = jnp.array(x_enc[:1])
        init_x_dec = jnp.array(x_dec[:1])

        # Initialize with separate encoder and decoder coordinates
        variables = self.autoencoder.init(key, init_u, init_x_enc, init_x_dec, train=False)

        if self.initial_variables is not None:
            warm_params = self.initial_variables.get("params", {})
            if isinstance(warm_params, Mapping):
                variables = dict(variables)
                variables["params"] = {
                    **variables["params"],
                    **{
                        name: value
                        for name, value in warm_params.items()
                        if name in {"encoder", "decoder"}
                    },
                }

            warm_batch_stats = self.initial_variables.get("batch_stats", None)
            if isinstance(warm_batch_stats, Mapping):
                current_batch_stats = dict(variables.get("batch_stats", {}))
                current_batch_stats.update(
                    {
                        name: value
                        for name, value in warm_batch_stats.items()
                        if name in {"encoder", "decoder"}
                    }
                )
                if current_batch_stats:
                    variables["batch_stats"] = current_batch_stats

        if self.extra_init_params_fn is not None:
            extra = self.extra_init_params_fn(key)
            variables = {**variables, "params": {**variables["params"], **extra}}

        return variables

    def _print_metrics(self, epoch, verbose):
        """Override to handle dict-valued metrics."""
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

    def _get_init_state(self, key, lr, lr_decay_step, lr_decay_factor):
        """Override to attach optional auxiliary state owned by MMSFM."""
        optimizer = self._get_optimizer(lr, lr_decay_step, lr_decay_factor)

        key, init_key = jax.random.split(key)
        init_variables = self._get_init_variables(init_key)

        aux_state = None
        if self.extra_init_aux_state_fn is not None:
            key, aux_key = jax.random.split(key)
            aux_state = self.extra_init_aux_state_fn(aux_key)

        key, state_key = jax.random.split(key)
        return ExtendedTrainState.create(
            apply_fn=self.autoencoder.apply,
            params=init_variables["params"],
            tx=optimizer,
            batch_stats=(
                init_variables["batch_stats"]
                if "batch_stats" in init_variables
                else None
            ),
            key=state_key,
            aux_state=aux_state,
        )

    def _train_one_epoch(self, key, state, step, train_step_fn, epoch, verbose):
        """Override to add wandb logging and optional MMSFM aux-state updates."""
        epoch_loss = 0.0
        n_batches = 0
        epoch_log_sums: dict[str, float] = {}
        for batch in (
            pbar := tqdm(
                self.train_dataloader,
                disable=(verbose != "full"),
                desc=f"epoch {epoch}",
            )
        ):
            if self.pre_step_aux_update_fn is not None:
                key, pre_aux_key = jax.random.split(key)
                update_result = self.pre_step_aux_update_fn(
                    state,
                    step=step,
                    key=pre_aux_key,
                    epoch=epoch,
                    batch=batch,
                )
                if update_result is not None:
                    state, aux_log = update_result
                    if aux_log:
                        aux_log_clean = {
                            str(k): _to_python_scalar(v)
                            for k, v in aux_log.items()
                        }
                        self.aux_history.append(dict(aux_log_clean))
                        if self.wandb_run is not None:
                            self.wandb_run.log(
                                {
                                    **aux_log_clean,
                                    "train/epoch": epoch,
                                    "train/step": step,
                                }
                            )

            key, subkey = jax.random.split(key)
            loss_value, state, step_log_metrics = train_step_fn(subkey, state, batch)

            epoch_loss += loss_value
            step += 1
            n_batches += 1
            if step_log_metrics:
                for metric_name, metric_value in step_log_metrics.items():
                    epoch_log_sums[str(metric_name)] = epoch_log_sums.get(str(metric_name), 0.0) + float(metric_value)

            if self.aux_update_fn is not None:
                key, aux_key = jax.random.split(key)
                update_result = self.aux_update_fn(
                    state,
                    step=step,
                    key=aux_key,
                    epoch=epoch,
                )
                if update_result is not None:
                    state, aux_log = update_result
                    if aux_log:
                        aux_log_clean = {
                            str(k): _to_python_scalar(v)
                            for k, v in aux_log.items()
                        }
                        self.aux_history.append(dict(aux_log_clean))
                        if self.wandb_run is not None:
                            self.wandb_run.log(
                                {
                                    **aux_log_clean,
                                    "train/epoch": epoch,
                                    "train/step": step,
                                }
                            )

            if verbose == "full":
                pbar.set_description(f"epoch {epoch} (loss {loss_value:.3E})")
            if jnp.any(jnp.isnan(epoch_loss)):
                raise TrainNanError()

        epoch_loss /= max(n_batches, 1)
        self.training_loss_history.append(epoch_loss)

        # Log training loss to wandb
        if self.wandb_run is not None:
            epoch_loss = self.training_loss_history[-1]
            payload = {
                "train/loss": float(epoch_loss),
                "train/epoch": epoch,
                "train/step": step,
            }
            if epoch_log_sums:
                for metric_name, total_value in epoch_log_sums.items():
                    payload[f"train/{metric_name}"] = float(total_value / max(n_batches, 1))
            self.wandb_run.log(payload)

        return state, step

    def _get_train_step_fn(self):
        """Override to support both 4-item and 5-item batch tuples.

        Some functional_autoencoders releases only support
        (u_dec, x_dec, u_enc, x_enc). Sobolev training uses an additional
        decoder gradient tensor du_dec.
        """
        loss_supports_aux_state = bool(self._loss_supports_aux_state)

        @jax.jit
        def step_func(k, state, batch):
            if len(batch) == 4:
                u_dec, x_dec, u_enc, x_enc = batch
                extra_kwargs = {}
            elif len(batch) == 5:
                u_dec, x_dec, u_enc, x_enc, du_dec = batch
                extra_kwargs = {"du_dec": du_dec}
            else:
                raise ValueError(
                    "Unexpected batch structure. Expected 4-tuple "
                    "(u_dec, x_dec, u_enc, x_enc) or 5-tuple "
                    "(u_dec, x_dec, u_enc, x_enc, du_dec). "
                    f"Got {len(batch)} elements."
                )

            grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
            loss_kwargs = {
                "key": k,
                "batch_stats": state.batch_stats,
                "u_enc": u_enc,
                "x_enc": x_enc,
                "u_dec": u_dec,
                "x_dec": x_dec,
            }
            if loss_supports_aux_state:
                loss_kwargs["aux_state"] = state.aux_state
            loss_kwargs.update(extra_kwargs)
            (loss_value, batch_stats), grads = grad_fn(
                state.params,
                **loss_kwargs,
            )
            step_log_metrics = None
            if isinstance(batch_stats, Mapping) and "batch_stats" in batch_stats:
                step_log_metrics = batch_stats.get("log_metrics")
                batch_stats = batch_stats["batch_stats"]
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=batch_stats)
            return loss_value, state, step_log_metrics

        return step_func

    def _evaluate(self, key, state):
        """Override to add wandb logging and best model tracking."""
        super()._evaluate(key, state)

        # `training_loss_history` is appended at the end of `_train_one_epoch`, so
        # the 0-based epoch index is `len(...) - 1`.
        epoch = max(len(self.training_loss_history) - 1, 0)

        # Track best model if enabled
        if self.save_best_model:
            # Prefer an explicit scalar MSE, but also support dict-valued
            # diagnostics like {"mse": ..., "rel_mse": ...}.
            mse_value = None
            for metric in self.metrics:
                metric_value = self.metrics_history[metric.name][-1]
                if isinstance(metric_value, dict) and "mse" in metric_value:
                    mse_value = float(metric_value["mse"])
                    break
                if "mse" in metric.name.lower() and not isinstance(metric_value, dict):
                    mse_value = float(metric_value)
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
                    figures = _normalize_visualization_payload(self.vis_callback(state, epoch))
                    if figures and wandb is not None:
                        log_dict = {"eval/epoch_vis": epoch}
                        for name, fig in figures.items():
                            log_dict[f"eval/{name}"] = wandb.Image(fig)
                        self.wandb_run.log(log_dict)
                    for fig in figures.values():
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
                    "aux_state": (
                        jax.tree.map(np.array, self.best_state.aux_state)
                        if getattr(self.best_state, "aux_state", None) is not None
                        else None
                    ),
                    "best_metric_value": self.best_metric_value,
                },
                f,
            )
        print(f"Best model checkpoint saved to {self.best_model_path} (MSE={self.best_metric_value:.6f})")
