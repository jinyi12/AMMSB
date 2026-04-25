# NTK Joint-CSP Balancing

This note records the maintained update ordering for `--loss-type=ntk_bridge_balanced` on the no-SIGReg joint FiLM + latent-CSP surface.

## Canonical Update Order

The canonical procedure is:

1. Start from the current iterate `theta_t` and stored NTK trace state `eta_t`.
2. On NTK refresh steps, estimate the shared-encoder reconstruction and bridge traces from the **current** parameters `theta_t`.
3. Update the EMA trace state immediately.
4. Convert that refreshed state into reconstruction and bridge weights.
5. Use those refreshed weights in the optimizer step that maps `theta_t -> theta_{t+1}`.

This is the maintained meaning of NTK balancing in MMSFM, and it matches the timing used by the inline prior path in [mmsfm/fae/ntk_prior_balancing.py](../mmsfm/fae/ntk_prior_balancing.py).

## Remaining Difference From The Inline Prior Path

The maintained joint-CSP path now matches the inline prior path in **timing**:

- traces are evaluated at the current iterate
- the refreshed EMA state is formed before the optimizer step
- the refreshed weights are used in that same step

The maintained joint-CSP path now also matches the inline prior path on the
reconstruction side:

- the reconstruction trace is estimated from the exact current training batch
- the bridge trace is estimated from a separate matched multiscale bridge batch

So the remaining implementation difference is narrower: the bridge trace still
uses a separate matched bundle because the bridge objective is not defined on
the ordinary single-time reconstruction batch.

## Mathematical Difference

Let

- `L_r(theta)` be the reconstruction loss
- `L_b(theta)` be the bridge loss
- `g_r,t = grad L_r(theta_t)`
- `g_b,t = grad L_b(theta_t)`
- `tau_r(theta_t), tau_b(theta_t)` be the shared-encoder NTK traces
- `eta_t` be the stored EMA trace state

The balancing map used by the repo is

```text
w_r(eta) = 0.5 * (eta_r + eta_b) / (eta_r + eps)
w_b(eta) = lambda_bridge * 0.5 * (eta_r + eta_b) / (eta_b + eps)
```

### Canonical pre-step refresh

On a refresh step, estimate traces from `theta_t`, update `eta_t -> eta_t^+`, then use `w(eta_t^+)` in the same gradient step:

```text
theta_{t+1}
= theta_t
 - alpha_t [ w_r(eta_t^+) g_r,t + w_b(eta_t^+) g_b,t ].
```

### Lagged post-step refresh

If the trace refresh happens after the gradient step, the step uses stale weights `w(eta_t)` and the refreshed weights only affect future steps:

```text
theta_{t+1}
= theta_t
 - alpha_t [ w_r(eta_t) g_r,t + w_b(eta_t) g_b,t ].
```

The difference on refresh steps is therefore

```text
Delta(theta)_canonical - Delta(theta)_lagged
= -alpha_t [
    (w_r(eta_t^+) - w_r(eta_t)) g_r,t
  + (w_b(eta_t^+) - w_b(eta_t)) g_b,t
].
```

So the mathematical distinction comes from **timing and update order**, not from whether the Hutchinson estimator runs inside or outside JIT.

## Maintained Implementation Rule

- Moving Hutchinson trace estimation outside the train-step JIT is allowed and encouraged for efficiency.
- Moving the refresh to **after** the optimizer step is **not** canonical, because it changes the algorithm into a lagged-weight update.
- The maintained outside-JIT implementation for joint-CSP therefore refreshes NTK traces **before** the jitted train step on refresh steps.

## Current Ownership

- [mmsfm/fae/joint_csp_support.py](../mmsfm/fae/joint_csp_support.py) owns joint-CSP trace estimation and pre-step NTK refresh scheduling.
- [mmsfm/fae/wandb_trainer.py](../mmsfm/fae/wandb_trainer.py) owns the `pre_step_aux_update_fn` hook that lets the trace refresh run outside the train-step JIT while preserving canonical timing.
- [mmsfm/fae/ntk_prior_balancing.py](../mmsfm/fae/ntk_prior_balancing.py) remains the reference implementation for the weighting rule itself.
