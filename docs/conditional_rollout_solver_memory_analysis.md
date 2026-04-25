# Conditional Rollout Solver Memory Analysis

This note records the April 16, 2026 diagnosis and fix for the GPU-memory
accumulation that appeared during token-native `conditional_rollout` latent
cache generation.

Use it together with [csp_conditional_evaluation_core.md](csp_conditional_evaluation_core.md)
and [runbooks/csp.md](runbooks/csp.md).

## Scope

The issue affected the token-native rollout sampling path used by:

- [scripts/csp/build_conditional_rollout_latent_cache.py](../scripts/csp/build_conditional_rollout_latent_cache.py)
- [scripts/fae/tran_evaluation/coarse_consistency_runtime.py](../scripts/fae/tran_evaluation/coarse_consistency_runtime.py)
  when it reconstructs token-native conditioned runtimes

The failure mode was observed during real `m=100, r=100, k=8` rollout-cache
generation on GPU.

## Observed Failure

The rollout latent cache saved correctly, but GPU memory increased roughly in
proportion to the number of completed condition chunks.

Observed symptom before the fix:

- one saved rollout chunk corresponded to only about `37.5 MiB` of latent data
  for `1 condition x 100 realizations x 6 times x 128 x 128 x float32`
- actual GPU memory growth was about `3 GiB` per saved condition chunk
- the real run reached about `20.8 GiB` by only `5` saved chunks

That growth rate is too large to be explained by the saved latent payload
itself. The problem was therefore not the resumable cache store.

## Root Cause

The accumulation came from the token ODE/SDE sampling path, not from cache
writing.

### 1. Forward sampling was still using a training-style adjoint

Before the fix, token rollout sampling could reach `diffrax` solves with the
default `RecursiveCheckpointAdjoint` behavior. For inference-time rollout cache
generation, this is the wrong default. It preserves extra solver state and
backward-oriented bookkeeping that are useful for differentiation, but not for
forward-only sampling.

The maintained forward-only default is now explicit in
[csp/token_sample.py](../csp/token_sample.py):

- [_resolve_forward_sampling_adjoint](../csp/token_sample.py#L15) returns
  `diffrax.DirectAdjoint()` when the caller does not override the adjoint
- [sample_token_conditional_batch](../csp/token_sample.py#L129) and
  [sample_token_conditional_trajectory](../csp/token_sample.py#L98) both route
  through that forward-only default

The runtime loaders now also record and use the forward sampling default for
sampling-only rollout paths:

- [scripts/fae/tran_evaluation/coarse_consistency_runtime.py](../scripts/fae/tran_evaluation/coarse_consistency_runtime.py#L557)
- [scripts/fae/tran_evaluation/coarse_consistency_runtime.py](../scripts/fae/tran_evaluation/coarse_consistency_runtime.py#L814)
- [scripts/csp/conditional_rollout_runtime.py](../scripts/csp/conditional_rollout_runtime.py#L141)
- [scripts/csp/conditional_eval/rollout_context.py](../scripts/csp/conditional_eval/rollout_context.py#L252)

### 2. Sampling was being performed as many small repeated solves

For one outer rollout cache chunk, the token runtime expands one coarse
condition into many realizations, then microbatches those realizations.

The active microbatch loop is in
[scripts/csp/token_run_context.py](../scripts/csp/token_run_context.py#L315).

For the real run:

- `condition_chunk_size = 1`
- `n_realizations = 100`
- `sampling_max_batch_size = 4`

So one saved rollout chunk required `25` internal sampling calls.

When those calls repeatedly construct or retain compiled solver state on GPU,
memory can appear to leak chunk-by-chunk even though the outer cache payload is
small.

### 3. Device outputs were not being forced back to host at chunk scope

The sampling path now explicitly forces each microbatch result to complete and
move off device before the next chunk is processed:

- `jax.block_until_ready(...)`
- `jax.device_get(...)`

This is implemented in
[scripts/csp/token_run_context.py](../scripts/csp/token_run_context.py#L320).

That change did not fix the root cause by itself, but it removed one additional
source of device residency across microbatches.

## Why The Cache Store Was Not The Problem

The rollout latent cache store writes small condition chunks and does not retain
all previous chunks in GPU memory. The saved payload for a single condition is
small compared with the observed memory growth.

The disproportionate `~3 GiB` growth per saved chunk therefore pointed to:

- compiled executable accumulation
- solver workspaces
- retained device-side sampling outputs

not to `.cache/` storage itself.

## Implemented Fix

The fix had three parts.

### 1. Make forward-only sampling explicit

`conditional_rollout` cache generation is inference, not training. The code now
defaults to a forward adjoint:

- [csp/token_sample.py](../csp/token_sample.py#L15)

This aligns the solver configuration with the scientific task: generate rollout
samples without gradient tracking.

### 2. Keep one stable compiled batch-sampling path

The token batch sampler now routes through a single jitted implementation:

- [_sample_token_conditional_batch_impl](../csp/token_sample.py#L67)

It computes `generation_zt` once and reuses one stable batch-sampling shape,
instead of repeatedly rebuilding solver structure at Python level for each
microbatch.

### 3. Materialize microbatches on host immediately

The microbatch loop now forces chunk completion and transfers it to host before
continuing:

- [scripts/csp/token_run_context.py](../scripts/csp/token_run_context.py#L320)

This prevents device outputs from lingering across the outer rollout-cache
condition loop.

## Validation

### Real-run outcome

After the fix, the real `conditional_rollout` latent-cache rerun completed with
flat GPU memory:

- stable observed GPU memory during rollout cache generation:
  about `546 MiB`
- final latent cache:
  `100` saved condition chunks, complete

### Longer smoke validation

A longer `/tmp` rollout smoke run progressed from `1` to `7` saved chunks while
remaining at about `546 MiB`, then completed all `8` chunks.

That removed the previous monotone chunk-by-chunk GPU growth.

## What This Fix Does Not Address

This note is specifically about the rollout-solver GPU accumulation bug.

It does not solve the later coarse-consistency host-memory bloat caused by
whole-cache loading of decoded `.npz` exports. That is a separate CPU-side
evaluation issue in:

- [scripts/fae/tran_evaluation/coarse_consistency_cache.py](../scripts/fae/tran_evaluation/coarse_consistency_cache.py)
- [scripts/fae/tran_evaluation/coarse_consistency_runtime.py](../scripts/fae/tran_evaluation/coarse_consistency_runtime.py)

## Scientific Programming Lessons

The fix follows a simple rule for scientific inference code:

- use the simplest solver mode that matches the task
- make forward-only inference explicit instead of inheriting training defaults
- keep batch shapes stable across repeated solves
- materialize device results at chunk boundaries
- treat `.cache/` stores as stage boundaries, not as in-memory working sets

For this codebase, `conditional_rollout` cache generation should remain a
forward-only sampling computation with simple chunking, not a training-style
solver path with hidden backward-oriented state.
