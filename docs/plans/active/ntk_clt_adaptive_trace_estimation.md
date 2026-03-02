# Plan: Interval-Gated Global Hutchinson NTK Trace

**Status**: active  
**Created**: 2026-02-28  
**Updated**: 2026-03-01 (rollback to global Hutchinson)

## Goal

Use interval-gated global Hutchinson NTK trace estimation for NTK-scaled loss
balancing, and remove CLT/pilot-sample calibration complexity.

## Method

At training step `s`, for each NTK-scaled loss path:

- If `s % trace_update_interval == 0`, estimate full-batch NTK trace with
  global Hutchinson probes (`ntk_hutchinson_probes`).
- Otherwise, reuse the last stored trace.

Weighting remains:

- `w = numerator / (trace + epsilon)`
- `numerator = scale_norm` or `total_trace_est` when
  `--ntk-estimate-total-trace` is enabled.

## State and Metrics

`batch_stats["ntk"]` stores only:

- `step`
- `trace`
- `trace_ema`
- `total_trace_est`
- `weight`
- `is_trace_update`

Evaluation diagnostics keep only:

- `ntk_trace`
- `ntk_weight`
- `ntk_total_trace`
- `ntk_trace_ema`
- `mse`

## CLI

Removed:

- `--ntk-calibration-interval`
- `--ntk-cv-threshold`
- `--ntk-calibration-pilot-samples`

Added:

- `--ntk-trace-update-interval`

Kept:

- `--ntk-hutchinson-probes`
