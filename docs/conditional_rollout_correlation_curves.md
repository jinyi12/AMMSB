# Conditional Rollout Correlation Curves

This note records the maintained mathematical contract for the
`fig_conditional_rollout_field_corr_<role>` family in
`--rollout_condition_mode chatterjee_knn`, together with a worked diagnosis of
the April 19, 2026 `k=6` token-native rollout run.

Use this document together with
[csp_conditional_evaluation_core.md](csp_conditional_evaluation_core.md),
[runbooks/csp.md](runbooks/csp.md), and
[tran_coarse_consistency.md](tran_coarse_consistency.md).

## Scope

This note covers the maintained conditional-rollout two-point figures:

- `fig_conditional_rollout_field_corr_<role>`
- `fig_conditional_rollout_recoarsened_field_corr_<role>`

It does not redefine the compatibility-only `exact_query` path. That mode uses
a different observed-side estimator based on a single-field ergodic
pair-correlation surrogate.

The relevant code surfaces are:

- [scripts/csp/conditional_eval/rollout_reports.py](../scripts/csp/conditional_eval/rollout_reports.py)
- [scripts/csp/conditional_eval/rollout_assignment_cache.py](../scripts/csp/conditional_eval/rollout_assignment_cache.py)
- [scripts/csp/conditional_eval/rollout_neighborhood_cache.py](../scripts/csp/conditional_eval/rollout_neighborhood_cache.py)
- [scripts/csp/conditional_eval/rollout_generated_cache.py](../scripts/csp/conditional_eval/rollout_generated_cache.py)
- [scripts/fae/tran_evaluation/second_order.py](../scripts/fae/tran_evaluation/second_order.py)

## Notation

Fix one selected coarse condition, indexed by `m`.

- `H_c` is the conditioning coarse scale. In the maintained rollout runs this is
  the coarse root `H_c = 6`.
- `H_t` is one response scale, for example `H_t = 4`.
- `k` is the neighbor-support size, for example `k = 6`.
- `L` is the number of realization draws used in the local conditional
  ensemble, for example `L = 100`.

Let

$$
\mathcal N_m = \{i_{m,1}, \dots, i_{m,k}\}
$$

be the saved Chatterjee Gaussian kNN support for coarse condition `m`, with
normalized weights

$$
w_{m,\ell} \ge 0,
\qquad
\sum_{\ell=1}^k w_{m,\ell} = 1.
$$

These support indices and weights come from the saved reference cache in
[rollout_reference_cache.py](../scripts/csp/conditional_eval/rollout_reference_cache.py),
and the sampled realization assignments come from
[rollout_assignment_cache.py](../scripts/csp/conditional_eval/rollout_assignment_cache.py).

## Step 1: Data-side Local Empirical Conditional

At response scale `H_t`, the data-side local empirical conditional is

$$
\widehat P^{\mathrm{ref}}_{m,t}
=
\sum_{\ell=1}^k w_{m,\ell}\,
\delta_{U^{(i_{m,\ell})}_{H_t}}.
$$

This is not a global dataset average. It is a row-specific conditional
reference built from the saved `k` neighbors of one coarse root state.

The assignment cache then draws `L` reference indices with replacement:

$$
J^{\mathrm{ref}}_{m,1}, \dots, J^{\mathrm{ref}}_{m,L}
\stackrel{\mathrm{iid}}{\sim}
\mathrm{Categorical}(w_{m,1}, \dots, w_{m,k}).
$$

The reference ensemble plotted in the correlation figure is therefore

$$
X_r = U^{(J^{\mathrm{ref}}_{m,r})}_{H_t},
\qquad
r = 1, \dots, L.
$$

This is exactly what
[rollout_reports.py](../scripts/csp/conditional_eval/rollout_reports.py)
loads in `_rollout_report_reference_fields(...)`.

## Step 2: Model-side Local Empirical Conditional

The model side also uses the saved neighbor support.

The assignment cache draws a second set of `L` neighbor indices

$$
J^{\mathrm{gen}}_{m,1}, \dots, J^{\mathrm{gen}}_{m,L}
\stackrel{\mathrm{iid}}{\sim}
\mathrm{Categorical}(w_{m,1}, \dots, w_{m,k}),
$$

stored as `generated_assignment_indices`.

For each draw `r`, the rollout cache samples one conditional rollout using the
selected support seed `J^{\mathrm{gen}}_{m,r}`; the neighborhood latent cache is
built from exactly those assignments in
[rollout_neighborhood_cache.py](../scripts/csp/conditional_eval/rollout_neighborhood_cache.py).
After decode, the generated figure row is loaded from the decoded rollout store
in [rollout_generated_cache.py](../scripts/csp/conditional_eval/rollout_generated_cache.py).

So the generated response-scale ensemble has the form

$$
Y_r \sim \widehat P^\theta_{m,t},
\qquad
r = 1, \dots, L,
$$

where `\widehat P^\theta_{m,t}` is the model-side local empirical conditional
induced by the same saved `k`-neighbor support and the rollout sampler.

This point matters for interpretation:

- the blue reference curve depends on a finite `k`-neighbor local support,
- the red generated curve also depends on that same finite local support,
- but the red curve is not a direct average of the `k` observed neighbor fields;
  it is the model ensemble generated from `L` rollout draws seeded by that
  support.

## Step 3: Pooled Directional Pair-Correlation Estimator

For either ensemble `Z_1, \dots, Z_L`, where `Z_r` is either `X_r` or `Y_r`,
write `Z_r(a,b)` for the pixel value at row `a`, column `b`.

First compute the pixelwise ensemble mean

$$
\overline Z(a,b)
=
\frac{1}{L} \sum_{r=1}^L Z_r(a,b),
$$

and the centered fields

$$
\widetilde Z_r(a,b)
=
Z_r(a,b) - \overline Z(a,b).
$$

For a horizontal lag `h`, a line-pair product means

$$
\widetilde Z_r(a,b)\,\widetilde Z_r(a,b+h),
$$

that is, two points from the same row of the same field, separated by lag `h`.
There are no cross-row and no cross-field products inside a single term.

The maintained estimator pools all valid same-line products across all rows and
all ensemble members before normalizing:

$$
\widehat R_{e_1}(h)
=
\frac{
\sum_{r=1}^L \sum_{a=1}^n \sum_{b=1}^{n-h}
\widetilde Z_r(a,b)\,\widetilde Z_r(a,b+h)
}{
\sqrt{
\left(
\sum_{r=1}^L \sum_{a=1}^n \sum_{b=1}^{n-h}
\widetilde Z_r(a,b)^2
\right)
\left(
\sum_{r=1}^L \sum_{a=1}^n \sum_{b=1}^{n-h}
\widetilde Z_r(a,b+h)^2
\right)
}
}.
$$

The vertical curve is analogous:

$$
\widehat R_{e_2}(h)
=
\frac{
\sum_{r=1}^L \sum_{b=1}^n \sum_{a=1}^{n-h}
\widetilde Z_r(a,b)\,\widetilde Z_r(a+h,b)
}{
\sqrt{
\left(
\sum_{r=1}^L \sum_{b=1}^n \sum_{a=1}^{n-h}
\widetilde Z_r(a,b)^2
\right)
\left(
\sum_{r=1}^L \sum_{b=1}^n \sum_{a=1}^{n-h}
\widetilde Z_r(a+h,b)^2
\right)
}
}.
$$

This is the pooled estimator implemented by
`rollout_ensemble_directional_paircorr(...)` in
[second_order.py](../scripts/fae/tran_evaluation/second_order.py).

It is not the same as the Tran-style "normalize each realization first, then
average" estimator. Here the pooling happens before the final normalization.

## Step 4: Bootstrap Bands

The central curves are not bootstrap means. The bootstrap is only used for the
uncertainty bands.

Given `Z_1, \dots, Z_L`, each bootstrap replicate resamples ensemble members
with replacement:

$$
I^{(b)}_1, \dots, I^{(b)}_L
\stackrel{\mathrm{iid}}{\sim}
\mathrm{Uniform}\{1, \dots, L\},
$$

forms the bootstrap ensemble

$$
Z^{*(b)}_r = Z_{I^{(b)}_r},
$$

and recomputes the same pooled statistic

$$
\widehat R^{*(b)}_{e_k}(h).
$$

The plotted band is the pointwise percentile interval

$$
\left[
q_{0.025}\{\widehat R^{*(b)}_{e_k}(h)\},
q_{0.975}\{\widehat R^{*(b)}_{e_k}(h)\}
\right].
$$

In `chatterjee_knn`, both reference and generated bands are sample-index
bootstrap percentile bands.

## Step 5: What Isotropy Means Here

If the unconditional field law is rotation invariant, then the unconditional
directional correlation satisfies

$$
R(h e_1) = R(h e_2)
$$

in expectation.

That is the right expectation for the global dataset, up to finite-domain,
discretization, and finite-sample effects. The Tran-style truncated Gaussian
filter used in the dataset is intended to remain approximately isotropic in
that global sense.

But the conditional figures do not plot the unconditional law. They plot a
row-specific local empirical conditional.

Even if the global law is isotropic, conditioning on one realized coarse field
`C = c` can break rotational symmetry:

$$
\mathcal L(U \mid C = c)
\neq
\mathcal L(Q U \mid C = c)
$$

for a rotation `Q`, unless `c` itself is rotation invariant.

So a separation between the `e_1` and `e_2` curves in one selected
`field_corr_<role>` panel does not by itself imply that the global dataset or
the global filtering procedure is anisotropic.

## Step 6: Why Finite Neighbor Support Matters

Finite neighbor support matters in two ways.

First, the local empirical conditional itself changes when `k` is small. A
small support can concentrate on a set of neighbors that share a directional
feature, even when the full dataset is globally isotropic.

Second, the sampled assignment cache adds Monte Carlo variability on top of
that local support because the plotted blue and red ensembles are both built
from `L` draws from the saved support weights.

But finite support is not the whole story. Once the local condition is fixed,
the resulting conditional law may itself be directionally biased. In that case
increasing `L` only reduces Monte Carlo error; it does not force the two
directional curves to coincide.

## Worked Diagnosis: April 19, 2026 `k=6` Run

This section records the concrete diagnosis for the token-native rollout run

`conditional_rollout_m100_r100_k6_cache_gpu_launch_20260419_000046`

and its representative figure

`fig_conditional_rollout_field_corr_best.pdf`.

The selected `best` panel corresponds to:

- selected row `m = 18`
- `test_sample_index = 211`
- response scale `H_t = 4`
- support size `k = 6`
- realization count `L = 100`

The saved support for row `18` is:

- support indices `[99, 191, 313, 437, 373, 443]`
- support weights `[0.1931, 0.1888, 0.1709, 0.1584, 0.1448, 0.1440]`

The key comparison is:

| Object at `H=4` | Estimator | `max |R_{e_1} - R_{e_2}|` | Interpretation |
| --- | --- | ---: | --- |
| Full test subset used by the rollout run | Tran-style ensemble average | `0.0306` | Globally close to isotropic |
| Row-18 6-neighbor support only | Tran-style ensemble average | `0.0403` | Local support is already a bit more directional |
| Row-18 plotted reference curve | Maintained pooled conditional estimator | `0.2625` | Strong conditional anisotropy on the blue side |
| Row-18 plotted generated curve | Maintained pooled conditional estimator | `0.4102` | Even stronger conditional anisotropy on the red side |

The resulting interpretation is:

1. the global `H=4` dataset used by this rollout run is still close to
   isotropic,
2. the selected row `18` local conditional is much less isotropic than the
   global dataset,
3. the finite `k=6` support contributes to that localization on both sides,
4. the separation seen in `fig_conditional_rollout_field_corr_best.pdf` is not
   just plotting noise from neighbor resampling,
5. the generated `\widetilde U_{H=4}` ensemble is reproducing or amplifying a
   row-specific conditional anisotropy rather than revealing a global dataset
   anisotropy.

## Practical Reading Rules

For future interpretation of `field_corr_<role>` panels in the maintained
`chatterjee_knn` rollout path:

1. treat the blue curve as a sampled local empirical conditional reference, not
   as the global dataset correlation curve,
2. treat the red curve as a model-side local empirical conditional driven by
   the same saved support, not as an unconditional model average,
3. do not use one selected panel to infer global isotropy of the dataset,
4. if a panel looks anisotropic, compare it against a global test-subset
   isotropy calculation before concluding that the filtering or the dataset
   construction is anisotropic,
5. if the scientific question is "how much comes from finite support `k`?",
   compare against a larger-`k` rerun or against the exact weighted support
   mixture without assignment resampling.
