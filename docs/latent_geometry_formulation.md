# Latent Geometry Diagnostics (Current Implementation)

This document summarizes the *current* mathematical formulation implemented in:

- `scripts/fae/tran_evaluation/latent_geometry.py`
- (aggregation + cross-model comparison) `scripts/fae/tran_evaluation/compare_latent_geometry_models.py`

It is intended to be read as "what the code computes", not as a claim of the
best possible estimator.

The maintained cross-run surface is the canonical transformer pair selected by
default in `scripts/fae/tran_evaluation/compare_latent_geometry_models.py` and
documented in `docs/experiments/transformer_pair_geometry.md`. The companion
CSV `docs/experiments/transformer_pair_geometry_registry.csv` is retained as a
provenance record for the fixed pair. Historical multi-run latent128
publication comparisons are no longer the active maintained workflow.

For transformer-token FAEs, encoder outputs are stored through the maintained
flattened downstream latent surface, then restored to token memory
`[batch, n_latents, emb_dim]` before any direct decoder apply, JVP, VJP, or
Hessian probe. This preserves the estimator definitions below while fixing the
runtime representation boundary.

## Notation

Fix a coordinate set (grid/point cloud)
$$
X = \{x_j\}_{j=1}^{n_x},\qquad x_j \in \mathbb{R}^{d_x}.
$$

For a trained decoder, define the discretized decoder map
$$
D_X : \mathbb{R}^{d_z} \to \mathbb{R}^{m},\qquad
D_X(z) := \mathrm{vec}\big(D(z, X)\big),
$$
where $m = n_x \cdot c$ is the number of decoder outputs after flattening
($c$ = number of output channels).

Let the Jacobian with respect to latent coordinates be
$$
J(z) := \frac{\partial D_X(z)}{\partial z} \in \mathbb{R}^{m \times d_z}.
$$

The (discretized) pullback metric is
$$
g(z) := J(z)^{\top} J(z) \in \mathbb{R}^{d_z \times d_z}.
$$

The evaluation samples latent codes $\{z_i\}_{i=1}^{n}$ from the encoder and
computes pointwise quantities $f(g(z_i))$, which are then aggregated over $i$
and over time/scale index.

## A. Pullback-Metric Moments (Trace, Trace-Squared, Effective Rank)

### Definitions

The two core moments are
$$
\mathrm{Tr}(g(z)) = \|J(z)\|_F^2,\qquad
\mathrm{Tr}(g(z)^2) = \|g(z)\|_F^2.
$$

The "participation ratio" effective rank is
$$
r_{\mathrm{eff}}(z) := \frac{\mathrm{Tr}(g(z))^2}{\mathrm{Tr}(g(z)^2)}.
$$

### Hutchinson-style estimators (Rademacher probes)

The implementation uses Rademacher probes $v \in \{\pm 1\}^{d_z}$ with
$\mathbb{E}[v v^{\top}] = I$ to estimate moments without forming $J$ or $g$.

For a single latent point $z$, draw probes $\{v_k\}_{k=1}^{K}$ and compute:
$$
\widehat{\mathrm{Tr}(g)}(z)
= \frac{1}{K}\sum_{k=1}^{K} \|J(z) v_k\|_2^2
= \frac{1}{K}\sum_{k=1}^{K} v_k^{\top} g(z) v_k,
$$
and
$$
\widehat{\mathrm{Tr}(g^2)}(z)
= \frac{1}{K}\sum_{k=1}^{K} \|g(z) v_k\|_2^2
= \frac{1}{K}\sum_{k=1}^{K} v_k^{\top} g(z)^2 v_k.
$$

In code, $J(z)v$ is computed via a JVP, and $g(z)v = J(z)^{\top}(J(z)v)$ is
computed via a transpose-linearization (VJP of the linearized map).

The reported effective rank per sample is then
$$
\widehat{r_{\mathrm{eff}}}(z)
= \mathrm{clip}_{[1,d_z]}\left(
\frac{\widehat{\mathrm{Tr}(g)}(z)^2}{\max(\widehat{\mathrm{Tr}(g^2)}(z), \varepsilon)}
\right),
$$
where $\varepsilon = \texttt{config.eps}$ is a numerical safeguard.

## B. Volume And Projected-Spectrum Surrogates (Volumetric Robustness, Near-Null Mass)

These quantities are computed from a *randomly projected* surrogate spectrum of
$g(z)$.

### Random subspace and projected Gram matrix

Let $d_{\mathrm{proj}} = \min(d_z, \texttt{config.n\_probes})$.
Draw a Gaussian matrix $A \in \mathbb{R}^{d_z \times d_{\mathrm{proj}}}$ with
i.i.d. $N(0,1)$ entries, compute a QR factorization, and keep the first
$d_{\mathrm{proj}}$ orthonormal columns:
$$
Q \in \mathbb{R}^{d_z \times d_{\mathrm{proj}}},\qquad Q^{\top}Q=I.
$$

For each $z$, form the projected Jacobian action
$$
Y(z) := J(z) Q \in \mathbb{R}^{m \times d_{\mathrm{proj}}},
$$
and its Gram matrix
$$
G_{\mathrm{proj}}(z) := Y(z)^{\top}Y(z)
= Q^{\top} g(z)\, Q \in \mathbb{R}^{d_{\mathrm{proj}} \times d_{\mathrm{proj}}}.
$$

Let $\{\lambda_j(z)\}_{j=1}^{d_{\mathrm{proj}}}$ be the eigenvalues of
$G_{\mathrm{proj}}(z)$.
The implementation applies:
$$
\tilde{\lambda}_j(z) := \max\left(\varepsilon, \frac{d_z}{d_{\mathrm{proj}}}\lambda_j(z)\right),
$$
to heuristically scale projected eigenvalues to a $d_z$-dimensional proxy and
to enforce positivity.

### Volumetric robustness

$$
\rho_{\mathrm{vol},\gamma}(z)
:=
\frac{\det(g(z)+\gamma I)^{1/d_z}}
{\mathrm{Tr}(g(z))/d_z+\gamma}.
$$

Equivalently, with eigenvalues \(\lambda_1,\dots,\lambda_{d_z}\) of \(g(z)\),
$$
\rho_{\mathrm{vol},\gamma}(z)
=
\frac{\left(\prod_{j=1}^{d_z}(\lambda_j+\gamma)\right)^{1/d_z}}
{\frac{1}{d_z}\sum_{j=1}^{d_z}(\lambda_j+\gamma)}
\in (0,1].
$$

The implementation uses a shared per-time ridge
$$
\gamma_t := \max\!\left(\eta\,\frac{\langle \mathrm{Tr}(g)\rangle_t}{d_z}, \varepsilon\right),
\qquad \eta=\texttt{config.vol\_ridge\_rel},
$$
and estimates \(\mathrm{Tr}\log(g+\gamma_t I)\) with stochastic Lanczos
quadrature.

### Near-null mass

Define the projected trace surrogate
$$
\mathrm{Tr}_{\mathrm{proj}}(z) := \sum_{j=1}^{d_{\mathrm{proj}}}\tilde{\lambda}_j(z).
$$

Define a threshold
$$
\theta(z) := \tau \cdot \max\left(\frac{\mathrm{Tr}_{\mathrm{proj}}(z)}{d_z}, \varepsilon\right),
\qquad \tau = \texttt{config.near\_null\_tau},
$$
and report
$$
m_{\mathrm{null}}(z)
:= \frac{1}{d_{\mathrm{proj}}}
\sum_{j=1}^{d_{\mathrm{proj}}}\mathbf{1}\{\tilde{\lambda}_j(z) < \theta(z)\}.
$$

## C. Hessian Frobenius Proxy (Extrinsic Curvature)

Let $D_X(z) = (D_1(z),\dots,D_m(z))$ be the decoder outputs.
For each output coordinate, define its Hessian
$$
H_a(z) := \nabla^2_z D_a(z) \in \mathbb{R}^{d_z \times d_z}.
$$

The implementation targets the aggregated second-derivative energy
$$
\|H D_X(z)\|_F^2 := \sum_{a=1}^{m} \|H_a(z)\|_F^2.
$$

### Two-level Hutchinson estimator

Draw independent Rademacher vectors
$$
r \in \{\pm 1\}^{m},\qquad v \in \{\pm 1\}^{d_z}.
$$
Define a random scalar functional
$$
s(z) := r^{\top} D_X(z).
$$
Then $\nabla^2_z s(z) = \sum_{a=1}^{m} r_a H_a(z)$, and the code computes
$$
h(z;r,v) := \|\nabla^2_z s(z)\, v\|_2^2.
$$

With $\mathbb{E}[r r^{\top}] = I$ and $\mathbb{E}[v v^{\top}] = I$,
$$
\mathbb{E}_{r,v}\big[h(z;r,v)\big]
= \sum_{a=1}^{m}\|H_a(z)\|_F^2
= \|H D_X(z)\|_F^2.
$$

Practically, for each $z_i$ the implementation averages $h(z_i;r,v)$ over
`\texttt{config.n_hvp_probes}` draws, producing one scalar per $z_i$; then it
reports quantiles (median, $p90$, $p99$) across the sampled $\{z_i\}$.

## D. Aggregation Over Time/Scale Index

Let $t \in \{0,\dots,T-1\}$ index the modeled time/scale marginals.
From encoder-produced latent codes at each $t$, sample a subset
$\{z_{t,i}\}_{i=1}^{n_t}$ (without replacement, with $n_t \le \texttt{config.n_samples}$).

For each metric $M(z)$ (e.g., $\widehat{\mathrm{Tr}(g)}(z)$, $\widehat{r_{\mathrm{eff}}}(z)$,
$\rho_{\mathrm{vol},\gamma}(z)$, $m_{\mathrm{null}}(z)$), the per-time
reported mean is
$$
\overline{M}_t := \frac{1}{n_t}\sum_{i=1}^{n_t} M(z_{t,i}).
$$

The code also reports an approximate 95% confidence interval across latent
samples at fixed $t$ using a normal approximation:
$$
\mathrm{CI}_{95}(M)_t
= \overline{M}_t \pm 1.96 \cdot \frac{s_t}{\sqrt{n_t}},
$$
where $s_t$ is the sample standard deviation of $\{M(z_{t,i})\}_{i=1}^{n_t}$
(with $n_t>1$).

## E. Heuristic Risk Flags

Let $d_z$ be the latent dimension and let the per-time means be denoted by
$\overline{m_{\mathrm{null}}}_t$ and $\overline{r_{\mathrm{eff}}}_t$ etc.

At each time index, the code flags:

1. **Collapse risk**:
$$
\texttt{collapse\_risk}_t :=
\mathbf{1}\left\{
\overline{m_{\mathrm{null}}}_t > 0.5
\;\;\text{ or }\;\;
\overline{r_{\mathrm{eff}}}_t < 0.35\, d_z
\right\}.
$$

2. **Folding risk** (heavy tails in Hessian proxy):
$$
\texttt{folding\_risk}_t :=
\mathbf{1}\left\{
p99_t > \max(10 \cdot \text{median}_t,\varepsilon)
\right\}.
$$
Overall flags are the logical OR across $t$ for collapse and folding.

## F. Pairwise Relative Improvement (Used in Comparison Script)

In `compare_latent_geometry_models.py`, the maintained comparison is a fixed
baseline/treatment pair. Relative changes are reported for a baseline $b$ and
treatment $t$ using per-metric direction conventions:

1. If "higher is better":
$$
\Delta_{\mathrm{rel}} := \frac{t - b}{|b| + 10^{-12}}.
$$

2. If "lower is better":
$$
\Delta_{\mathrm{rel}} := \frac{b - t}{|b| + 10^{-12}}.
$$

The maintained metric directions are:

- higher is better: `trace_g`, effective rank, `rho_vol`
- lower is better: near-null mass, Hessian Frobenius `p99`

These sign conventions are written into the pairwise summary payload and the
pairwise delta table so positive `\Delta_{\mathrm{rel}}` always means
"treatment improved relative to baseline".
