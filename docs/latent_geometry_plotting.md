# Latent Geometry Plotting: Definitions and Style

This document centralizes the **metric definitions** and **plot styling conventions** for the latent-geometry evaluation, so figures can remain clean (no overlaid definition text).

Relevant code:

- Per-run evaluation plots: `scripts/fae/tran_evaluation/report.py`
- Cross-model comparison plots + effect tables: `scripts/fae/tran_evaluation/compare_latent_geometry_models.py`
- Metric computation: `scripts/fae/tran_evaluation/latent_geometry.py`
- Latent scatter diagnostics: `scripts/fae/tran_evaluation/visualize_autoencoder_latent_space.py`

## 1. Notation

Let the decoder (evaluated on a fixed coordinate set \(X\)) define
$$
D_X : \mathbb{R}^{d_z} \to \mathbb{R}^{m},\qquad D_X(z)=\mathrm{vec}(D(z,X)),
$$
with Jacobian
$$
J(z) := \partial_z D_X(z)\in\mathbb{R}^{m\times d_z}.
$$
The (discretized) pullback metric is
$$
g(z) := J(z)^\top J(z)\in\mathbb{R}^{d_z\times d_z}.
$$

Throughout, metrics are computed pointwise over sampled latents \(z\) and then aggregated over modeled time indices \(t\).

## 2. Metrics (y-axes)

### 2.1 Pullback trace

Per-latent quantity:
$$
\mathrm{Tr}(g(z)).
$$

Cross-model summaries typically plot the time mean
$$
\langle \mathrm{Tr}(g)\rangle_t.
$$

### 2.2 Effective rank (participation ratio)

Using the participation-ratio definition:
$$
r_{\mathrm{eff}}(g) := \frac{\mathrm{Tr}(g)^2}{\mathrm{Tr}(g^2)}.
$$

Cross-model summaries typically plot
$$
\langle r_{\mathrm{eff}}(g)\rangle_t.
$$

### 2.3 Volumetric robustness

Using the regularized volumetric robustness score:
$$
\rho_{\mathrm{vol},\gamma}
:=
\frac{\det(g+\gamma I)^{1/d_z}}
{\mathrm{Tr}(g)/d_z+\gamma},
$$
where \(\gamma>0\) is a shared ridge chosen per time index.

The implementation estimates \(\mathrm{Tr}\log(g+\gamma I)\) with stochastic
Lanczos quadrature and then reconstructs the regularized geometric mean
\(\det(g+\gamma I)^{1/d_z}\).

Cross-model summaries currently plot
$$
\langle \rho_{\mathrm{vol},\gamma}\rangle_t \in (0,1].
$$

### 2.4 Near-null mass (projected spectrum)

Define a near-null indicator on projected eigenvalues \(\tilde\lambda_j\) via a relative threshold:
$$
m_{\mathrm{null}}
  := \frac{1}{d_{\mathrm{proj}}}\sum_{j=1}^{d_{\mathrm{proj}}}
     \mathbf{1}\{\tilde{\lambda}_j < \tau \max(\mathrm{Tr}_{\mathrm{proj}}/d_z,\varepsilon)\}.
$$

Cross-model summaries typically plot
$$
\langle m_{\mathrm{null}}\rangle_t \in [0,1].
$$

### 2.5 Curvature proxy (decoder Hessian energy)

Define the Hessian-energy proxy of the decoder map \(D_X\):
$$
\|H_z D\|_F^2 := \sum_{a=1}^{m}\|\nabla_z^2 D_a(z)\|_F^2.
$$

In the per-run plots, we show quantiles (median, P90, P99) over sampled latents.

In cross-model summaries we plot a time-mean of a tail quantile (currently \(q_{0.99}\)):
$$
\langle q_{0.99}(\|H_z D\|_F^2)\rangle_t,
$$
typically on a log y-scale.

## 3. Robustness flags

These are plotted as indicator-valued diagnostics:

### 3.1 Collapse risk

Flags if the latent geometry is “too null” by either criterion:
$$
\mathbb{I}\{\mathrm{collapse}\}=1
\quad\text{if}\quad
m_{\mathrm{null}}>0.5
\;\;\text{or}\;\;
r_{\mathrm{eff}}(g)<0.35\,d_z.
$$

### 3.2 Folding risk

Flags if the curvature tail is heavy:
$$
\mathbb{I}\{\mathrm{folding}\}=1
\quad\text{if}\quad
q_{0.99}(\|H_z D\|_F^2)>\max(10\cdot \mathrm{median}(\|H_z D\|_F^2),\varepsilon).
$$

## 4. Styling conventions

### 4.1 Palette (ChromaPalette `EasternHues`)

Publication plots use the ChromaPalette `EasternHues` palette:

`[#D9AB42, #A35E47, #0F4C3A, #78C2C4, #C73E3A, #563F2E, #B47157, #2B5F75]`.

Conventions used in this repo:

- `C_OBS = #2B5F75` (steel blue)
- `C_GEN = #C73E3A` (red)

### 4.2 Cross-model comparisons

- Model-metric bar plots: color encodes the **configuration** (loss/prior/scale/decoder), shared across optimizers; optimizer category is a hatch (ADAM = none, MUON = `//`).
- Canonical chain plots: color encodes the **stage** (L2/NTK/NTK+Prior), shared across optimizers; optimizer category is a linestyle (ADAM = solid, MUON = dashed).

### 4.3 Per-run latent-geometry plots

- Effective rank uses the base blue.
- Volumetric robustness uses the base red.
- Near-null mass uses a secondary accent (currently `tab:purple`).
- Flags use green/red traffic-light colormap.

### 4.4 Sequential colormap

Sequential time-ordered latent scatter plots and publication field-image panels use `cmaps.bamako` from `import colormaps as cmaps`.

- This replaces the older `viridis` default.
- The latent-128 publication execution flow is documented in [docs/publication_latent128_msbm_execution.md](publication_latent128_msbm_execution.md).
