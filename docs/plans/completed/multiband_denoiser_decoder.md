# Multi-Band Diffusion Denoiser Decoder

**Status**: Planning
**Date**: 2026-02-21

---

## Motivation

The current `DiffusionDenoiserDecoder` uses a single MLP backbone to predict the denoised field at all frequency scales simultaneously. For datasets with multi-scale structure (large and small inclusions, successively smoothed fields), this forces one network to learn both coarse geometry and fine detail — leading to spectral bias where high-frequency features are underrepresented.

The multi-band decoder decomposes the output into S+1 frequency branches: one base (DC/coarse) branch and S detail branches, connected by coarse-to-fine cascading and per-layer learnable high-frequency scaling (HFS). The final output is a fixed sum of all branch predictions.

---

## Mathematical Formulation

### Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $x \in \Omega$ | $[B, N, 2]$ | Spatial coordinates |
| $t \in [0,1]$ | $[B]$ | Diffusion timestep |
| $z \in \mathbb{R}^d$ | $[B, d]$ | Latent code from FAE encoder |
| $z_t$ | $[B, N, 1]$ | Noisy field at time $t$ |
| $j \in \{0,\dots,S\}$ | — | Branch index ($j=0$: base; $j>0$: detail) |
| $l \in \{1,\dots,L\}$ | — | Layer index within a branch |
| $C_\text{trunk}$ | — | Shared trunk width |
| $C_\text{branch}$ | — | Per-branch hidden width |

### 1. Input Encoding

**Random Fourier Features (RFF) positional encoding**:
$$\gamma(x) = \left[\cos(2\pi B x),\; \sin(2\pi B x)\right], \quad B \sim \mathcal{N}(0, \sigma^2 I)$$

**Sinusoidal time embedding**:
$$\gamma(t)_i = \begin{cases} \sin\!\left(t \cdot e^{-i \log(10000) / (d/2)}\right) & i < d/2 \\ \cos\!\left(t \cdot e^{-(i-d/2) \log(10000) / (d/2)}\right) & i \geq d/2 \end{cases}$$

**Tile latent and broadcast**:
$$\tilde{z} = z \cdot \mathbf{1}^{\top} \in \mathbb{R}^{B \times N \times d}, \quad \tilde{\gamma}(t) = \gamma(t) \cdot \mathbf{1}^{\top} \in \mathbb{R}^{B \times N \times d_t}$$

### 2. Shared Coordinate Trunk

The trunk ingests the full per-point feature vector and produces a foundational representation $H_0$:

$$H_0 = \Phi_\text{trunk}\!\left([\tilde{z} \,\|\, \gamma(x) \,\|\, z_t \,\|\, \tilde{\gamma}(t)]\right) \in \mathbb{R}^{B \times N \times C_\text{trunk}}$$

where $\Phi_\text{trunk}$ is an MLP with $L_\text{trunk}$ hidden layers of width $C_\text{trunk}$, GELU activations, and optional LayerNorm:

$$H_0^{(l)} = \text{GELU}\!\left(\text{Norm}\!\left(W_l^{\text{trunk}} H_0^{(l-1)} + b_l^{\text{trunk}}\right)\right)$$

### 3. Branch Initialization and Coarse-to-Fine Cascade

All branches start from the same trunk output $H_0$. Let $h_l^{(j)}$ denote the hidden state at layer $l$ in branch $j$.

**Base branch ($j = 0$)**:
$$h_l^{(0)} = \text{GELU}\!\left(\text{Norm}\!\left(W_l^{(0)} h_{l-1}^{(0)} + b_l^{(0)}\right)\right)$$

**Detail branches ($j > 0$)** — coarse-to-fine cascade via concatenation:
$$h_l^{(j)} = \text{GELU}\!\left(\text{Norm}\!\left(W_l^{(j)} \left[h_{l-1}^{(j)} \,\|\, W_\text{trans}^{(j,l)} h_l^{(j-1)}\right] + b_l^{(j)}\right)\right)$$

Here $W_\text{trans}^{(j,l)} \in \mathbb{R}^{C_\text{branch} \times C_\text{branch}}$ is a learned transition projection that maps the same-layer hidden state of the preceding coarser branch into the current branch's feature space. The cascade ensures that the spatial blueprint established by low-frequency branches anchors the placement of high-frequency details.

### 4. Layer-wise Feature Map Decomposition (Spectral HFS)

Inside each layer of detail branches ($j > 0$), we apply channel-wise High-Frequency Scaling (HFS). Split the hidden feature vector along the channel dimension:

$$h_l^{(j)} = \left[h_\text{base} \,\|\, h_\text{detail}\right], \quad h_\text{base} \in \mathbb{R}^{(1-\rho) C_\text{branch}}, \quad h_\text{detail} \in \mathbb{R}^{\rho C_\text{branch}}$$

where $\rho = $ `hfs_ratio` is the fraction of channels designated as the detail subspace.

A small conditioning MLP computes per-channel scaling from the latent code and timestep:

$$\lambda_l^{(j)}(z, t) = \text{MLP}_\text{HFS}\!\left([z \,\|\, \gamma(t)]\right) \in \mathbb{R}^{\rho C_\text{branch}}$$

This is broadcast across all $N$ spatial points (frequency filtering is global, not spatially varying).

The detail subspace is scaled while the base subspace is passed through unchanged:

$$\tilde{h}_\text{detail} = \lambda_l^{(j)}(z,t) \odot h_\text{detail}$$

$$\tilde{h}_l^{(j)} = \left[h_\text{base} \,\|\, \tilde{h}_\text{detail}\right]$$

### 5. Multi-Head Output and Fixed-Sum Blending

Each branch has a dedicated linear readout head:

$$F_S(x) = W_\text{out}^{(0)} h_L^{(0)}, \qquad D_j(x) = W_\text{out}^{(j)} h_L^{(j)}, \quad j = 1,\dots,S$$

The final predicted field is the fixed sum of all branches:

$$\hat{u}(x) = F_S(x) + \sum_{j=1}^{S} D_j(x)$$

No learnable blending weights are used at the output stage — this prevents detail branch suppression. The HFS scaling $\lambda_l^{(j)}$ provides the only learned modulation of frequency content.

### 6. Diffusion Training Objective (unchanged from base decoder)

The noisy mixture at diffusion time $t$:
$$z_t = (1 - t)\, u_\text{clean} + t\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

Velocity in the rectified flow formulation:
$$v = \frac{z_t - u_\text{clean}}{t}$$

The decoder predicts $\hat{u} = \hat{x}_0 = \text{predict\_x}(z, x, z_t, t)$, from which the predicted velocity is:
$$\hat{v} = \frac{z_t - \hat{u}}{t}$$

Training loss:
$$\mathcal{L} = \underbrace{\mathbb{E}_{t,\varepsilon}\left[\|v - \hat{v}\|^2\right]}_{\text{velocity loss}} + w_0 \underbrace{\mathbb{E}\left[\|u_\text{clean} - \hat{u}\|^2\right]}_{\text{x0 anchor}} + \underbrace{\beta \,\mathbb{E}\!\left[\|z\|^2\right]}_{\text{latent reg.}}$$

where latent regularization follows the project convention: $\beta \cdot \mathbb{E}_\text{batch}\!\left[\sum_d z_d^2\right]$.

---

## Architecture Diagram

```
  Input per point: [z̃, γ(x), z_t, γ̃(t)]
               ↓
  ┌────────────────────────────────┐
  │       Shared Trunk Φ_trunk     │  L_trunk layers, width C_trunk
  └────────────┬───────────────────┘
               │  H₀ ∈ R^{B×N×C_trunk}
     ┌─────────┼──────────┐
     ↓         ↓          ↓
  ┌───────┐ ┌────────┐ ┌────────┐
  │Base   │ │Detail 1│ │Detail 2│   ... (S branches)
  │ j=0   │→│  j=1   │→│  j=2   │   (coarse-to-fine cascade →)
  │       │ │ + HFS  │ │ + HFS  │
  │L_br   │ │L_br    │ │L_br    │   (L_branch layers each)
  │layers │ │layers  │ │layers  │
  └───┬───┘ └───┬────┘ └───┬────┘
      ↓         ↓          ↓
   W_out⁰    W_out¹     W_out²
      └─────────┴──────────┘
                 ↓  SUM
             x̂ = F_S(x) + Σ D_j(x)   [B, N, out_dim]
```

---

## Implementation Plan

### Files to Create / Modify

| File | Action | Change |
|------|--------|--------|
| `scripts/fae/fae_naive/multiband_denoiser_decoder.py` | **CREATE** | Core decoder class (~200 lines) |
| `scripts/fae/fae_naive/decoder_builders.py` | **MODIFY** | Add builder + dispatch case (~40 lines) |
| `scripts/fae/fae_naive/train_attention_components.py` | **MODIFY** | Pass-through new kwargs (~15 lines) |
| `scripts/fae/fae_naive/train_attention_denoiser.py` | **MODIFY** | CLI args + validation + wiring (~50 lines) |

---

### Step 1: `multiband_denoiser_decoder.py` (CREATE)

**Inheritance**: `MultiBandDiffusionDenoiserDecoder(DiffusionDenoiserDecoder)`

Override only `setup()` and `predict_x()`. All diffusion sampling logic is inherited:
- `sample()`, `sample_trajectory()`, `one_step_generate()` — Euler ODE/SDE via `jax.lax.scan`
- `predict_x_from_mixture()` — calls `_mix_with_noise` then `predict_x`
- `_forward()` — init proxy (zeros noisy field, t = time_eps)
- `_time_embedding()`, `_mix_with_noise()`, `_v_from_xz()`, `_make_time_grid()`

**New dataclass fields** (additive to parent's `features`, `architecture`, `scaling`, etc.):

```python
n_bands: int = 3              # Total branches = 1 base + (n_bands-1) detail
trunk_features: int = 128     # Shared trunk hidden width
branch_features: int = 128    # Per-branch hidden width
n_trunk_layers: int = 3       # Number of trunk hidden layers
n_branch_layers: int = 2      # Number of hidden layers per branch
hfs_ratio: float = 0.5        # Fraction of branch channels for HFS detail subspace
```

**`setup()` builds** (all with explicit `name=` for Flax parameter stability):

```
trunk_dense_{l}          for l in 0..n_trunk_layers-1
trunk_norm_{l}           (if norm_type == "layernorm")

branch_{j}_dense_{l}     for j in 0..n_bands-1, l in 0..n_branch_layers-1
branch_{j}_norm_{l}      (if layernorm)

# Detail branches only:
trans_{j}_layer_{l}      nn.Dense(branch_features)  # coarser-branch projection
hfs_{j}_layer_{l}_fc1   nn.Dense(64)                # HFS scaling MLP, layer 1
hfs_{j}_layer_{l}_fc2   nn.Dense(n_detail_channels) # HFS scaling MLP, layer 2

output_head_{j}          nn.Dense(out_dim)           # per-branch readout
```

where `n_detail_channels = int(branch_features * hfs_ratio)`.

**`predict_x(z, x, noisy_field, t, train)` pseudocode**:

```python
x_enc = self.positional_encoding(x)          # [B, N, 2*n_freqs]
t_emb = self._time_embedding(t)              # [B, time_emb_dim]
t_emb_exp = broadcast(t_emb, [B, N, ...])
z_tiled  = broadcast(z,     [B, N, ...])

# --- Shared trunk ---
h = concat([z_tiled, x_enc, noisy_field, t_emb_exp], axis=-1)
for l in range(n_trunk_layers):
    h = trunk_dense[l](h)
    h = maybe_norm(h, trunk_norm[l])
    h = gelu(h)
H0 = h  # [B, N, trunk_features]

# --- Branch cascade ---
prev_hiddens = [None] * n_branch_layers   # per-layer hidden states of prev branch
branch_preds = []

for j in range(n_bands):
    h = H0
    cur_hiddens = []
    for l in range(n_branch_layers):
        if j > 0:
            # Coarse-to-fine: concat with projected previous branch's layer-l state
            coarse = trans[j][l](prev_hiddens[l])      # [B, N, branch_features]
            h = concat([h, coarse], axis=-1)

        h = branch_dense[j][l](h)                      # [B, N, branch_features]
        h = maybe_norm(h, branch_norm[j][l])

        if j > 0:
            # HFS: split into base/detail subspaces
            h_base   = h[..., :n_base]
            h_detail = h[..., n_base:]

            # Compute per-channel scaling from (z, t_emb) — global, not per-point
            lam_in = concat([z, t_emb], axis=-1)       # [B, d + d_t]
            lam = gelu(hfs_fc1[j][l](lam_in))
            lam = hfs_fc2[j][l](lam)                   # [B, n_detail]
            lam = broadcast(lam, [B, N, n_detail])

            h_detail = lam * h_detail
            h = concat([h_base, h_detail], axis=-1)

        h = gelu(h)
        cur_hiddens.append(h)

    prev_hiddens = cur_hiddens
    branch_preds.append(output_head[j](h))             # [B, N, out_dim]

return sum(branch_preds)                               # [B, N, out_dim]
```

---

### Step 2: `decoder_builders.py` (MODIFY)

```python
from scripts.fae.fae_naive.multiband_denoiser_decoder import MultiBandDiffusionDenoiserDecoder

def build_multiband_denoiser_decoder(
    out_dim, features, positional_encoding,
    n_bands=3, trunk_features=128, branch_features=128,
    n_trunk_layers=3, n_branch_layers=2, hfs_ratio=0.5,
    denoiser_time_emb_dim=32, denoiser_diffusion_steps=1000,
    denoiser_beta_schedule="cosine", denoiser_norm="layernorm",
    denoiser_sampler="ode", denoiser_sde_sigma=1.0,
) -> MultiBandDiffusionDenoiserDecoder:
    return MultiBandDiffusionDenoiserDecoder(
        out_dim=out_dim, features=features,
        positional_encoding=positional_encoding,
        n_bands=n_bands, trunk_features=trunk_features,
        branch_features=branch_features, n_trunk_layers=n_trunk_layers,
        n_branch_layers=n_branch_layers, hfs_ratio=hfs_ratio,
        time_emb_dim=denoiser_time_emb_dim,
        diffusion_steps=denoiser_diffusion_steps,
        beta_schedule=denoiser_beta_schedule,
        norm_type=denoiser_norm, sampler=denoiser_sampler,
        sde_sigma=denoiser_sde_sigma,
    )
```

Add `elif decoder_type == "multiband_denoiser":` to `build_decoder()` dispatch. Add new kwargs (`n_bands`, `trunk_features`, `branch_features`, `n_trunk_layers`, `n_branch_layers`, `hfs_ratio`) to `build_decoder()` signature with defaults.

---

### Step 3: `train_attention_components.py` (MODIFY)

Add to `build_autoencoder()` signature:
```python
n_bands: int = 3,
trunk_features: int = 128,
branch_features: int = 128,
n_trunk_layers: int = 3,
n_branch_layers: int = 2,
hfs_ratio: float = 0.5,
```

Pass all through to `build_decoder()`. Update `architecture_info` dict to record these when `decoder_type == "multiband_denoiser"`.

---

### Step 4: `train_attention_denoiser.py` (MODIFY)

**New CLI args** (after existing denoiser args):
```
--n-bands            int, default=3
--trunk-features     int, default=128
--branch-features    int, default=128
--n-trunk-layers     int, default=3
--n-branch-layers    int, default=2
--hfs-ratio          float, default=0.5
```

**Update**:
- `--decoder-type` choices: add `"multiband_denoiser"`
- `DENOISER_DECODER_TYPES`: add `"multiband_denoiser"` (ensures denoiser loss + metrics are used)
- Validation block for multiband args (check ranges, warn when set but decoder_type != multiband_denoiser)
- Pass args to `build_autoencoder()`

**No changes needed** to loss function, metrics, or reconstruction helpers — they all call `decoder.predict_x_from_mixture()` / `decoder.sample()` which are inherited.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Inherit from `DiffusionDenoiserDecoder` | ~200 lines of diffusion sampling reused; future improvements auto-apply |
| Override only `setup()` + `predict_x()` | All other methods call `self.predict_x()` internally |
| HFS scaling input is `(z, t_emb)`, not `(z, t, x)` | Frequency decomposition is global per-sample; keeps HFS MLP small O(batch) not O(batch·N) |
| Fixed sum blending | Prevents detail branch suppression; simplest stable starting point |
| Layer-wise cascade (not output-only) | More expressive: branches adapt internal representations, not just residuals |
| No per-band loss | Loss operates on aggregate prediction; simplifies training |

---

## Verification

```bash
# 1. Smoke test (small config)
python scripts/fae/fae_naive/train_attention_denoiser.py \
  --data-path <path.npz> \
  --output-dir /tmp/test_multiband \
  --decoder-type multiband_denoiser \
  --n-bands 3 --trunk-features 64 --branch-features 64 \
  --n-trunk-layers 2 --n-branch-layers 2 \
  --max-steps 50 --wandb-disabled

# 2. Run project tests
pytest tests/

# 3. Import check
python -c "from scripts.fae.fae_naive.multiband_denoiser_decoder import MultiBandDiffusionDenoiserDecoder"
```

Expected: loss decreases over 50 steps; all existing tests pass; shapes `[B, N, 1]` throughout.
