# FAE Experiment Scripts

Organized by purpose. All scripts assume `conda activate 3MASB`.

## Folder Structure

```
experiments/
├── ablation_adam/               # Main ablation table (Paper Table 1)
│   ├── adam_l2.sh               # A1: Baseline — no gradient correction
│   ├── adam_ntk.sh              # A2: + NTK per-time trace-equalization
│   ├── adam_prior.sh            # A3: + Latent diffusion prior
│   └── adam_ntk_prior.sh        # A4: + NTK + Prior (proposed method)
│
├── ablation_muon/               # Pareto / supplementary (Appendix)
│   ├── muon_l2.sh               # M1: Muon only (Type I)
│   ├── muon_ntk.sh              # M2: Muon + NTK (Type I + II)
│   ├── muon_prior.sh            # M3: Muon + Prior (Type I + III)
│   └── muon_ntk_prior.sh        # M4: Muon + NTK + Prior (full stack)
│
├── denoiser/                    # Iterative decoder experiments
│   ├── denoiser_heek.sh         # D1: Heek et al. protocol (50K)
│   ├── denoiser_heek_150k.sh    # D2: Extended training (150K)
│   ├── denoiser_muon_ntk_prior.sh  # D3: Full stack + denoiser
│   ├── denoiser_adam_ntk_prior.sh  # D3-Adam: Adam variant of D3
│   ├── film_velocity.sh         # FiLM velocity loss (no prior)
│   ├── film_velocity_prior.sh   # FiLM velocity + latent diffusion prior
│   ├── film_x0.sh               # FiLM x0 loss (ablation vs velocity)
│   └── scaled_velocity.sh       # ScaledDenoiserDecoder velocity loss
│
├── pipeline/                    # End-to-end MSBM & orchestrators
│   ├── latent_msbm_pipeline.sh  # Train + generate trajectories + eval
│   ├── latent_msbm_evaluation.sh# Post-training evaluation suite
│   ├── fae_pipeline.sh          # Data generation + basic FAE training
│   └── spacetime_bias_prototype.sh  # Multi-phase prototype w/ path biasing
│
└── preliminary/                 # Single-scale predecessors & exploratory
    ├── deterministic_film.sh    # Original Muon+L2 baseline (single-scale)
    ├── film_adam_l2.sh          # Adam+L2 (single-scale)
    ├── film_adam_ntk.sh         # Adam+NTK (single-scale)
    ├── film_muon_ntk.sh         # Muon+NTK (single-scale)
    ├── film_prior.sh            # Muon+Prior (single-scale)
    ├── latent_noise_sweep.sh    # Latent noise scale sweep (σ ∈ {0.01..0.075})
    └── standard_dual_stream_bottleneck.sh  # Standard decoder + dual-stream pooling
```

## Ablation Design

### Main table: 2×2 factorial on Adam (Paper §X)

|          | No Prior    | Prior       |
|----------|:-----------:|:-----------:|
| **L2**   | A1 baseline | A3          |
| **NTK**  | A2          | **A4** (ours) |

### Pareto analysis: Adam vs Muon (Paper §X / Appendix)

Muon (Jordan 2024) applies SVD-based spectral gradient normalization.
While this has theoretical benefits for multi-scale optimization (Type I:
gradient direction), empirical results show it degrades reconstruction
fidelity relative to Adam. The Pareto frontier of PSD fidelity vs latent
quality (effective rank, variance spread) characterizes this trade-off.

## Existing Results

| Script         | Run ID        | Output Dir |
|----------------|---------------|------------|
| A1 adam_l2     | `run_bnqm4evk` | `results/fae_film_adam_l2_99pct` |
| A2 adam_ntk    | `run_2hnr5shv` | `results/fae_film_adam_ntk_99pct` |
| A3 adam_prior  | —              | `results/fae_film_adam_prior` |
| A4 adam_ntk_prior | —           | `results/fae_film_adam_ntk_prior` |
| M1 muon_l2     | `run_qffzfzrj` | `results/fae_film_muon_99pct` |
| M2 muon_ntk    | `run_tug7ucuw` | `results/fae_film_muon_ntk_99pct` |
| M3 muon_prior  | `run_66nrnp5e` | `results/fae_film_prior_multiscale` |
| M4 muon_ntk_prior | —          | `results/fae_film_muon_ntk_prior` |
| D1 denoiser    | `run_9vl5sblh` | `results/fae_denoiser_film_heek_multiscale` |
