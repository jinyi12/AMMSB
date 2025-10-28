# PCA Field Reconstruction Visualizations

This module provides visualization tools for reconstructed random fields from PCA coefficient trajectories in the multi-marginal flow matching framework.

## Overview

For PCA-based applications (e.g., `pca_main.py`), the learned trajectories exist in PCA coefficient space. To better understand the learned dynamics, we can reconstruct the original 2D random fields and visualize them.

## How It Works

1. **PCA Coefficients → Original Space**: The inverse PCA transformation is applied:
   ```
   x = [φ][μ]^(1/2)[η] + [x̄]
   ```
   where:
   - `[η]` are the PCA coefficients (trajectory outputs)
   - `[φ]` are the PCA eigenvectors
   - `[μ]` are the eigenvalues
   - `[x̄]` is the mean

2. **Flattened → 2D Fields**: The reconstructed data is reshaped from flattened vectors to 2D spatial fields

3. **Visualization**: Multiple visualization types show the reconstructed fields

## New Visualizations

The following visualizations are automatically generated when running `pca_main.py`:

### 1. Field Snapshots (`field_snapshots_{ode/sde}.png`)
- Grid showing multiple sample fields at different time points
- Rows = different samples, Columns = different time points
- Consistent color scaling across all fields
- Helps visualize how spatial structures evolve over time

### 2. Field Evolution GIF (`field_evolution_{ode/sde}_sample{N}.gif`)
- Animated evolution of individual sample fields through time
- One GIF per sample (default: samples 0, 1, 2)
- Shows smooth temporal evolution of spatial patterns

### 3. Field Statistics (`field_statistics_{ode/sde}.png`)
- Comparison of mean and standard deviation between generated and test fields
- Two subplots:
  - Mean field value over time
  - Standard deviation over time
- Validates that generated fields match test data statistics

### 4. Spatial Correlation (`spatial_correlation_{ode/sde}.png`)
- 2D autocorrelation structure at different time points
- Shows how spatial correlations evolve with coarse-graining
- Averaged over multiple samples for stability

## Files Added

- **`pca_field_plot.py`**: Core module with all field reconstruction and visualization functions
- **`README_PCA_VISUALIZATIONS.md`**: This documentation file

## Files Modified

- **`plotter.py`**: Updated `plot_all()` to accept optional `pca_info` parameter and call field visualizations
- **`pca_main.py`**: Modified to pass `pca_info` to plotter

## Usage

When running `pca_main.py`, the field visualizations are automatically generated if PCA info is available:

```bash
PYTHONPATH="$PWD" python ./scripts/pca_main.py \
    --data_path ./data/mm_data.npz \
    --flowmatcher sb \
    --agent_type triplet \
    --spline cubic \
    --monotonic \
    --modelname mlp
```

The visualizations will be:
1. Saved in the output directory (`results/...`)
2. Logged to Weights & Biases under:
   - `visualizations/field_snapshots_{ode/sde}`
   - `visualizations/field_evolution_{ode/sde}_sample{N}`
   - `evalplots/field_statistics_{ode/sde}`
   - `evalplots/spatial_correlation_{ode/sde}`

## Key Functions

### `reconstruct_fields_from_coefficients(coeffs, pca_info, resolution)`
Reconstructs 2D fields from PCA coefficients.

### `visualize_all_field_reconstructions(traj_coeffs, testdata, pca_info, zt, outdir, run, score)`
Main entry point that creates all field visualizations.

### Individual Plot Functions
- `plot_field_snapshots()`: Grid of fields at multiple timepoints
- `plot_field_evolution_gif()`: Animated field evolution
- `plot_field_statistics()`: Statistical comparison plots
- `plot_spatial_correlation()`: Spatial autocorrelation analysis

## Benefits

1. **Physical Interpretation**: See the actual random fields being generated, not just abstract PCA coefficients
2. **Validation**: Verify that spatial structures and statistics match the test data
3. **Debugging**: Identify if/when the model generates unrealistic spatial patterns
4. **Understanding**: Better intuition for how coarse-graining affects field structure over time

## Dependencies

- numpy
- matplotlib
- scipy (for spatial correlation analysis)
- wandb

All dependencies are already in the project requirements.
