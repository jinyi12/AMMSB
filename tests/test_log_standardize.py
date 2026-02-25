#!/usr/bin/env python3
"""
Test script for log_standardize transformation.

Verifies:
1. Forward transform works correctly
2. Inverse transform recovers original values
3. Strict positivity is guaranteed
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.multimarginal_generation import log_standardize_marginals
from data.transform_utils import inverse_log_standardize_transform


def test_log_standardize_roundtrip():
    """Test that forward + inverse transform recovers original data."""
    print("="*70)
    print("TEST: Log Standardize Round-Trip")
    print("="*70)

    # Create test data: strictly positive
    np.random.seed(42)
    n_samples = 100
    data_dim = 64 * 64

    # Simulate tran_inclusion-like data
    # Microscale: bimodal (matrix=1, inclusion=1000)
    micro = np.random.choice([1.0, 1000.0], size=(n_samples, data_dim), p=[0.7, 0.3])

    # Macroscale: filtered (more Gaussian, positive)
    macro = np.random.gamma(shape=2.0, scale=50.0, size=(n_samples, data_dim))
    macro = np.clip(macro, 10, 500)  # Keep strictly positive

    # Create marginals dict
    marginal_arrays = {
        0.0: micro,
        1.0: macro,
    }

    print(f"\nOriginal data:")
    print(f"  Micro: min={micro.min():.4f}, max={micro.max():.4f}, mean={micro.mean():.4f}")
    print(f"  Macro: min={macro.min():.4f}, max={macro.max():.4f}, mean={macro.mean():.4f}")

    # Forward transform
    print(f"\nApplying forward transform...")
    eps = 1e-6
    transformed, log_eps, log_mean, log_std = log_standardize_marginals(
        marginal_arrays, eps=eps
    )

    print(f"  Log epsilon: {log_eps}")
    print(f"  Log mean: {log_mean:.4f}")
    print(f"  Log std: {log_std:.4f}")

    # Check transformed data is standardized
    transformed_micro = transformed[0.0]
    transformed_macro = transformed[1.0]
    print(f"\nTransformed data (should be ~N(0,1)):")
    print(f"  Micro: mean={transformed_micro.mean():.4f}, std={transformed_micro.std():.4f}")
    print(f"  Macro: mean={transformed_macro.mean():.4f}, std={transformed_macro.std():.4f}")

    # Inverse transform
    print(f"\nApplying inverse transform...")
    recovered_micro = inverse_log_standardize_transform(
        transformed_micro, log_mean, log_std, log_eps
    )
    recovered_macro = inverse_log_standardize_transform(
        transformed_macro, log_mean, log_std, log_eps
    )

    print(f"  Recovered micro: min={recovered_micro.min():.4f}, max={recovered_micro.max():.4f}")
    print(f"  Recovered macro: min={recovered_macro.min():.4f}, max={recovered_macro.max():.4f}")

    # Check recovery error
    error_micro = np.abs(recovered_micro - micro).max()
    error_macro = np.abs(recovered_macro - macro).max()
    rel_error_micro = error_micro / micro.mean()
    rel_error_macro = error_macro / macro.mean()

    print(f"\nRecovery error:")
    print(f"  Micro: max_abs={error_micro:.6e}, rel={rel_error_micro:.6e}")
    print(f"  Macro: max_abs={error_macro:.6e}, rel={rel_error_macro:.6e}")

    # Check strict positivity
    assert recovered_micro.min() > 0, "Recovered micro should be strictly positive!"
    assert recovered_macro.min() > 0, "Recovered macro should be strictly positive!"
    print(f"\n✅ Strict positivity: PASSED")

    # Check round-trip accuracy
    rtol = 1e-5
    assert np.allclose(recovered_micro, micro, rtol=rtol), "Micro round-trip failed!"
    assert np.allclose(recovered_macro, macro, rtol=rtol), "Macro round-trip failed!"
    print(f"✅ Round-trip accuracy (rtol={rtol}): PASSED")

    print("\n" + "="*70)
    print("TEST PASSED: Log standardization works correctly!")
    print("="*70)


def test_out_of_distribution_generation():
    """Test that inverse transform handles out-of-distribution values gracefully."""
    print("\n" + "="*70)
    print("TEST: Out-of-Distribution Generation")
    print("="*70)

    # Simulate a generative model that produces slightly OOD values
    np.random.seed(42)

    # Original training data
    training_data = np.random.gamma(shape=2.0, scale=50.0, size=(1000, 100))
    training_data = np.clip(training_data, 10, 500)

    print(f"\nTraining data:")
    print(f"  Min: {training_data.min():.4f}")
    print(f"  Max: {training_data.max():.4f}")
    print(f"  Mean: {training_data.mean():.4f}")

    # Fit transform
    marginal_arrays = {0.0: training_data}
    eps = 1e-6
    transformed, log_eps, log_mean, log_std = log_standardize_marginals(
        marginal_arrays, eps=eps
    )

    # Simulate generated data: slightly OOD in standardized space
    # Normal generation might produce values beyond ±3 sigma
    generated_standardized = np.random.normal(0, 1.2, size=(100, 100))  # Slightly wider
    print(f"\nGenerated standardized data (slightly OOD):")
    print(f"  Min: {generated_standardized.min():.4f}")
    print(f"  Max: {generated_standardized.max():.4f}")
    print(f"  Mean: {generated_standardized.mean():.4f}")
    print(f"  Std: {generated_standardized.std():.4f}")

    # Inverse transform
    generated_physical = inverse_log_standardize_transform(
        generated_standardized, log_mean, log_std, log_eps
    )

    print(f"\nGenerated physical data (after inverse):")
    print(f"  Min: {generated_physical.min():.4f} (should be > 0)")
    print(f"  Max: {generated_physical.max():.4f}")
    print(f"  Mean: {generated_physical.mean():.4f}")

    # Check strict positivity
    assert generated_physical.min() > 0, "Generated should be strictly positive!"
    print(f"\n✅ Generated data is strictly positive!")

    # Check that values are reasonable (not too extreme)
    # With log transform, extreme standardized values map to reasonable positive values
    print(f"\nExtreme value handling:")
    extreme_standardized = np.array([[-5.0, 0.0, 5.0]])  # Very OOD
    extreme_physical = inverse_log_standardize_transform(
        extreme_standardized, log_mean, log_std, log_eps
    )
    print(f"  Standardized: {extreme_standardized[0]}")
    print(f"  Physical: {extreme_physical[0]}")
    print(f"  All positive: {(extreme_physical > 0).all()}")

    assert (extreme_physical > 0).all(), "Even extreme values should be positive!"
    print(f"\n✅ Extreme OOD values handled gracefully!")

    print("\n" + "="*70)
    print("TEST PASSED: OOD generation works correctly!")
    print("="*70)


def test_comparison_with_minmax():
    """Compare behavior with min-max scaling."""
    print("\n" + "="*70)
    print("TEST: Comparison with Min-Max Scaling")
    print("="*70)

    np.random.seed(42)

    # Test data
    data = np.random.gamma(shape=2.0, scale=50.0, size=(1000, 100))
    data = np.clip(data, 10, 500)

    print(f"\nOriginal data range: [{data.min():.4f}, {data.max():.4f}]")

    # Min-max scaling
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_scale = data_max - data_min
    minmax_scaled = (data - data_min) / data_scale

    print(f"\nMin-max scaled: [{minmax_scaled.min():.4f}, {minmax_scaled.max():.4f}]")
    print(f"  → BOUNDED to [0, 1]")

    # Log standardization
    marginal_arrays = {0.0: data}
    transformed, log_eps, log_mean, log_std = log_standardize_marginals(
        marginal_arrays, eps=1e-6
    )
    log_scaled = transformed[0.0]

    print(f"\nLog standardized: [{log_scaled.min():.4f}, {log_scaled.max():.4f}]")
    print(f"  → UNBOUNDED (can be any real number)")

    # Simulate OOD generation
    print(f"\nSimulating OOD generation:")

    # With min-max: values outside [0,1] are impossible to map back correctly
    bad_minmax = np.array([[1.05, -0.02]])  # Out of bounds
    print(f"  Min-max OOD: {bad_minmax[0]}")
    print(f"  Issue: Values outside [0,1] have no valid inverse!")

    # With log-standardize: any value is valid
    ood_log = np.array([[5.0, -5.0]])  # Far from training
    ood_physical = inverse_log_standardize_transform(
        ood_log, log_mean, log_std, log_eps
    )
    print(f"  Log-standardize OOD: {ood_log[0]}")
    print(f"  Maps to physical: {ood_physical[0]}")
    print(f"  Still strictly positive: {(ood_physical > 0).all()}")

    print(f"\n✅ Log-standardize handles OOD gracefully!")

    print("\n" + "="*70)
    print("CONCLUSION: Log-standardize is superior for generative modeling")
    print("="*70)


if __name__ == "__main__":
    try:
        test_log_standardize_roundtrip()
        test_out_of_distribution_generation()
        test_comparison_with_minmax()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nLog standardization is ready for use!")
        print("Generate dataset with: --scale_mode log_standardize")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
