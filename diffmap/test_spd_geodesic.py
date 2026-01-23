import numpy as np
import scipy.linalg
from spd_geodesic import get_effective_dim, alpha_normalize, sinkhorn_normalization, fixed_geodesic

def test_get_effective_dim():
    print("Testing get_effective_dim...")
    # K1 eigenvalues: 10, 5, 0.1
    K1 = np.diag([10, 5, 0.1])
    # K2 eigenvalues: 8, 4, 0.05
    K2 = np.diag([8, 4, 0.05])
    
    # Threshold 1.0. K1 has 2 > 1. K2 has 2 > 1. Min is 2.
    dim = get_effective_dim(K1, K2, 1.0)
    assert dim == 2, f"Expected dim 2, got {dim}"
    print("Passed.")

def test_sinkhorn():
    print("Testing sinkhorn_normalization...")
    np.random.seed(42)
    A = np.random.rand(5, 5)
    A = (A + A.T) / 2 # Symmetric
    
    # Test single iteration
    K1, x1 = sinkhorn_normalization(A, max_iter=1, verbose=False)
    # Check if it ran (just basic check)
    assert K1.shape == A.shape
    
    # Test convergence
    K_full, x_full = sinkhorn_normalization(A, max_iter=1000, tol=1e-6, verbose=False)
    # Check bistochastic property: row sums and col sums should be constant (usually not 1 unless scaled, but Sinkhorn scales to 1 if r=1)
    # The implementation uses r=ones(n).
    # K = X A X. u = r./(A*v).
    # At convergence u=v=x.
    # u = 1 ./ (A*u) => u .* (A*u) = 1.
    # Row sums of K: sum(K, 2) = sum(X A X, 2) = X * sum(A X, 2) = x .* (A * x).
    # Since x .* (A*x) = 1, row sums should be 1.
    
    row_sums = np.sum(K_full, axis=1)
    col_sums = np.sum(K_full, axis=0)
    
    assert np.allclose(row_sums, 1.0), f"Row sums: {row_sums}"
    assert np.allclose(col_sums, 1.0), f"Col sums: {col_sums}"
    print("Passed.")

def test_fixed_geodesic_full_rank():
    print("Testing fixed_geodesic (full rank)...")
    A = np.diag([2.0, 1.0])
    B = np.diag([8.0, 4.0])
    
    # Geodesic between scalars a, b is a^(1-t) * b^t
    # For diagonal matrices, diagonal elements follow this.
    # t=0.5 -> sqrt(2*8)=4, sqrt(1*4)=2. Expected: diag([4, 2])
    
    S = fixed_geodesic(A, B, 0.5, dim=None)
    expected = np.diag([4.0, 2.0])
    assert np.allclose(S, expected), f"Expected {expected}, got {S}"
    
    # Endpoints
    S0 = fixed_geodesic(A, B, 0.0)
    assert np.allclose(S0, A)
    S1 = fixed_geodesic(A, B, 1.0)
    assert np.allclose(S1, B)
    print("Passed.")

def test_fixed_geodesic_low_rank():
    print("Testing fixed_geodesic (low rank)...")
    np.random.seed(42)
    # Create random SPD matrices
    X = np.random.randn(10, 10)
    A = X @ X.T + np.eye(10)*0.1
    Y = np.random.randn(10, 10)
    B = Y @ Y.T + np.eye(10)*0.1
    
    # Low rank approx with dim=5
    S = fixed_geodesic(A, B, 0.5, dim=5)
    
    # Check shape
    assert S.shape == (10, 10)
    
    # Check SPD (symmetry and positive eigenvalues)
    assert np.allclose(S, S.T)
    evals = scipy.linalg.eigvalsh(S)
    # It might have small negative values due to low rank approx truncation?
    # The Bonnabel method projects onto low rank manifold.
    # The result S = U R2 U' is PSD.
    assert np.all(evals > -1e-10)
    print("Passed.")

if __name__ == "__main__":
    test_get_effective_dim()
    test_sinkhorn()
    test_fixed_geodesic_full_rank()
    test_fixed_geodesic_low_rank()
