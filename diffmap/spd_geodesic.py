import numpy as np
import scipy.linalg
from sklearn.utils.extmath import randomized_svd

def get_effective_dim(K1, K2, threshold):
    """
    Determines the effective dimension of the kernel matrices based on a threshold.
    
    Args:
        K1: First kernel matrix (SPD).
        K2: Second kernel matrix (SPD).
        threshold: Threshold for eigenvalues.
        
    Returns:
        int: The effective dimension.
    """
    # Compute eigenvalues. Since K1, K2 are SPD, we can use eigvalsh.
    # We want sorted descending.
    d1 = scipy.linalg.eigvalsh(K1)
    d1 = np.sort(d1)[::-1]
    
    d2 = scipy.linalg.eigvalsh(K2)
    d2 = np.sort(d2)[::-1]
    
    # Find first index where eigenvalue < threshold
    idx1 = np.where(d1 < threshold)[0]
    dim1 = idx1[0] if len(idx1) > 0 else len(d1)
    
    idx2 = np.where(d2 < threshold)[0]
    dim2 = idx2[0] if len(idx2) > 0 else len(d2)
    
    # In MATLAB: Dim=min(find(d1<th,1,'first'),find(d2<th,1,'first'));
    # Note: MATLAB is 1-indexed, but here we return the count (which matches 0-indexed index if we take the first one)
    # Example: [10, 5, 0.1], th=1. idx is 2. dim is 2. (2 eigenvalues >= 1)
    
    dim = min(dim1, dim2)
    if dim == 0:
        return -1 # Or handle as error/full rank depending on usage
    return dim

def alpha_normalize(W, alpha):
    """
    Performs the alpha-normalization step (Diffusion Maps).
    
    Args:
        W: Affinity matrix.
        alpha: Alpha parameter (0 to 1).
        
    Returns:
        A: The alpha-normalized matrix.
    """
    # D = diag(sum(W)^-alpha)
    d = np.sum(W, axis=1)
    # Avoid division by zero if any row sum is 0 (unlikely for connected graph)
    d_inv_alpha = np.power(d, -alpha)
    D = np.diag(d_inv_alpha)
    
    # A = D * W * D
    A = D @ W @ D
    return A

def sinkhorn_normalization(A, max_iter=100, tol=1e-6, bound_c=1e-6, verbose=False):
    """
    Performs Sinkhorn-Knoop normalization to make matrix bistochastic.
    
    Args:
        A: Input matrix (non-negative).
        max_iter: Maximum iterations. Set to 1 for single iteration normalization.
        tol: Tolerance for convergence.
        bound_c: Lower bound for scaling factors to avoid numerical issues.
        verbose: Whether to print progress.
        
    Returns:
        K: Bistochastic (or column-stochastic approx) matrix.
        x: Scaling vector (sqrt(u*v)).
    """
    n = A.shape[0]
    r = np.ones(n) # target row sum
    
    # Initialize u = ones(n, 1)
    u = np.ones(n)
    v = r / (A @ u)
    x = np.sqrt(u * v)
    
    if np.min(x) <= bound_c:
        if verbose:
            print(f"Warning: Initial x has values <= {bound_c}")
    
    disps = []
    
    for ite in range(max_iter):
        # Check convergence
        # rhsdisc = u * (A * v) - r  <-- elementwise multiply u with (A@v)
        rhsdisc = u * (A @ v) - r
        disp = np.max(np.abs(rhsdisc))
        disps.append(disp)
        
        if verbose:
            print(f"ite {ite}: disps = {disp:.4e}")
            
        if disp < tol:
            break
        if ite > 0 and abs(disps[-1] - disps[-2]) < tol:
            break
            
        # Update
        u = r / (A @ v)
        v = r / (A @ u)
        x = np.sqrt(u * v)
        
        # Bound check
        if np.any(x < bound_c):
            if verbose:
                print(f"boundC not satisfied in ite {ite}.")
            x[x < bound_c] = bound_c
            
        u = x
        v = x
        
    # Final K calculation
    # In MATLAB: ColumnStochasticK=diag((1./sum(A)).^(-1/2))*K*diag((1./sum(A)).^(1/2));
    # But here we want the bistochastic result K = D * A * D
    # The MATLAB SingleIterationNorm returns:
    # K=diag((1./sum(A)).^(1/2))*A*diag((1./sum(A)).^(1/2));
    # which corresponds to one iteration of Sinkhorn starting with u=1/sqrt(sum(A))?
    # Wait, let's look at SK_sym_v4. It returns x.
    # The bistochastic matrix is D * A * D where D = diag(x).
    # Let's verify: K = diag(u) * A * diag(v). Since symmetric, u=v=x.
    
    X = np.diag(x)
    K = X @ A @ X
    
    return K, x

def fixed_geodesic(A, B, t, dim=None):
    """
    Computes the geodesic interpolation between two SPD matrices A and B.
    
    Args:
        A, B: SPD matrices.
        t: Interpolation parameter (0 to 1).
        dim: Rank for low-rank approximation. If None or <=0, uses full rank.
        
    Returns:
        S: Interpolated matrix at t.
    """
    if dim is not None and dim > 0:
        # Low rank approximation
        # [U1, ~] = eigs(A, Dim) -> Randomized SVD
        U1, _, _ = randomized_svd(A, n_components=dim, random_state=42)
        U2, _, _ = randomized_svd(B, n_components=dim, random_state=42)
        
        VA = U1 # Shape (N, dim)
        VB = U2 # Shape (N, dim)
        
        # [OA, SAB, OB] = svd(VA' * VB)
        # Note: numpy/scipy svd returns U, S, Vh. Vh is V.T.
        # MATLAB svd(X) returns U, S, V.
        # So OA -> U_inner, SAB -> S_inner, OB -> V_inner.T (if using python svd)
        # Let's match MATLAB: [OA, SAB, OB] = svd(VA' * VB)
        # Python: U, s, Vh = svd(VA.T @ VB)
        # OA = U
        # SAB = s
        # OB = Vh.T
        
        U_inner, S_inner, Vh_inner = scipy.linalg.svd(VA.T @ VB)
        OA = U_inner
        SAB = S_inner
        OB = Vh_inner.T
        
        # Clip singular values for stability
        SAB[SAB > 1] = 1.0
        
        UA = VA @ OA
        UB = VB @ OB
        
        theta = np.arccos(SAB)
        
        # X = (eye - UA*UA') * UB * pinv(diag(sin(theta)))
        # pinv(diag(sin(theta))) is just 1/sin(theta) on diagonal, handle 0
        sin_theta = np.sin(theta)
        inv_sin_theta = np.zeros_like(sin_theta)
        mask = np.abs(sin_theta) > 1e-10
        inv_sin_theta[mask] = 1.0 / sin_theta[mask]
        
        # Projection complement
        # UA is (N, dim). UA @ UA.T is (N, N).
        # This might be large if N is large.
        # (I - UA UA') UB = UB - UA (UA' UB)
        # UA' UB = (VA OA)' (VB OB) = OA' VA' VB OB
        # We know VA' VB = OA diag(SAB) OB'.
        # So OA' (OA diag(SAB) OB') OB = diag(SAB) = cos(theta)
        # So UA' UB = diag(cos(theta))
        # Thus (I - UA UA') UB = UB - UA diag(cos(theta))
        
        # Let's compute explicitly as in MATLAB for safety, or optimize if needed.
        # X=(eye(size(A))-UA*transpose(UA))*UB*pinv(diag(sin(theta)));
        # Optimized: X = (UB - UA @ np.diag(np.cos(theta))) @ np.diag(inv_sin_theta)
        
        X = (UB - UA @ np.diag(np.cos(theta))) @ np.diag(inv_sin_theta)
        
        # U = UA * diag(cos(theta*t)) + X * diag(sin(theta*t))
        U = UA @ np.diag(np.cos(theta * t)) + X @ np.diag(np.sin(theta * t))
        
        # RA2 = UA' * A * UA
        RA2 = UA.T @ A @ UA
        # RB2 = UB' * B * UB
        RB2 = UB.T @ B @ UB
        
        # [UR1, SR1, VR1] = svd(RA2)
        # RA = UR1 * sqrt(SR1) * VR1'
        U_r1, s_r1, Vh_r1 = scipy.linalg.svd(RA2)
        RA = U_r1 @ np.diag(np.sqrt(s_r1)) @ Vh_r1
        
        U_r2, s_r2, Vh_r2 = scipy.linalg.svd(RB2)
        RB = U_r2 @ np.diag(np.sqrt(s_r2)) @ Vh_r2
        
        # R2 = RA * expm(t * logm(RA^-1 * RB^2 * RA^-1)) * RA
        # Note: RB^2 is RB2 (if RB was sqrt of RB2)
        # Actually RB computed above is sqrt(RB2). So RB^2 approx RB2.
        # Let's follow formula: RA^(-1) * RB^2 * RA^(-1)
        
        RA_inv = scipy.linalg.inv(RA)
        inner_term = RA_inv @ RB2 @ RA_inv
        
        # logm can return complex, but for SPD it should be real.
        log_term = scipy.linalg.logm(inner_term)
        exp_term = scipy.linalg.expm(t * log_term)
        
        R2 = RA @ exp_term @ RA
        
        # S = U * R2 * U'
        S = U @ R2 @ U.T
        
        return S
        
    else:
        # Full rank computation
        if t == 0:
            return A
        if t == 1:
            return B
            
        # S = A^0.5 * (A^-0.5 * B * A^-0.5)^t * A^0.5
        # Stable implementation:
        # [U, S_vals, V] = svd(A^-0.5 * B * A^-0.5)
        # S = A^0.5 * (U * S_vals^t * V') * A^0.5
        
        # Compute A^0.5 and A^-0.5
        # A is SPD.
        vals_A, vecs_A = scipy.linalg.eigh(A)
        # vals_A might have small negatives due to numerics, clip them
        vals_A[vals_A < 0] = 0
        
        sqrt_vals_A = np.sqrt(vals_A)
        inv_sqrt_vals_A = np.zeros_like(vals_A)
        mask = vals_A > 1e-10
        inv_sqrt_vals_A[mask] = 1.0 / sqrt_vals_A[mask]
        
        A_half = vecs_A @ np.diag(sqrt_vals_A) @ vecs_A.T
        A_inv_half = vecs_A @ np.diag(inv_sqrt_vals_A) @ vecs_A.T
        
        # Middle term
        M = A_inv_half @ B @ A_inv_half
        
        # SVD of M
        # Since M is symmetric (congruence of SPD is SPD), svd matches eig.
        # But MATLAB code used svd(abs(...)).
        # Let's use svd.
        U_m, s_m, Vh_m = scipy.linalg.svd(M)
        
        # Middle power
        # U * s^t * V'
        M_t = U_m @ np.diag(np.power(s_m, t)) @ Vh_m
        
        S = A_half @ M_t @ A_half
        
        return S
