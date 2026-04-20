"""
Analyse why some off-diagonals of R are negative.

Three hypotheses:
 1. Architectural: W_out is unconstrained, so W_out @ ReLU(...) can be negative.
 2. Geometric (the real reason): a Parseval frame with F > n vectors in R^n
    that is centered at the origin must have off-diagonal inner products
    summing to a negative value. Mixed signs are forced.
 3. Loss-optimality: mixed-sign off-diagonals average near zero, so each
    feature's cross-talk has no systematic bias.
"""
import torch
import numpy as np


def load_R(name):
    sd = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = sd["W_in"].numpy()
    W_out = sd["W_out"].numpy()
    n, F = W_in.shape
    R = np.zeros((F, F))
    for j in range(F):
        R[:, j] = W_out @ np.maximum(W_in[:, j], 0.0)
    return W_in, W_out, R, n, F


for name in ["small_20f_5n_L4", "small_20f_2n_L4", "oct_7f_3n_L4", "oct_15f_4n_L4"]:
    W_in, W_out, R, n, F = load_R(name)
    off_mask = ~np.eye(F, dtype=bool)
    off = R[off_mask]
    print(f"\n=== {name} (F={F}, n={n}) ===")
    print(f"  #off-diag entries: {off.size}")
    print(f"  signed mean:   {off.mean():+.4f}")
    print(f"  abs mean:      {np.mean(np.abs(off)):+.4f}")
    print(f"  fraction > 0:  {(off > 0).mean():.3f}")
    print(f"  fraction < 0:  {(off < 0).mean():.3f}")
    print(f"  sum of off:    {off.sum():+.3f}")
    print(f"  sum of diag:   {np.diag(R).sum():+.3f}  (= n*alpha = {n * np.diag(R).mean() * F / n:.3f})")
    print(f"  1^T R 1:       {R.sum():+.3f}")

    # Centroid of the effective frame rows (rows of V where R = V S V^T)
    # If R = α P and P = V V^T with V orthonormal cols of an F x n matrix,
    # then V[j, :] is the frame vector for feature j in R^n.
    # We can extract: R = U diag(sv) V^T. For P = V V^T (projection), we'd
    # want U ≈ V. Take the symmetric part of R/α and compute its top-n eigvecs.
    alpha = np.diag(R).mean() * F / n
    R_sym = (R + R.T) / 2 / alpha
    w, Veig = np.linalg.eigh(R_sym)
    # top n eigvalues close to 1 (since sym part of α P ≈ P with eigs 1 on n of them)
    idx = np.argsort(w)[-n:]
    V = Veig[:, idx]          # F x n, rows are frame vectors (up to sign)
    row_norms_sq = np.sum(V**2, axis=1)
    centroid = V.sum(axis=0)  # n-vector: sum of frame vectors
    print(f"  frame row norms² (should be n/F = {n/F:.3f}): "
          f"mean={row_norms_sq.mean():.3f}, std={row_norms_sq.std():.3f}")
    print(f"  frame centroid (should be ≈ 0 if centered): "
          f"{np.linalg.norm(centroid):.3f}")
    print(f"  predicted sum-of-off = ||centroid||² - n = "
          f"{np.linalg.norm(centroid)**2 - n:+.3f}")
