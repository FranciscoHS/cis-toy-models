"""
Deeper structure analysis of the single-feature response matrix R.

Key hypothesis: R ≈ α * P where P is a rank-n orthogonal projection with
uniform diagonal n/F (i.e., Col(W_out) is a Parseval-frame subspace).

Checks:
 1. Is R ≈ α * P for a projection P? Test: is R^2 ≈ α R?
 2. What is α, relative to the "natural" prediction?
 3. Is diag(R) uniform? (Should be n/F times α, for uniform-diagonal projection.)
 4. Structure of off-diagonals: are they grouped by codeword?
 5. Can we predict α from the task / loss?

We look at R in the basis of its column space (Col(W_out)) and measure
deviation from the uniform-diagonal projection.
"""

import torch
import numpy as np
from pathlib import Path


def load_R(name):
    sd = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = sd["W_in"].detach().numpy()
    W_out = sd["W_out"].detach().numpy()
    n, F = W_in.shape
    R = np.zeros((F, F))
    for j in range(F):
        R[:, j] = W_out @ np.maximum(W_in[:, j], 0.0)
    return W_in, W_out, R, n, F


def projection_decomposition(R):
    """Find best α, P such that R ≈ α * P with P rank-r orthogonal projection.
    We take top-r SVD of R. If R is approximately rank-r, R = U S V^T with r
    nonzero singular values. Best scale: make S ≈ α * I_r.
    """
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    r_eff = np.sum(s > 1e-6)
    sv_top = s[:r_eff]
    alpha = sv_top.mean()
    # Asymmetric of R (R vs R^T)
    asym = np.linalg.norm(R - R.T, 'fro') / np.linalg.norm(R, 'fro')
    # Check if U ≈ V (would mean R symmetric)
    return dict(
        sv_top=sv_top.tolist(),
        alpha=float(alpha),
        sv_std_over_mean=float(sv_top.std() / alpha),
        asym=float(asym),
    )


def grouped_offdiag(W_in, R):
    """Group features by their positive-sign pattern (codeword) and look
    at the block structure of R w.r.t. these groups.
    """
    F = R.shape[0]
    pos_patterns = [tuple((W_in[:, j] > 0).astype(int).tolist()) for j in range(F)]
    groups = {}
    for j, p in enumerate(pos_patterns):
        groups.setdefault(p, []).append(j)

    same_vals = []
    diff_vals = []
    for j in range(F):
        for i in range(F):
            if i == j:
                continue
            if pos_patterns[i] == pos_patterns[j]:
                same_vals.append(R[i, j])
            else:
                diff_vals.append(R[i, j])
    same_vals = np.array(same_vals)
    diff_vals = np.array(diff_vals)
    return dict(
        n_groups=len(groups),
        n_same_codeword=int(len(same_vals)),
        n_diff_codeword=int(len(diff_vals)),
        same_mean=float(same_vals.mean()) if len(same_vals) else 0.0,
        same_meanabs=float(np.mean(np.abs(same_vals))) if len(same_vals) else 0.0,
        diff_mean=float(diff_vals.mean()) if len(diff_vals) else 0.0,
        diff_meanabs=float(np.mean(np.abs(diff_vals))) if len(diff_vals) else 0.0,
    )


def uniform_projection_comparison(R, n_target):
    """Compare R to the 'uniform Parseval frame projection' of rank n.

    A uniform rank-n projection P of I_F has P_jj = n/F for all j.
    We normalize R by its optimal scale α such that R/α has rank-n pattern;
    here α := mean_diag(R) / (n/F).
    """
    F = R.shape[0]
    uniform_diag = n_target / F
    mean_diag = np.diag(R).mean()
    alpha = mean_diag / uniform_diag

    P_hat = R / alpha
    # How close is P_hat to a rank-n orthogonal projection with uniform diag?
    # Key properties of such P: diag = n/F, P^2 = P, rank = n
    diag_deviation = np.diag(P_hat) - uniform_diag
    proj_idempotent_gap = np.linalg.norm(P_hat @ P_hat - P_hat, 'fro')
    # Rank check: sum of top-n singular values vs ||P||_F^2
    U, s, Vt = np.linalg.svd(P_hat)
    s_top = s[:n_target]
    return dict(
        alpha=float(alpha),
        uniform_diag=float(uniform_diag),
        diag_dev_max=float(np.abs(diag_deviation).max()),
        diag_dev_std=float(diag_deviation.std()),
        top_n_sv_sum=float(s_top.sum()),
        ideal_sv_sum=float(n_target),
        proj_idempotent_gap=float(proj_idempotent_gap),
        sym_gap=float(np.linalg.norm(P_hat - P_hat.T, 'fro')),
    )


def analyze(name):
    print(f"\n=== {name} ===")
    W_in, W_out, R, n, F = load_R(name)
    print(f"  n={n}, F={F}")

    proj = projection_decomposition(R)
    print(f"  top-{n} sv: mean={proj['alpha']:.3f}, std/mean={proj['sv_std_over_mean']:.4f}")
    print(f"  asymmetry ||R-R^T||/||R|| = {proj['asym']:.4f}")

    group = grouped_offdiag(W_in, R)
    print(f"  groups={group['n_groups']}, "
          f"same-codeword pair count={group['n_same_codeword']}")
    if group['n_same_codeword'] > 0:
        print(f"    same-codeword off-diag: mean={group['same_mean']:.4f}, "
              f"mean|.|={group['same_meanabs']:.4f}")
    print(f"    diff-codeword off-diag: mean={group['diff_mean']:.4f}, "
          f"mean|.|={group['diff_meanabs']:.4f}")

    proj2 = uniform_projection_comparison(R, n)
    print(f"  If R = α * P with P rank-{n} uniform-diag projection:")
    print(f"    α = {proj2['alpha']:.3f}")
    print(f"    diag deviation (max, std) = ({proj2['diag_dev_max']:.4f}, {proj2['diag_dev_std']:.4f})")
    print(f"    top-{n} SV sum = {proj2['top_n_sv_sum']:.3f} (ideal if projection: {proj2['ideal_sv_sum']:.3f})")
    print(f"    |P^2 - P|_F = {proj2['proj_idempotent_gap']:.4f}")
    print(f"    |P - P^T|_F = {proj2['sym_gap']:.4f}")

    return dict(name=name, n=n, F=F, **proj, group=group, uniform=proj2)


if __name__ == "__main__":
    configs = [
        "small_10f_1n_L4",
        "small_20f_2n_L4",
        "small_20f_3n_L4",
        "small_20f_4n_L4",
        "small_20f_5n_L4",
        "small_50f_5n_L4",
        "small_100f_10n_L4",
        "small_200f_20n_L4",
        "small_500f_50n_L4",
        "small_1000f_100n_L4",
    ]
    for c in configs:
        analyze(c)
