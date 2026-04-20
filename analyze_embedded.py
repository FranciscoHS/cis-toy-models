"""
Compare effective weights (W_in @ E, E^T @ W_out) of embedded-variant models
to the plain-model solution. Check:
 1. Does the effective single-feature response R still satisfy the rank-n
    projection structure (diag uniform, Frob bound saturated)?
 2. Do the effective W_in columns cluster into codewords (combinatorial
    ReLU coding)?

If yes: the embedded model has learned the same mechanism expressed in rotated
coordinates.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embedded(name):
    d = torch.load(f"weights/{name}.pt", map_location="cpu")
    return d["W_in"].numpy(), d["W_out"].numpy(), d["E"].numpy(), d["meta"]


def effective_single_feature_response(W_in, W_out, E):
    """R[:, j] = E^T @ W_out @ ReLU(W_in @ E @ e_j)
             = E^T @ W_out @ ReLU(W_in @ E[:, j])
    Shape: F x F.
    """
    D, F = E.shape
    n = W_in.shape[0]
    assert W_in.shape == (n, D)
    assert W_out.shape == (D, n)
    R = np.zeros((F, F))
    for j in range(F):
        emb_j = E[:, j]                             # D-dim
        z = W_in @ emb_j                            # n-dim
        h = np.maximum(z, 0.0)
        r_out = W_out @ h                           # D-dim
        R[:, j] = E.T @ r_out                       # F-dim
    return R


def analyze(name):
    W_in, W_out, E, meta = load_embedded(name)
    F, n, D = meta["F"], meta["n"], meta["D"]
    loss_exp = meta["loss_exp"]
    E_kind = meta["E_kind"]

    R = effective_single_feature_response(W_in, W_out, E)
    diag = np.diag(R)
    mean_diag = diag.mean()
    alpha = mean_diag * F / n
    s_opt = np.trace(R) / (R**2).sum()
    frob_sI = np.linalg.norm(s_opt * R - np.eye(F), "fro") ** 2

    U, sv, Vt = np.linalg.svd(R, full_matrices=False)

    # Effective W_in: rows define n directions in R^F. They play the same role
    # as the plain model's W_in rows. But we should look at "effective codewords"
    # W_in @ E (n x F): each column is the effective projection to the n neurons
    # when feature j is active singly.
    W_in_eff = W_in @ E         # n x F
    W_out_eff = E.T @ W_out     # F x n
    # "codeword" of feature j = sign pattern of W_in_eff[:, j]
    codewords = [tuple((W_in_eff[:, j] > 0).astype(int).tolist()) for j in range(F)]
    uniq_cw = set(codewords)

    print(f"\n=== {name} ===")
    print(f"  F={F}, n={n}, D={D}, E={E_kind}, L{loss_exp}")
    print(f"  diag(R) mean={mean_diag:.4f}, std={diag.std():.4f}, α={alpha:.3f}")
    print(f"  ||s_opt R - I||^2 = {frob_sI:.3f}  (F-n = {F-n})")
    print(f"  top-{n} singular values: "
          f"{np.array2string(sv[:n], precision=3)}  (rest max: {sv[n:].max() if len(sv)>n else 0:.4e})")
    print(f"  effective codewords: {len(uniq_cw)}/{2**n}")
    for p in sorted(uniq_cw, key=lambda x: -codewords.count(x)):
        cnt = codewords.count(p)
        if cnt > 0:
            print(f"    {p}: {cnt}")

    # Off-diag mean|.|
    off = R[~np.eye(F, dtype=bool)]
    print(f"  off-diag mean|.| = {np.mean(np.abs(off)):.4f}")
    welch = alpha * np.sqrt(n * (F-n) / (F**2 * (F-1)))
    print(f"  α × Welch        = {welch:.4f}")

    return dict(name=name, F=F, n=n, D=D, E_kind=E_kind, loss_exp=loss_exp,
                alpha=alpha, mean_diag=mean_diag, diag_std=diag.std(),
                frob_sI=frob_sI, sv=sv.tolist(),
                n_codewords=len(uniq_cw), off_mean_abs=float(np.mean(np.abs(off))),
                welch=welch)


if __name__ == "__main__":
    configs = [
        "embed_20f_2n_D20_orth_L2",
        "embed_20f_2n_D20_orth_L4",
        "embed_20f_2n_D40_unit_L2",
        "embed_20f_2n_D40_unit_L4",
        "embed_20f_2n_D80_unit_L4",
        "embed_20f_5n_D20_orth_L4",
        "embed_20f_5n_D40_unit_L4",
        "embed_20f_5n_D80_unit_L4",
        "embed_20f_5n_D200_unit_L4",
    ]
    results = []
    for c in configs:
        try:
            r = analyze(c)
            results.append(r)
        except FileNotFoundError:
            print(f"[missing] {c}")

    # Compare to plain models
    print("\n" + "=" * 70)
    print("Comparison: embedded α vs plain-model α")
    print("=" * 70)
    plain_alpha = {
        (20, 2): 3.243,
        (20, 5): 2.045,
    }
    for r in results:
        if r["loss_exp"] == 4:
            key = (r["F"], r["n"])
            if key in plain_alpha:
                pa = plain_alpha[key]
                print(f"  {r['name']:<35}  α={r['alpha']:.3f}  "
                      f"(plain: {pa:.3f}, ratio: {r['alpha']/pa:.3f})")
