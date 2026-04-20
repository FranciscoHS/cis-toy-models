"""Critique verification script."""
import torch
import numpy as np

def load(p):
    d = torch.load(p, map_location="cpu")
    return d

def R_eff(W_in, W_out, E):
    # W_in: n x D, W_out: D x n, E: D x F
    D, F = E.shape
    n = W_in.shape[0]
    R = np.zeros((F, F))
    for j in range(F):
        emb_j = E[:, j]
        z = W_in @ emb_j
        h = np.maximum(z, 0.0)
        r_out = W_out @ h
        R[:, j] = E.T @ r_out
    return R

def codewords(W_in, E):
    W_in_eff = W_in @ E  # n x F
    return [tuple((W_in_eff[:, j] > 0).astype(int).tolist()) for j in range(W_in_eff.shape[1])]

def analyze(name):
    d = load(f"weights/{name}.pt")
    W_in = d["W_in"].numpy()
    W_out = d["W_out"].numpy()
    E = d["E"].numpy() if "E" in d else None
    meta = d.get("meta", {})

    if E is None:
        # plain model: no E; just compute R directly
        F = W_in.shape[1]
        n = W_in.shape[0]
        R = np.zeros((F, F))
        for j in range(F):
            z = W_in[:, j]
            h = np.maximum(z, 0.0)
            R[:, j] = W_out @ h
        W_in_eff = W_in
    else:
        F = E.shape[1]
        n = W_in.shape[0]
        R = R_eff(W_in, W_out, E)
        W_in_eff = W_in @ E

    diag = np.diag(R)
    alpha = diag.mean() * F / n
    s_opt = np.trace(R) / (R**2).sum()
    frob = np.linalg.norm(s_opt * R - np.eye(F), "fro")**2

    cws = [tuple((W_in_eff[:, j] > 0).astype(int).tolist()) for j in range(F)]
    from collections import Counter
    ccount = Counter(cws)

    # Also examine sign patterns of pre-ReLU at features
    # (should match sign of W_in_eff[:, j] trivially)
    # and look at the effective W_in directions and compare across configs
    print(f"=== {name} === F={F}, n={n}  meta={meta}")
    print(f"  diag mean={diag.mean():.4f} std={diag.std():.4f} alpha={alpha:.4f}")
    print(f"  ||sR - I||^2 = {frob:.4f}  (F-n = {F-n})")
    print(f"  codeword counts: {dict(ccount)}")
    # For n=2, show the raw W_in_eff columns as 2-vectors and their angles
    if n == 2:
        angles = np.arctan2(W_in_eff[1, :], W_in_eff[0, :])
        # Also normalized directions:
        norms = np.linalg.norm(W_in_eff, axis=0)
        dirs = W_in_eff / norms
        print(f"  angles (deg) sorted: {np.sort(np.degrees(angles))}")
        print(f"  norms: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}")
    return dict(F=F, n=n, alpha=alpha, frob=frob, codewords=dict(ccount), W_in_eff=W_in_eff)

print("\n=== Plain model baseline ===")
plain = analyze("small_20f_2n_L4")

print("\n=== Embedded models ===")
configs = [
    "embed_20f_2n_D20_orth_L4",
    "embed_20f_2n_D40_unit_L4",
    "embed_20f_2n_D80_unit_L4",
]
results = {}
for c in configs:
    results[c] = analyze(c)

# Now the gauge question: given plain W_in_eff, can we align embedded W_in_eff to it
# via an orthogonal rotation?
print("\n=== Gauge comparison across configs (n=2) ===")
P = plain["W_in_eff"]    # 2 x F
for c, r in results.items():
    Q = r["W_in_eff"]    # 2 x F
    # Find best rotation R minimizing ||R Q - P||_F. Procrustes.
    M = P @ Q.T  # 2x2
    U, s, Vt = np.linalg.svd(M)
    Rot = U @ Vt
    # Check det for proper rotation; allow reflection too
    Q_aligned = Rot @ Q
    resid = np.linalg.norm(Q_aligned - P, "fro") / np.linalg.norm(P, "fro")
    # Also compare column-norm distributions
    n_plain = np.linalg.norm(P, axis=0)
    n_emb = np.linalg.norm(Q, axis=0)
    print(f"  {c}:  ||R Q - P||/||P|| = {resid:.4f}")
    # And compare sorted angles (rotation-invariant up to global phase)
    ang_p = np.sort(np.arctan2(P[1,:], P[0,:]) % np.pi)
    ang_q = np.sort(np.arctan2(Q[1,:], Q[0,:]) % np.pi)
    print(f"    sorted angles (mod pi) diff L-inf: {np.abs(ang_p - ang_q).max():.4f} rad")
