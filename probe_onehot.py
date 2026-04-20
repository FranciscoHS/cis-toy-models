"""
Probe single-feature (one-hot, value=+1) response of trained CiS models.

For input x = v * e_j with v = +1:
  y = W_out @ ReLU(W_in @ e_j) = W_out @ ReLU(W_in[:, j])

We look at y as a row vector in R^F:
  y[j]   = active-feature readout for feature j
  y[i]   = cross-talk (interference) on other feature i != j

Question: what does y look like? Is active >> inactive? Are inactives
all the same magnitude? Are there groupings?
"""

import torch
import numpy as np
from pathlib import Path
import json

WEIGHTS_DIR = Path("weights")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def onehot_response(W_in, W_out):
    """Return R[F,F] where R[:, j] = W_out @ ReLU(W_in[:, j]).
    Column j is the model's output y when x = e_j.
    """
    F = W_in.shape[1]
    n = W_in.shape[0]
    R = np.zeros((F, F))
    for j in range(F):
        z = W_in[:, j]
        relu_z = np.maximum(z, 0.0)
        R[:, j] = W_out @ relu_z
    return R


def analyze(name):
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        print(f"  [missing] {path}")
        return None
    sd = torch.load(path, map_location="cpu")
    W_in = sd["W_in"].detach().numpy()   # (n, F)
    W_out = sd["W_out"].detach().numpy() # (F, n)
    n, F = W_in.shape
    assert W_out.shape == (F, n)

    R = onehot_response(W_in, W_out)
    diag = np.diag(R)
    off = R - np.diag(diag)

    mean_diag = diag.mean()
    std_diag = diag.std()
    off_mask = ~np.eye(F, dtype=bool)
    off_vals = R[off_mask]
    mean_off_abs = np.mean(np.abs(off_vals))
    std_off = off_vals.std()

    ratio = mean_diag / mean_off_abs if mean_off_abs > 1e-10 else np.inf

    print(f"\n=== {name}: n={n}, F={F}, 2^n-1={2**n - 1} ===")
    print(f"  diagonal: mean={mean_diag:.4f}, std={std_diag:.4f}, min={diag.min():.4f}, max={diag.max():.4f}")
    print(f"  off-diag: mean|.|={mean_off_abs:.4f}, std={std_off:.4f}")
    print(f"  ratio diag/off-abs = {ratio:.3f}")
    print(f"  Frob ||R - I||^2 = {np.linalg.norm(R - np.eye(F), 'fro')**2:.3f}  (lower bound F-n = {F-n})")
    s = np.trace(R) / (R**2).sum()
    print(f"  optimal scale s = tr(R)/||R||_F^2 = {s:.4f}")
    print(f"  Frob ||sR - I||^2 = {np.linalg.norm(s*R - np.eye(F), 'fro')**2:.3f}")
    # Singular values of R
    sv = np.linalg.svd(R, compute_uv=False)
    print(f"  singular values of R: {np.array2string(sv, precision=3, suppress_small=True)}")

    # Count codewords: sign pattern of W_in[:, j] (+ for >0, - for <=0)
    signs = np.sign(W_in)  # each column is a sign pattern
    sign_tuples = [tuple(signs[:, j].astype(int).tolist()) for j in range(F)]
    unique_codewords = sorted(set(sign_tuples))
    print(f"  unique codewords: {len(unique_codewords)} / {2**n}")
    # Codeword groups (based on positive-sign subset, which is what ReLU keeps)
    pos_patterns = [tuple((W_in[:, j] > 0).astype(int).tolist()) for j in range(F)]
    unique_pos = sorted(set(pos_patterns))
    print(f"  unique pos-patterns (ReLU-effective): {len(unique_pos)}")
    counts = {p: pos_patterns.count(p) for p in unique_pos}
    for p, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {p}: {c}")

    return dict(name=name, n=n, F=F, mean_diag=float(mean_diag), std_diag=float(std_diag),
                mean_off_abs=float(mean_off_abs), std_off=float(std_off),
                ratio=float(ratio), F_minus_n=F-n,
                frob_I=float(np.linalg.norm(R - np.eye(F), 'fro')**2),
                frob_sI=float(np.linalg.norm(s*R - np.eye(F), 'fro')**2),
                s=float(s),
                svs=sv.tolist(),
                pos_pattern_counts={str(p): c for p, c in counts.items()},
                R=R.tolist())


if __name__ == "__main__":
    configs = [
        # 20f with varying n (the main sweep)
        "small_20f_2n_L4",
        "small_20f_3n_L4",
        "small_20f_4n_L4",
        "small_20f_5n_L4",
        # scaling series at 10:1
        "small_10f_1n_L4",
        "small_50f_5n_L4",
        "small_100f_10n_L4",
        "small_200f_20n_L4",
        "small_500f_50n_L4",
        "small_1000f_100n_L4",
    ]

    results = {}
    for name in configs:
        r = analyze(name)
        if r:
            results[name] = r

    Path("data").mkdir(exist_ok=True)
    with open("data/onehot_probe.json", "w") as f:
        # strip R for JSON compactness but keep aggregate stats
        compact = {k: {kk: vv for kk, vv in v.items() if kk != "R"} for k, v in results.items()}
        json.dump(compact, f, indent=2)
    # separately save full R matrices as npz
    np.savez("data/onehot_probe_R.npz",
             **{k: np.array(v["R"]) for k, v in results.items()})
    print("\nSaved data/onehot_probe.json and data/onehot_probe_R.npz")
