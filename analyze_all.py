"""
Analyze R structure for all L4 models. Produces a summary table and alpha-vs-size
scatter.
"""
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_R(name):
    sd = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = sd["W_in"].detach().numpy()
    W_out = sd["W_out"].detach().numpy()
    n, F = W_in.shape
    R = np.zeros((F, F))
    for j in range(F):
        R[:, j] = W_out @ np.maximum(W_in[:, j], 0.0)
    return W_in, W_out, R, n, F


def summarize(name):
    try:
        W_in, W_out, R, n, F = load_R(name)
    except FileNotFoundError:
        return None
    diag = np.diag(R)
    mean_diag = diag.mean()
    alpha = mean_diag * F / n  # since P_jj = n/F, diag = α * n/F
    # Frob bound
    s = np.trace(R) / (R**2).sum()
    frob_sI = np.linalg.norm(s * R - np.eye(F), "fro") ** 2
    frob_Fmn = F - n
    # singular values
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    top_n_sv = sv[:n]
    sv_spread = top_n_sv.std() / top_n_sv.mean()
    # off-diag stats
    off = R[~np.eye(F, dtype=bool)]
    # split by codeword sharing
    pos = [tuple((W_in[:, j] > 0).astype(int).tolist()) for j in range(F)]
    same = []
    diff = []
    for i in range(F):
        for j in range(F):
            if i == j:
                continue
            (same if pos[i] == pos[j] else diff).append(R[i, j])
    same = np.array(same) if same else np.zeros(0)
    diff = np.array(diff) if diff else np.zeros(0)
    # predicted |off-diag| from Welch bound: α * sqrt(n(F-n)/(F²(F-1)))
    welch_mag = alpha * np.sqrt(n * (F - n) / (F**2 * (F - 1))) if F > 1 else 0.0
    # unique codewords
    n_codewords = len(set(pos))

    return dict(
        name=name, n=n, F=F, F_over_n=F / n, two_n_minus_1=2**n - 1,
        mean_diag=float(mean_diag), diag_uniform_std=float(diag.std()),
        alpha=float(alpha),
        frob_sI=float(frob_sI), frob_bound=float(frob_Fmn),
        frob_ratio=float(frob_sI / max(frob_Fmn, 1e-10)),
        top_n_sv_mean=float(top_n_sv.mean()),
        sv_spread=float(sv_spread),
        off_mean=float(off.mean()), off_mean_abs=float(np.mean(np.abs(off))),
        same_codeword_mean=float(same.mean()) if len(same) else None,
        diff_codeword_mean=float(diff.mean()) if len(diff) else None,
        diff_codeword_mean_abs=float(np.mean(np.abs(diff))) if len(diff) else None,
        welch_predicted_abs=float(welch_mag),
        n_codewords=int(n_codewords),
    )


if __name__ == "__main__":
    configs = [
        # trivial
        "trivial_2f_2n_L4", "trivial_3f_3n_L4", "trivial_5f_5n_L4",
        "fix5n_5f_L4",
        # F = 2^n - 1 (one codeword per feature)
        "oct_3f_2n_L4", "oct_7f_3n_L4", "oct_15f_4n_L4", "oct_31f_5n_L4",
        # 20f varying n
        "small_20f_2n_L4", "small_20f_3n_L4", "small_20f_4n_L4",
        "small_20f_5n_L4", "small_20f_6n_L4", "small_20f_7n_L4", "small_20f_8n_L4",
        # scaling series 10:1
        "small_10f_1n_L4", "small_50f_5n_L4",
        "small_100f_10n_L4", "small_200f_20n_L4", "small_500f_50n_L4",
        "small_1000f_100n_L4",
        # fixed n=5, varying F
        "fix5n_10f_L4", "fix5n_15f_L4", "fix5n_20f_L4",
        "fix5n_31f_L4", "fix5n_50f_L4", "fix5n_100f_L4",
    ]

    rows = []
    for c in configs:
        r = summarize(c)
        if r:
            rows.append(r)

    # Print table
    hdr = f"{'name':<24} {'F':>5} {'n':>3} {'2^n-1':>6} {'#cw':>4} {'mean_diag':>10} {'α':>7} {'frob_ratio':>11} {'sv_spread':>10} {'|off_diff|':>11} {'welch_pred':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<24} {r['F']:>5d} {r['n']:>3d} {r['two_n_minus_1']:>6d} "
              f"{r['n_codewords']:>4d} {r['mean_diag']:>10.4f} {r['alpha']:>7.3f} "
              f"{r['frob_ratio']:>11.4f} {r['sv_spread']:>10.4f} "
              f"{(r['diff_codeword_mean_abs'] or 0):>11.4f} "
              f"{r['welch_predicted_abs']:>11.4f}")

    # Save
    Path("data").mkdir(exist_ok=True)
    with open("data/R_structure_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Plot alpha vs F, n
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: alpha vs F at fixed n=5
    fixn5 = [r for r in rows if r['n'] == 5]
    fixn5.sort(key=lambda r: r['F'])
    axes[0].plot([r['F'] for r in fixn5], [r['alpha'] for r in fixn5], 'o-', label='α observed (n=5)')
    # Theoretical β from single-feature L4 ETF: β = 1/(1 + (γ²/(F-1))^{1/3}), α = βF/n
    Fs = np.array([r['F'] for r in fixn5])
    n_ = 5
    gamma = (Fs - n_) / n_
    beta = 1.0 / (1.0 + (gamma**2 / (Fs - 1)) ** (1/3))
    alpha_pred = beta * Fs / n_
    # at F=5 (trivial), γ=0, β=1, α=1
    axes[0].plot(Fs, alpha_pred, 's--', label='α predicted (single-feat L4+ETF)', alpha=0.5)
    axes[0].set_xlabel('F')
    axes[0].set_ylabel('α = mean_diag * F/n')
    axes[0].set_title('α scaling at fixed n=5')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: alpha vs n at fixed F=20
    fixF20 = [r for r in rows if r['F'] == 20 and r['name'].startswith('small_')]
    fixF20.sort(key=lambda r: r['n'])
    ns = np.array([r['n'] for r in fixF20])
    Fs = 20 * np.ones_like(ns)
    gamma = (Fs - ns) / ns
    beta = 1.0 / (1.0 + (gamma**2 / (Fs - 1)) ** (1/3))
    alpha_pred = beta * Fs / ns
    axes[1].plot([r['n'] for r in fixF20], [r['alpha'] for r in fixF20], 'o-', label='α observed (F=20)')
    axes[1].plot(ns, alpha_pred, 's--', label='α predicted (single-feat L4+ETF)', alpha=0.5)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('α')
    axes[1].set_title('α scaling at fixed F=20')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('figures/alpha_scaling.png', dpi=150)
    print('\nSaved figures/alpha_scaling.png')
