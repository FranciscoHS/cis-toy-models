"""
Visualize R matrices for a few key configurations: 20f/5n (user's case),
31f/5n (one codeword per feature), and 15f/4n (F = 2^n - 1).
"""
import torch
import numpy as np
from pathlib import Path
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


def plot_R(name, ax, perm=None):
    W_in, W_out, R, n, F = load_R(name)
    if perm is None:
        codewords = [tuple((W_in[:, j] > 0).astype(int).tolist()) for j in range(F)]
        perm = np.array(sorted(range(F), key=lambda j: codewords[j]), dtype=int)
    R_p = R[np.ix_(perm, perm)]
    vmax = np.abs(R).max()
    im = ax.imshow(R_p, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{name}\n(F={F}, n={n}, 2^n-1={2**n - 1})")
    ax.set_xlabel("j (input feature)")
    ax.set_ylabel("i (output dim)")
    return perm


names = [
    "small_20f_2n_L4",
    "small_20f_5n_L4",
    "oct_15f_4n_L4",
    "oct_31f_5n_L4",
]
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, name in zip(axes, names):
    plot_R(name, ax)
fig.tight_layout()
fig.savefig("figures/R_heatmaps.png", dpi=150)
print("Saved figures/R_heatmaps.png")

# Detailed plot for 20f/5n: decompose into diagonal + off-diagonal distribution
W_in, W_out, R, n, F = load_R("small_20f_5n_L4")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) sorted R heatmap
codewords = [tuple((W_in[:, j] > 0).astype(int).tolist()) for j in range(F)]
perm = np.array(sorted(range(F), key=lambda j: codewords[j]), dtype=int)
R_p = R[np.ix_(perm, perm)]
vmax = np.abs(R).max()
im = axes[0].imshow(R_p, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=axes[0], shrink=0.8)
axes[0].set_title("R (features sorted by codeword)\n20f/5n L4 trained")
axes[0].set_xlabel("j")
axes[0].set_ylabel("i")

# (2) histogram of diag vs off-diag
diag = np.diag(R)
off_same, off_diff = [], []
for i in range(F):
    for j in range(F):
        if i == j:
            continue
        (off_same if codewords[i] == codewords[j] else off_diff).append(R[i, j])
axes[1].hist(diag, bins=15, alpha=0.8, label=f'diagonal (n={len(diag)})', color='tab:red')
axes[1].hist(off_same, bins=15, alpha=0.5, label=f'same codeword (n={len(off_same)})', color='tab:orange')
axes[1].hist(off_diff, bins=30, alpha=0.5, label=f'different codeword (n={len(off_diff)})', color='tab:blue')
# Predictions
alpha = diag.mean() * F / n
welch_off = alpha * np.sqrt(n * (F - n) / (F**2 * (F - 1)))
axes[1].axvline(alpha * n / F, color='red', linestyle=':', alpha=0.6,
                label=f'α·n/F = {alpha * n / F:.3f}')
axes[1].axvline(welch_off, color='blue', linestyle=':', alpha=0.6,
                label=f'+Welch = {welch_off:.3f}')
axes[1].axvline(-welch_off, color='blue', linestyle=':', alpha=0.6,
                label=f'-Welch = -{welch_off:.3f}')
axes[1].legend(fontsize=9)
axes[1].set_xlabel("R entry value")
axes[1].set_ylabel("count")
axes[1].set_title("Distribution of R entries (20f/5n)")

# (3) Singular values of R
U, s, Vt = np.linalg.svd(R, full_matrices=False)
axes[2].plot(np.arange(1, len(s)+1), s, 'o-')
axes[2].axvline(n + 0.5, color='red', linestyle='--', alpha=0.5, label=f'rank bound n={n}')
axes[2].set_xlabel("index")
axes[2].set_ylabel("singular value")
axes[2].set_title("Singular values of R (20f/5n)")
axes[2].legend()
axes[2].set_yscale('log')

fig.tight_layout()
fig.savefig("figures/R_20f_5n_detail.png", dpi=150)
print("Saved figures/R_20f_5n_detail.png")

# Also check: are diff-codeword off-diagonals of (nearly) the same magnitude?
# For an ETF, they all have magnitude = sqrt(n(F-n)/(F^2(F-1))).
# Compare their magnitude distribution.
off_diff_arr = np.array(off_diff)
print(f"\n20f/5n: diff-codeword |off-diag| stats:")
print(f"  mean = {np.mean(np.abs(off_diff_arr)):.4f}")
print(f"  std = {np.std(np.abs(off_diff_arr)):.4f}")
print(f"  std/mean = {np.std(np.abs(off_diff_arr))/np.mean(np.abs(off_diff_arr)):.4f}")
print(f"  Welch prediction: {welch_off:.4f}")
