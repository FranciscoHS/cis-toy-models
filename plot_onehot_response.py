"""
Plot the network's output y = R[:, j] when the input is a one-hot x = e_j.
Shows directly what the active feature's response is vs. the cross-talk on
all other features, as a function of output index.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_plain_R(name):
    sd = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = sd["W_in"].numpy()
    W_out = sd["W_out"].numpy()
    n, F = W_in.shape
    R = np.zeros((F, F))
    for j in range(F):
        R[:, j] = W_out @ np.maximum(W_in[:, j], 0.0)
    return R, n, F


def load_embedded_R(name):
    d = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = d["W_in"].numpy()
    W_out = d["W_out"].numpy()
    E = d["E"].numpy()
    D, F = E.shape
    n = W_in.shape[0]
    R = np.zeros((F, F))
    for j in range(F):
        h = np.maximum(W_in @ E[:, j], 0.0)
        R[:, j] = E.T @ (W_out @ h)
    return R, n, F


def plot_response_bar(ax, R, j, F, n, title):
    response = R[:, j]
    colors = ["tab:red" if i == j else "tab:blue" for i in range(F)]
    ax.bar(np.arange(F), response, color=colors, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    # reference lines
    alpha = np.diag(R).mean() * F / n
    welch = alpha * np.sqrt(n * (F - n) / (F**2 * (F - 1))) if F > 1 else 0
    ax.axhline(alpha * n / F, color="tab:red", linestyle="--", alpha=0.6,
               label=rf"$\alpha\,n/F$ = {alpha * n / F:.3f}")
    if welch > 0:
        ax.axhline(welch, color="tab:blue", linestyle=":", alpha=0.6,
                   label=rf"$\pm\alpha\cdot$Welch = $\pm${welch:.3f}")
        ax.axhline(-welch, color="tab:blue", linestyle=":", alpha=0.6)
    ax.axhline(1.0, color="tab:green", linestyle="-.", alpha=0.5,
               label="target = 1")
    ax.set_xlabel("output feature index i")
    ax.set_ylabel(fr"$y_i$ for input $x = e_{{{j}}}$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(np.arange(0, F, max(1, F // 10)))


# Figure 1: one-hot response for four configurations at F=20
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 20f/5n L4 plain
R, n, F = load_plain_R("small_20f_5n_L4")
j_pick = 0  # arbitrary, all j look similar
plot_response_bar(axes[0, 0], R, j_pick, F, n,
                  rf"20f/5n $L^4$ (plain) — input $e_{{{j_pick}}}$")

# 20f/2n L4 plain
R, n, F = load_plain_R("small_20f_2n_L4")
plot_response_bar(axes[0, 1], R, j_pick, F, n,
                  rf"20f/2n $L^4$ (plain) — input $e_{{{j_pick}}}$")

# 20f/5n L4 embedded D=80
R, n, F = load_embedded_R("embed_20f_5n_D80_unit_L4")
plot_response_bar(axes[1, 0], R, j_pick, F, n,
                  rf"20f/5n $L^4$ embedded (D=80) — input $e_{{{j_pick}}}$")

# 20f/5n L2 for contrast (trained with L2, as in small_20f_5n_L2? check)
# small_20f_5n_L2 may not exist — use small_500f_50n_L2? no, keep at 20 for comparability.
# 100f/10n L4 for ratio = 10
R, n, F = load_plain_R("small_100f_10n_L4")
plot_response_bar(axes[1, 1], R, j_pick, F, n,
                  rf"100f/10n $L^4$ (plain) — input $e_{{{j_pick}}}$")

fig.suptitle("One-hot response: $y = W_{\\mathrm{out}}\\,\\mathrm{ReLU}(W_{\\mathrm{in}}\\,e_j)$\n"
             "Red bar = active feature (index j). Dashed lines = predictions from the rank-n Parseval-projection model.",
             fontsize=12)
fig.tight_layout()
fig.savefig("figures/onehot_response.png", dpi=150)
print("Saved figures/onehot_response.png")

# Figure 2: overlay the responses for all 20 features for 20f/5n plain
# This shows the active one is always ~α·n/F, others are spread
R, n, F = load_plain_R("small_20f_5n_L4")
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
for j in range(F):
    # circular-shift so the active feature sits at position 0 of the shifted index
    y = R[:, j]
    # roll so index j -> 0
    y_shifted = np.roll(y, -j)
    ax.plot(np.arange(F), y_shifted, alpha=0.45, linewidth=1.2)
ax.axhline(0, color="k", linewidth=0.5)
alpha = np.diag(R).mean() * F / n
welch = alpha * np.sqrt(n * (F - n) / (F**2 * (F - 1)))
ax.axhline(alpha * n / F, color="tab:red", linestyle="--",
           label=rf"active-feature readout $\alpha\,n/F$ = {alpha * n / F:.3f}")
ax.axhline(welch, color="tab:blue", linestyle=":",
           label=rf"$\pm\alpha\cdot$Welch = $\pm${welch:.3f}")
ax.axhline(-welch, color="tab:blue", linestyle=":")
ax.axhline(1.0, color="tab:green", linestyle="-.", alpha=0.6,
           label="target = 1")
ax.set_xlabel("feature offset (0 = active feature, shifted for overlay)")
ax.set_ylabel(r"$y_{j+k}$ when input is $e_j$")
ax.set_title("20f/5n $L^4$: one-hot response overlaid for all 20 features\n"
             "(each trace = one input $e_j$; active feature circularly shifted to index 0)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("figures/onehot_overlay.png", dpi=150)
print("Saved figures/onehot_overlay.png")

# Also a diagnostic print
print(f"\n20f/5n L4 diagnostic:")
print(f"  diag mean = {np.diag(R).mean():.4f}, std = {np.diag(R).std():.4f}")
print(f"  α = {alpha:.3f}, α·n/F = {alpha * n / F:.3f}, α·Welch = {welch:.3f}")
