"""
Visualize that the embedded L4 model recovers the same 3-direction codeword
structure as the plain L4 model for F=20, n=2, expressed via the effective
weights W_in_eff = W_in @ E (n x F).
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_plain(name):
    sd = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = sd["W_in"].numpy()     # n x F
    W_out = sd["W_out"].numpy()   # F x n
    return W_in, W_out


def load_embedded(name):
    d = torch.load(f"weights/{name}.pt", map_location="cpu")
    W_in = d["W_in"].numpy()    # n x D
    W_out = d["W_out"].numpy()  # D x n
    E = d["E"].numpy()          # D x F
    W_in_eff = W_in @ E         # n x F
    return W_in_eff


def plot_frame(ax, W_2xF, title, color):
    F = W_2xF.shape[1]
    for j in range(F):
        ax.plot([0, W_2xF[0, j]], [0, W_2xF[1, j]], "-", color=color, alpha=0.6, linewidth=1.5)
        ax.plot(W_2xF[0, j], W_2xF[1, j], "o", color=color, markersize=5)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")


plain_Win, _ = load_plain("small_20f_2n_L4")
emb_D20 = load_embedded("embed_20f_2n_D20_orth_L4")
emb_D40 = load_embedded("embed_20f_2n_D40_unit_L4")
emb_D80 = load_embedded("embed_20f_2n_D80_unit_L4")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
plot_frame(axes[0], plain_Win,
           f"Plain model\n(no embedding)", "tab:blue")
plot_frame(axes[1], emb_D20,
           f"Embedded, D=F=20\nE orthogonal", "tab:orange")
plot_frame(axes[2], emb_D40,
           f"Embedded, D=40\nE Gaussian unit-cols", "tab:green")
plot_frame(axes[3], emb_D80,
           f"Embedded, D=80\nE Gaussian unit-cols", "tab:red")

for ax in axes:
    lim = 0.7
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

fig.suptitle("Effective W_in columns (n=2, F=20, L$^4$)  —  20 features plotted as 2D vectors\n"
             "All models: 3 clusters corresponding to codewords (1,1), (1,0), (0,1). No (0,0) cluster.",
             fontsize=13)
fig.tight_layout()
fig.savefig("figures/step2_frames.png", dpi=150)
print("Saved figures/step2_frames.png")
