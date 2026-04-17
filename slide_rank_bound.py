"""
Two-panel slide visual for the rank-n bound on single-feature recovery
(F=20, n=5).

Left panel: singular values of R = W_out @ ReLU(W_in[:,j]) for the trained
            linear model, against the identity's spectrum. Visualizes
            Eckart-Young: only n=5 of 20 directions can be aligned, and
            each missing direction contributes >=1 to ||R - I||_F^2.

Right panel: three-bar comparison of ||sR - I||_F^2 (with optimal
             uniform rescale s):
             - trained linear (saturates the bound)
             - rank-n theoretical bound (F - n = 15)
             - trained MLP decoder (rank lifted; far below the bound)
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from small_models import SimpleMLP, DEVICE
from mlp_decoder import BottleneckMLP

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
F = 20
N = 5


def linear_R(W_in, W_out):
    Z = np.maximum(W_in, 0.0)
    return W_out @ Z


def mlp_R(model):
    model.eval()
    R = np.zeros((F, F))
    with torch.no_grad():
        for j in range(F):
            x = torch.zeros(1, F, device=DEVICE)
            x[0, j] = 1.0
            R[:, j] = model(x).cpu().numpy()[0]
    return R


def opt_scale_loss(R):
    s = np.trace(R) / (R ** 2).sum()
    err = ((s * R) - np.eye(F)) ** 2
    return s, err.sum()


def best_rank_n_of_I(W_out, n):
    """Orthogonal projection onto image(W_out). Achieves ||P - I||_F^2 = F - n."""
    Q, _ = np.linalg.qr(W_out)
    return Q @ Q.T


def make_figure():
    sd_lin = torch.load(WEIGHTS_DIR / f"small_{F}f_{N}n_L4.pt",
                        map_location=DEVICE, weights_only=True)
    W_in = sd_lin["W_in"].cpu().numpy()
    W_out = sd_lin["W_out"].cpu().numpy()
    R_lin = linear_R(W_in, W_out)
    s_lin, err_lin = opt_scale_loss(R_lin)
    sR_lin = s_lin * R_lin
    P_rank = best_rank_n_of_I(W_out, N)
    err_rank = ((P_rank - np.eye(F)) ** 2).sum()

    mlp = BottleneckMLP(F, N).to(DEVICE)
    mlp.load_state_dict(torch.load(WEIGHTS_DIR / f"mlp_decoder_{F}f_{N}n_L4.pt",
                                   map_location=DEVICE, weights_only=True))
    R_mlp = mlp_R(mlp)
    s_mlp, err_mlp = opt_scale_loss(R_mlp)
    sR_mlp = s_mlp * R_mlp

    bound = F - N

    print(f"Linear: s={s_lin:.4f}  ||sR - I||^2 = {err_lin:.3f}  (bound {bound})")
    print(f"Rank-{N} projection of I: ||P - I||^2 = {err_rank:.3f}")
    print(f"MLP:    s={s_mlp:.4f}  ||sR - I||^2 = {err_mlp:.3f}")

    fig = plt.figure(figsize=(16.5, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25],
                          wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    cmap = "RdBu_r"
    vmax = 1.0

    panels = [
        (np.eye(F), "target: identity $I$\n(perfect recovery)"),
        (sR_lin, f"trained linear, $n = 5$\n$\\|sR - I\\|_F^2 = {err_lin:.2f}$"),
        (P_rank, f"best rank-5 projection of $I$\n"
                 f"$\\|P - I\\|_F^2 = {err_rank:.2f}$  (= $F - n$)"),
        (sR_mlp, f"MLP decoder (rank lifted)\n"
                 f"$\\|sR - I\\|_F^2 = {err_mlp:.2f}$"),
    ]
    for ax, (M, title) in zip(axes[:4], panels):
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes[:4], orientation="horizontal",
                        fraction=0.06, pad=0.08, shrink=0.5)
    cbar.set_label("entry value", fontsize=10)

    axR = axes[4]
    bars = ["trained\nlinear", f"rank-{N}\nbound", "trained\nMLP"]
    vals = [err_lin, bound, err_mlp]
    colors = ["#1f77b4", "#888888", "#2ca02c"]
    xs = np.arange(len(bars))
    axR.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    for x, v in zip(xs, vals):
        axR.text(x, v + max(vals) * 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=11)
    axR.axhline(bound, color="gray", linestyle="--", lw=1.0, alpha=0.7,
                label=f"rank-{N} floor = $F - n = {bound}$")
    axR.set_xticks(xs)
    axR.set_xticklabels(bars, fontsize=11)
    axR.set_ylabel(r"$\|sR - I\|_F^2$")
    axR.set_title("trained linear saturates the floor;\n"
                  "MLP breaks past it")
    axR.set_ylim(0, max(vals) * 1.25)
    axR.legend(loc="upper right", fontsize=9)
    axR.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        r"Rank-$n$ bottleneck on single-feature recovery  ($F = 20$, $n = 5$)",
        fontsize=13, y=1.02)
    out = FIGURES_DIR / "slide_rank_bound.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure()
