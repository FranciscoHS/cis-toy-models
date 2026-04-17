"""
Rank-n bottleneck slide, extended to n in {2, 3, 4, 5}.

For each n, we have:
 - a trained linear 20f/nn L4 model
 - a trained MLP decoder 20f/nn L4 (rank constraint lifted)

We compute ||sR - I||_F^2 (optimal uniform rescale s) for each, alongside
the rank-n theoretical lower bound F - n. The bar plot shows how both
bottlenecks (identification + rank) ease as n grows:
 - Linear bars hug the rank-n floor across all n.
 - MLP bars sit between the linear bar and zero: closer to zero when
   identification has enough codewords (n=4,5), further from zero when
   codewords are the limit (n=2,3).
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from small_models import DEVICE
from mlp_decoder import BottleneckMLP

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
F = 20
NS = [2, 3, 4, 5]


def linear_R(W_in, W_out):
    return W_out @ np.maximum(W_in, 0.0)


def mlp_R(model):
    model.eval()
    R = np.zeros((F, F))
    with torch.no_grad():
        for j in range(F):
            x = torch.zeros(1, F, device=DEVICE)
            x[0, j] = 1.0
            R[:, j] = model(x).cpu().numpy()[0]
    return R


def opt_scale_err(R):
    s = np.trace(R) / (R ** 2).sum()
    return s, ((s * R - np.eye(F)) ** 2).sum()


def make_figure():
    rows = []
    for n in NS:
        sd = torch.load(WEIGHTS_DIR / f"small_{F}f_{n}n_L4.pt",
                        map_location=DEVICE, weights_only=True)
        R_lin = linear_R(sd["W_in"].cpu().numpy(), sd["W_out"].cpu().numpy())
        _, err_lin = opt_scale_err(R_lin)

        mlp = BottleneckMLP(F, n).to(DEVICE)
        mlp.load_state_dict(torch.load(
            WEIGHTS_DIR / f"mlp_decoder_{F}f_{n}n_L4.pt",
            map_location=DEVICE, weights_only=True))
        R_mlp = mlp_R(mlp)
        _, err_mlp = opt_scale_err(R_mlp)

        rows.append(dict(n=n, err_lin=float(err_lin),
                         err_mlp=float(err_mlp), bound=float(F - n)))
        n_codewords = 2 ** n - 1
        print(f"n={n}  codewords={n_codewords:>2}  "
              f"linear={err_lin:.3f}  bound={F - n:>2}  MLP={err_mlp:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(NS))
    width = 0.32

    ax.bar(xs - width / 2, [r["err_lin"] for r in rows], width,
           color="#1f77b4", edgecolor="black", linewidth=0.6,
           label="trained linear")
    ax.bar(xs + width / 2, [r["err_mlp"] for r in rows], width,
           color="#2ca02c", edgecolor="black", linewidth=0.6,
           label="trained MLP decoder")

    for xi, r in zip(xs, rows):
        ax.plot([xi - width, xi + width], [r["bound"], r["bound"]],
                color="red", lw=2.5, zorder=3)
    ax.plot([], [], color="red", lw=2.5, label=r"rank-$n$ floor  ($F - n$)")

    for xi, r in zip(xs, rows):
        ax.text(xi - width / 2, r["err_lin"] + 0.25, f"{r['err_lin']:.2f}",
                ha="center", va="bottom", fontsize=9, color="#1f77b4")
        ax.text(xi + width / 2, r["err_mlp"] + 0.25, f"{r['err_mlp']:.2f}",
                ha="center", va="bottom", fontsize=9, color="#2ca02c")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"$n = {n}$\n($2^n - 1 = {2**n - 1}$ codewords)"
                        for n in NS], fontsize=10)
    ax.set_ylabel(r"$\|sR - I\|_F^2$  (optimal uniform $s$)")
    ax.set_title("Single-feature recovery error vs number of neurons  "
                 r"($F = 20$)")
    ax.set_ylim(0, max(r["err_lin"] for r in rows) * 1.2)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "slide_rank_bound_sweep.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure()
