"""
Binary-code claim: features partition across sign-pattern codewords, with
a small number of empty codewords. Panels show the histogram of
#features-per-codeword for F=20 and N=2,3,4,5, sorted descending.
Horizontal reference at F/(2^N - 1) is the balanced-partition prediction.
"""

import itertools
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from small_models import SimpleMLP, DEVICE

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
F = 20
NS = [2, 3, 4, 5]


def load_W_in(n_neurons):
    sd = torch.load(WEIGHTS_DIR / f"small_{F}f_{n_neurons}n_L4.pt",
                    map_location=DEVICE, weights_only=True)
    return sd["W_in"].cpu().numpy()


def codeword_counts(W_in, n_neurons):
    signs = np.sign(W_in)
    all_codewords = list(itertools.product([-1, 1], repeat=n_neurons))
    counts = {cw: 0 for cw in all_codewords}
    for j in range(F):
        cw = tuple(int(s) for s in signs[:, j])
        counts[cw] += 1
    return counts


def make_figure():
    fig, axes = plt.subplots(1, len(NS), figsize=(16, 4.2), sharey=True)

    for ax, n in zip(axes, NS):
        W_in = load_W_in(n)
        counts = codeword_counts(W_in, n)
        sorted_pairs = sorted(counts.items(), key=lambda kv: -kv[1])
        sorted_counts = [c for _, c in sorted_pairs]
        sorted_labels = ["".join("1" if s > 0 else "0" for s in cw)
                         for cw, _ in sorted_pairs]
        n_codewords = 2 ** n
        n_nonzero_codewords = n_codewords - 1
        pred = F / n_nonzero_codewords
        x = np.arange(n_codewords)

        bar_colors = ["#1f77b4" if c > 0 else "#cccccc" for c in sorted_counts]
        ax.bar(x, sorted_counts, color=bar_colors, edgecolor="black",
               linewidth=0.5, width=0.8)
        ax.axhline(pred, color="red", linestyle="--", lw=1.2,
                   label=f"$F/(2^n\\!-\\!1) = {pred:.2f}$")

        ax.set_title(f"$n = {n}$   ($2^n = {n_codewords}$)", fontsize=12)
        ax.set_xlabel("codeword (sorted by count)")
        if n <= 3:
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_labels, fontsize=9,
                               family="monospace")
        else:
            ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("# features assigned")
    fig.suptitle(r"Features partition across sign-pattern codewords  "
                 r"($F = 20$, $L^4$-trained)",
                 fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "slide_codeword_hist.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure()
