"""
Produce the two visuals for slides 2 and 3:
  slide2: W_in columns as 2D points, colored by sign pattern. Shows the three
          empirically-observed groups and the empty fourth quadrant.
  slide3: schematic of the three-regime linear map. The (z1, z2) plane is split
          into four ReLU regions; each nonzero region corresponds to one group
          of features and is served by its own effective linear map.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from small_models import SimpleMLP, DEVICE

FIGURES_DIR = Path("figures")
WEIGHTS_DIR = Path("weights")
F = 20
N = 2

GROUP_LABELS = {(1, 1): "A  (++)", (1, -1): "B  (+-)", (-1, 1): "C  (-+)", (-1, -1): "empty (--)"}
GROUP_COLORS = {(1, 1): "#1f77b4", (1, -1): "#d62728", (-1, 1): "#2ca02c", (-1, -1): "#888888"}


def load_trained():
    sd = torch.load(WEIGHTS_DIR / "small_20f_2n_L4.pt", map_location=DEVICE,
                    weights_only=True)
    return sd["W_in"].cpu().numpy(), sd["W_out"].cpu().numpy()


def _group_features(W_in):
    signs = np.sign(W_in)
    groups = {}
    for j in range(F):
        key = (int(signs[0, j]), int(signs[1, j]))
        groups.setdefault(key, []).append(j)
    return groups


def _annotated_scatter(xs_by_group, out_path, xlabel, ylabel, title,
                       annot_offsets):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    lim = 0.9
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.fill_between([0, lim], 0, lim, color=GROUP_COLORS[(1, 1)], alpha=0.08)
    ax.fill_between([0, lim], -lim, 0, color=GROUP_COLORS[(1, -1)], alpha=0.08)
    ax.fill_between([-lim, 0], 0, lim, color=GROUP_COLORS[(-1, 1)], alpha=0.08)
    ax.fill_between([-lim, 0], -lim, 0, color=GROUP_COLORS[(-1, -1)], alpha=0.08)

    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, linestyle="--",
            alpha=0.6, zorder=1)

    centers = {}
    for key, pts in xs_by_group.items():
        pts = np.asarray(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=80,
                   color=GROUP_COLORS[key], edgecolor="black", linewidth=0.6,
                   alpha=0.55,
                   label=f"{GROUP_LABELS[key]}  (n={len(pts)})", zorder=5)
        centers[key] = pts.mean(axis=0)

    for key, c in centers.items():
        dx, dy = annot_offsets.get(key, (0.08, 0.08))
        ax.annotate(f"({c[0]:+.2f}, {c[1]:+.2f})",
                    xy=(c[0], c[1]), xytext=(c[0] + dx, c[1] + dy),
                    fontsize=11, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=GROUP_COLORS[key], alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=GROUP_COLORS[key],
                                    lw=0.7), zorder=6)

    if (1, -1) in centers and (-1, 1) in centers:
        b = centers[(1, -1)]
        c = centers[(-1, 1)]
        ax.annotate("", xy=(c[0], c[1]), xytext=(b[0], b[1]),
                    arrowprops=dict(arrowstyle="<->", color="gray",
                                    lw=1.0, alpha=0.7,
                                    connectionstyle="arc3,rad=0.25"),
                    zorder=4)
        mid = 0.5 * (b + c)
        ax.text(mid[0] - 0.08, mid[1] - 0.08, "swap neurons",
                fontsize=9, color="gray", ha="center", va="center",
                rotation=45)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved {out_path}")


def make_slide2_scatter(W_in):
    groups = _group_features(W_in)
    xs_by_group = {key: [(W_in[0, j], W_in[1, j]) for j in idx]
                   for key, idx in groups.items()}
    offsets = {(1, 1): (0.0, -0.18), (1, -1): (0.0, -0.18), (-1, 1): (-0.22, 0.0)}
    _annotated_scatter(
        xs_by_group,
        FIGURES_DIR / "slide2_win_scatter.png",
        xlabel=r"$W_{\mathrm{in}}[1, j]$  (routing to neuron 1)",
        ylabel=r"$W_{\mathrm{in}}[2, j]$  (routing to neuron 2)",
        title=r"$W_{\mathrm{in}}$ columns: three clusters $\Rightarrow$ 6 numbers",
        annot_offsets=offsets,
    )


def make_slide2_wout_scatter(W_in, W_out):
    groups = _group_features(W_in)
    xs_by_group = {key: [(W_out[j, 0], W_out[j, 1]) for j in idx]
                   for key, idx in groups.items()}
    offsets = {(1, 1): (0.0, -0.2), (1, -1): (0.0, -0.2), (-1, 1): (-0.25, 0.0)}
    _annotated_scatter(
        xs_by_group,
        FIGURES_DIR / "slide2_wout_scatter.png",
        xlabel=r"$W_{\mathrm{out}}[j, 1]$  (readout from neuron 1)",
        ylabel=r"$W_{\mathrm{out}}[j, 2]$  (readout from neuron 2)",
        title=r"$W_{\mathrm{out}}$ rows: same three clusters, same symmetry",
        annot_offsets=offsets,
    )


def make_slide3(W_in, W_out):
    signs = np.sign(W_in)
    groups = {}
    for j in range(F):
        key = (int(signs[0, j]), int(signs[1, j]))
        groups.setdefault(key, []).append(j)

    fig, ax = plt.subplots(figsize=(9, 7.5))

    lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=1.0)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_aspect("equal")

    ax.fill_between([0, lim], 0, lim, color=GROUP_COLORS[(1, 1)], alpha=0.12)
    ax.fill_between([0, lim], -lim, 0, color=GROUP_COLORS[(1, -1)], alpha=0.12)
    ax.fill_between([-lim, 0], 0, lim, color=GROUP_COLORS[(-1, 1)], alpha=0.12)
    ax.fill_between([-lim, 0], -lim, 0, color=GROUP_COLORS[(-1, -1)], alpha=0.12)

    def region_label(xy, lines):
        ax.annotate("\n".join(lines), xy=xy, ha="center", va="center",
                    fontsize=10.5,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec="gray", alpha=0.92))

    a_params = (W_in[:, groups[(1, 1)]].mean(axis=1),
                W_out[groups[(1, 1)], :].mean(axis=0))
    b_params = (W_in[:, groups[(1, -1)]].mean(axis=1),
                W_out[groups[(1, -1)], :].mean(axis=0))
    c_params = (W_in[:, groups[(-1, 1)]].mean(axis=1),
                W_out[groups[(-1, 1)], :].mean(axis=0))

    region_label((0.5, 0.5),
                 ["both neurons fire",
                  r"feature group $A$ served",
                  f"{len(groups[(1,1)])} features, prototype",
                  f"$W_{{\\rm in}} = ({a_params[0][0]:+.2f}, {a_params[0][1]:+.2f})$",
                  f"$W_{{\\rm out}} = ({a_params[1][0]:+.2f}, {a_params[1][1]:+.2f})$"])
    region_label((0.5, -0.5),
                 ["only neuron 1 fires",
                  r"feature group $B$ served",
                  f"{len(groups[(1,-1)])} features, prototype",
                  f"$W_{{\\rm in}} = ({b_params[0][0]:+.2f}, {b_params[0][1]:+.2f})$",
                  f"$W_{{\\rm out}} = ({b_params[1][0]:+.2f}, {b_params[1][1]:+.2f})$"])
    region_label((-0.5, 0.5),
                 ["only neuron 2 fires",
                  r"feature group $C$ served",
                  f"{len(groups[(-1,1)])} features, prototype",
                  f"$W_{{\\rm in}} = ({c_params[0][0]:+.2f}, {c_params[0][1]:+.2f})$",
                  f"$W_{{\\rm out}} = ({c_params[1][0]:+.2f}, {c_params[1][1]:+.2f})$"])
    region_label((-0.5, -0.5),
                 ["no neurons fire",
                  r"output $= 0$",
                  "(empirically, no",
                  "features land here)"])

    ax.set_xlabel(r"preactivation $z_1$ of neuron 1")
    ax.set_ylabel(r"preactivation $z_2$ of neuron 2")
    ax.set_title("Three-regime linear map:\n"
                 "which neurons fire selects which group of features is served",
                 fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    out = FIGURES_DIR / "slide3_three_regimes.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    W_in, W_out = load_trained()
    make_slide2_scatter(W_in)
    make_slide2_wout_scatter(W_in, W_out)
    make_slide3(W_in, W_out)
