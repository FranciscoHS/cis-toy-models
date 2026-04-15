"""
Analyze the geometry of the L4-trained model's weight matrices.

Computes:
1. Gram matrices (G_in = W_in^T W_in, G_out = W_out^T W_out) and their
   cosine-similarity versions
2. Histograms of pairwise cosine similarities vs the Welch bound
3. W_out/W_in alignment (per-feature cosine similarity)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cis_experiment import SimpleMLP, N_FEATURES, N_NEURONS

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_model(path):
    model = SimpleMLP()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def cosine_sim_matrix(vecs):
    """Cosine similarity matrix for a set of row vectors."""
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vecs / norms
    return normed @ normed.T


def analyze_gram(model, label=""):
    # W_in: (N_NEURONS, N_FEATURES) -- columns are feature encoding directions
    # W_out: (N_FEATURES, N_NEURONS) -- rows are feature decoding directions
    W_in = model.W_in.data    # (50, 500)
    W_out = model.W_out.data  # (500, 50)

    # Feature directions in hidden space:
    #   encoding: columns of W_in = rows of W_in^T, shape (500, 50)
    #   decoding: rows of W_out, shape (500, 50)
    enc_dirs = W_in.T  # (500, 50)
    dec_dirs = W_out    # (500, 50)

    # Gram matrices (inner products between feature directions)
    G_in = enc_dirs @ enc_dirs.T    # (500, 500)
    G_out = dec_dirs @ dec_dirs.T   # (500, 500)

    # Cosine similarity matrices
    C_in = cosine_sim_matrix(enc_dirs)
    C_out = cosine_sim_matrix(dec_dirs)

    # Welch bound: minimum max-coherence for 500 vectors in R^50
    # mu >= sqrt((N - d) / (d * (N - 1)))
    N, d = N_FEATURES, N_NEURONS
    welch = np.sqrt((N - d) / (d * (N - 1)))
    print(f"Welch bound: {welch:.4f}")

    # Extract off-diagonal elements
    mask = ~torch.eye(N_FEATURES, dtype=bool)
    cos_in_off = C_in[mask].numpy()
    cos_out_off = C_out[mask].numpy()

    # Per-feature W_out/W_in alignment
    enc_norms = enc_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    dec_norms = dec_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    alignment = ((enc_dirs / enc_norms) * (dec_dirs / dec_norms)).sum(dim=1).numpy()

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Gram matrix heatmaps
    im0 = axes[0, 0].imshow(C_in.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0, 0].set_title(f"Cosine sim: W_in columns ({label})")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(C_out.numpy(), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0, 1].set_title(f"Cosine sim: W_out rows ({label})")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Difference
    diff = (C_in - C_out).numpy()
    im2 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    axes[0, 2].set_title("C_in - C_out")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # Row 2: Histograms and alignment
    axes[1, 0].hist(cos_in_off, bins=100, alpha=0.7, density=True, label='W_in')
    axes[1, 0].axvline(welch, color='r', linestyle='--', label=f'Welch bound ({welch:.3f})')
    axes[1, 0].axvline(-welch, color='r', linestyle='--')
    axes[1, 0].set_xlabel("Pairwise cosine similarity")
    axes[1, 0].set_title(f"W_in pairwise cosines ({label})")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(cos_out_off, bins=100, alpha=0.7, density=True, label='W_out')
    axes[1, 1].axvline(welch, color='r', linestyle='--', label=f'Welch bound ({welch:.3f})')
    axes[1, 1].axvline(-welch, color='r', linestyle='--')
    axes[1, 1].set_xlabel("Pairwise cosine similarity")
    axes[1, 1].set_title(f"W_out pairwise cosines ({label})")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(alignment, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(np.mean(alignment), color='r', linestyle='--',
                        label=f'Mean = {np.mean(alignment):.3f}')
    axes[1, 2].set_xlabel("Cosine similarity (enc_i, dec_i)")
    axes[1, 2].set_title(f"W_out/W_in alignment per feature ({label})")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f"Weight geometry: {label}", fontsize=14)
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("=", "").replace("^", "")
    fig.savefig(FIGURES_DIR / f"geometry_{safe}.png", dpi=150)
    plt.close(fig)

    # Print stats
    print(f"\n--- {label} ---")
    print(f"  W_in cosines:  mean={cos_in_off.mean():.4f}, std={cos_in_off.std():.4f}, "
          f"max={np.abs(cos_in_off).max():.4f}")
    print(f"  W_out cosines: mean={cos_out_off.mean():.4f}, std={cos_out_off.std():.4f}, "
          f"max={np.abs(cos_out_off).max():.4f}")
    print(f"  Enc/dec alignment: mean={alignment.mean():.4f}, std={alignment.std():.4f}, "
          f"min={alignment.min():.4f}, max={alignment.max():.4f}")

    return {
        'C_in': C_in, 'C_out': C_out,
        'cos_in_off': cos_in_off, 'cos_out_off': cos_out_off,
        'alignment': alignment, 'welch': welch,
    }


if __name__ == "__main__":
    print("Loading models...")
    model_l4 = load_model("weights/model_l4.pt")
    model_l2 = load_model("weights/model_l2.pt")

    results_l4 = analyze_gram(model_l4, label="L4")
    results_l2 = analyze_gram(model_l2, label="L2")

    # Side-by-side comparison of cosine distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(results_l2['cos_in_off'], bins=100, alpha=0.5, density=True, label='L2')
    axes[0].hist(results_l4['cos_in_off'], bins=100, alpha=0.5, density=True, label='L4')
    axes[0].axvline(results_l4['welch'], color='r', linestyle='--',
                     label=f'Welch ({results_l4["welch"]:.3f})')
    axes[0].axvline(-results_l4['welch'], color='r', linestyle='--')
    axes[0].set_title("W_in pairwise cosines: L2 vs L4")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(results_l2['cos_out_off'], bins=100, alpha=0.5, density=True, label='L2')
    axes[1].hist(results_l4['cos_out_off'], bins=100, alpha=0.5, density=True, label='L4')
    axes[1].axvline(results_l4['welch'], color='r', linestyle='--',
                     label=f'Welch ({results_l4["welch"]:.3f})')
    axes[1].axvline(-results_l4['welch'], color='r', linestyle='--')
    axes[1].set_title("W_out pairwise cosines: L2 vs L4")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(results_l2['alignment'], bins=50, alpha=0.5, label='L2')
    axes[2].hist(results_l4['alignment'], bins=50, alpha=0.5, label='L4')
    axes[2].set_title("Enc/Dec alignment: L2 vs L4")
    axes[2].set_xlabel("Cosine similarity (enc_i, dec_i)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "geometry_L2_vs_L4.png", dpi=150)
    plt.close(fig)

    print("\n-> Saved geometry_L4.png, geometry_L2.png, geometry_L2_vs_L4.png")
    print("Done!")
