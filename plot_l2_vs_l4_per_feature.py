"""
Per-feature MSE comparison for the trained L2 and L4 models, conditional on
the feature being sampled active (m_j = 1). Under that conditioning, a model
that ignores a feature entirely (always outputs 0 for it) has expected MSE
E[ReLU(x)^2 | m=1] = E[max(v,0)^2 | v ~ U(-1,1)] = 1/6. The original plot
conditioned on x > 0 instead, which makes the 1/6 reference line off by 2x.
This script fixes that and plots only the per-feature comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cis_experiment import SimpleMLP, generate_batch, DEVICE, N_FEATURES, N_NEURONS, BATCH_SIZE, P

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")


def per_feature_mse_given_active(model, n_batches=200, batch_size=BATCH_SIZE):
    model.eval()
    sq_err_sum = torch.zeros(N_FEATURES, device=DEVICE)
    active_count = torch.zeros(N_FEATURES, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size)
            mask = (x != 0).float()
            err2 = (model(x) - y) ** 2
            sq_err_sum += (err2 * mask).sum(dim=0)
            active_count += mask.sum(dim=0)
    return (sq_err_sum / active_count.clamp(min=1)).cpu().numpy()


def load_model(path):
    m = SimpleMLP().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m


if __name__ == "__main__":
    model_l2 = load_model(WEIGHTS_DIR / "model_l2.pt")
    model_l4 = load_model(WEIGHTS_DIR / "model_l4.pt")

    err_l2 = per_feature_mse_given_active(model_l2)
    err_l4 = per_feature_mse_given_active(model_l4)

    ignore_level = 1.0 / 6.0

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(np.sort(err_l2), "o", markersize=2.5, alpha=0.75, label="L2-trained")
    ax.plot(np.sort(err_l4), "o", markersize=2.5, alpha=0.75, label="L4-trained")
    ax.axhline(y=ignore_level, color="r", linestyle="--", alpha=0.6,
               label=r"loss if feature ignored  $(\mathbb{E}[\mathrm{ReLU}(x)^2 \mid \mathrm{active}] = 1/6)$")
    ax.axvline(x=N_NEURONS, color="gray", linestyle=":", alpha=0.6,
               label=f"n_neurons = {N_NEURONS}")
    ax.set_xlabel("feature index (sorted by error)")
    ax.set_ylabel(r"MSE given feature active")
    ax.set_title(f"Per-feature error: L2 vs L4  "
                 f"({N_FEATURES} features, {N_NEURONS} neurons, p={P})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "l2_vs_l4_per_feature.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"L2 mean per-feature MSE: {err_l2.mean():.4f}  "
          f"max: {err_l2.max():.4f}  median: {np.median(err_l2):.4f}")
    print(f"L4 mean per-feature MSE: {err_l4.mean():.4f}  "
          f"max: {err_l4.max():.4f}  median: {np.median(err_l4):.4f}")
