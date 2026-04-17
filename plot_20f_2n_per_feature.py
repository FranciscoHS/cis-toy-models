"""
Per-feature MSE (given feature active) for the F=20, n=2 models, sorted.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from small_models import SimpleMLP, generate_batch, DEVICE

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
F = 20
N = 2
P = 0.02


def per_feature_mse_given_active(model, n_batches=400, batch_size=2048):
    model.eval()
    sq_err_sum = torch.zeros(F, device=DEVICE)
    active_count = torch.zeros(F, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, F, P)
            mask = (x != 0).float()
            err2 = (model(x) - y) ** 2
            sq_err_sum += (err2 * mask).sum(dim=0)
            active_count += mask.sum(dim=0)
    return (sq_err_sum / active_count.clamp(min=1)).cpu().numpy()


def load(loss_exp):
    m = SimpleMLP(F, N).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_DIR / f"small_{F}f_{N}n_L{loss_exp}.pt",
                                 map_location=DEVICE, weights_only=True))
    return m


if __name__ == "__main__":
    err_l2 = per_feature_mse_given_active(load(2))
    err_l4 = per_feature_mse_given_active(load(4))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(np.sort(err_l2), "s-", color="C1", markersize=6, label="L2-trained")
    ax.plot(np.sort(err_l4), "o-", color="C0", markersize=6, label="L4-trained")
    ax.axhline(y=1/6, color="r", linestyle="--", alpha=0.6,
               label=r"loss if feature ignored  $(\mathbb{E}[\mathrm{ReLU}(x)^2 \mid \mathrm{active}] = 1/6)$")
    ax.axvline(x=N, color="gray", linestyle=":", alpha=0.6,
               label=f"n_neurons = {N}")
    ax.set_xlabel("feature index (sorted by error)")
    ax.set_ylabel("MSE given feature active")
    ax.set_title(f"Per-feature error: L2 vs L4  ({F} features, {N} neurons, p={P})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "l2_vs_l4_per_feature_20f_2n.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"L2 sorted: {np.round(np.sort(err_l2), 4)}")
    print(f"L4 sorted: {np.round(np.sort(err_l4), 4)}")
