"""
Mean per-feature MSE vs model size at a fixed 10:1 expansion ratio, for
L2 and L4 trained models. Per-feature MSE conditions on m_j = 1 (feature
sampled active), so the baseline for a completely-ignored feature is
E[ReLU(x)^2 | m=1] = 1/6.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from small_models import SimpleMLP, generate_batch, DEVICE

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
P = 0.02

CONFIGS = [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), (1000, 100)]


def per_feature_mse_given_active(model, n_features, n_batches=200, batch_size=2048):
    model.eval()
    sq_err_sum = torch.zeros(n_features, device=DEVICE)
    active_count = torch.zeros(n_features, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, n_features, P)
            mask = (x != 0).float()
            err2 = (model(x) - y) ** 2
            sq_err_sum += (err2 * mask).sum(dim=0)
            active_count += mask.sum(dim=0)
    return (sq_err_sum / active_count.clamp(min=1)).cpu().numpy()


def mean_mse(n_features, n_neurons, loss_exp):
    path = WEIGHTS_DIR / f"small_{n_features}f_{n_neurons}n_L{loss_exp}.pt"
    m = SimpleMLP(n_features, n_neurons).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    per_feat = per_feature_mse_given_active(m, n_features)
    return per_feat.mean()


if __name__ == "__main__":
    feats, l2_vals, l4_vals = [], [], []
    for n_feat, n_neur in CONFIGS:
        l2 = mean_mse(n_feat, n_neur, 2)
        l4 = mean_mse(n_feat, n_neur, 4)
        feats.append(n_feat)
        l2_vals.append(l2)
        l4_vals.append(l4)
        print(f"F={n_feat:>4} n={n_neur:>3}  L2={l2:.4f}  L4={l4:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(feats, l4_vals, "o-", color="C0", label="L4", markersize=7)
    ax.plot(feats, l2_vals, "s-", color="C1", label="L2", markersize=7)
    ax.axhline(y=1/6, color="r", linestyle="--", alpha=0.6,
               label=r"loss if feature ignored  $(\mathbb{E}[\mathrm{ReLU}(x)^2 \mid \mathrm{active}] = 1/6)$")
    ax.set_xlabel("number of features  (neurons = features / 10)")
    ax.set_ylabel("mean per-feature MSE given feature active")
    ax.set_title("Mean per-feature MSE vs model size (10:1 expansion, p=0.02)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "scaling_mse_vs_size.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
