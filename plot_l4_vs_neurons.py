"""
Plot L4 loss vs neuron count for:
  (i)   ReLU model trained with L4 loss
  (ii)  Linear (no-ReLU) model trained with L4 loss
  (iii) Naive baseline (y_hat = 0)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


class ReLUMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)
    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


class LinearMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)
    def forward(self, x):
        return (self.W_out @ (self.W_in @ x.T)).T


def generate_batch(batch_size, n_features, p):
    mask = (torch.rand(batch_size, n_features, device=DEVICE) < p).float()
    values = torch.rand(batch_size, n_features, device=DEVICE) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


def l4_loss(model, n_features, p, n_batches=200):
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(2048, n_features, p)
            y_hat = model(x) if model is not None else torch.zeros_like(y)
            losses.append(((y_hat - y).abs() ** 4).mean().item())
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(n_batches))


if __name__ == "__main__":
    torch.manual_seed(0)
    configs = [(10, 1), (20, 2), (100, 10)]
    p = 0.02

    rows = []
    for n_feat, n_neur in configs:
        relu = ReLUMLP(n_feat, n_neur).to(DEVICE)
        relu.load_state_dict(torch.load(WEIGHTS_DIR / f"small_{n_feat}f_{n_neur}n_L4.pt", map_location=DEVICE))
        relu.eval()
        lin = LinearMLP(n_feat, n_neur).to(DEVICE)
        lin.load_state_dict(torch.load(WEIGHTS_DIR / f"linear_{n_feat}f_{n_neur}n_L4.pt", map_location=DEVICE))
        lin.eval()

        relu_l4, relu_se = l4_loss(relu, n_feat, p)
        lin_l4, lin_se = l4_loss(lin, n_feat, p)
        naive_l4, naive_se = l4_loss(None, n_feat, p)
        rows.append((n_neur, relu_l4, lin_l4, naive_l4, relu_se, lin_se, naive_se))
        print(f"{n_feat}f/{n_neur}n: ReLU={relu_l4:.6f}  Linear={lin_l4:.6f}  Naive={naive_l4:.6f}")

    rows = np.array(rows)
    ns = rows[:, 0]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, rows[:, 1], "o-", label="ReLU model (L4-trained)")
    plt.plot(ns, rows[:, 2], "s-", label="Linear model (L4-trained)")
    plt.plot(ns, rows[:, 3], "^--", label="Naive baseline (y_hat=0)", color="gray")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("number of neurons (10:1 feature:neuron ratio)")
    plt.ylabel("L4 loss  (mean |y_hat - y|^4)")
    plt.title("L4 loss vs model size (p=0.02)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "l4_loss_vs_neurons.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
