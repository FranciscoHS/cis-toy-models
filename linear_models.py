"""
Train linear (no-ReLU) models with L4 loss to see what encoding/decoding alone does.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


class LinearMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ (self.W_in @ x.T)).T


def generate_batch(batch_size, n_features, p, device=DEVICE):
    mask = (torch.rand(batch_size, n_features, device=device) < p).float()
    values = torch.rand(batch_size, n_features, device=device) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


def train_model(n_features, n_neurons, p, loss_exp, n_batches=10000):
    model = LinearMLP(n_features, n_neurons).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_batches)
    for step in range(n_batches):
        x, y = generate_batch(2048, n_features, p)
        y_hat = model(x)
        loss = (torch.abs(y_hat - y) ** loss_exp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model


def evaluate_per_feature(model, n_features, p):
    model.eval()
    per_feat = torch.zeros(n_features, device=DEVICE)
    count = torch.zeros(n_features, device=DEVICE)
    with torch.no_grad():
        for _ in range(100):
            x, y = generate_batch(2048, n_features, p)
            active = (x > 0).float()
            err = ((model(x) - y) ** 2 * active).sum(dim=0)
            per_feat += err
            count += active.sum(dim=0)
    per_feat = (per_feat / count.clamp(min=1)).cpu().numpy()
    model.train()
    return per_feat


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Device: {DEVICE}")

    configs = [(10, 1, 0.02), (20, 2, 0.02), (100, 10, 0.02)]

    for n_feat, n_neur, p in configs:
        print(f"\n=== LINEAR {n_feat}f, {n_neur}n (ratio {n_feat // n_neur}:1) ===")
        for loss_exp in [2, 4]:
            model = train_model(n_feat, n_neur, p, loss_exp)
            per_feat = evaluate_per_feature(model, n_feat, p)
            sorted_err = np.sort(per_feat)
            n_well = (per_feat < 0.05).sum()
            print(f"  L{loss_exp}: {n_well}/{n_feat} features < 0.05 MSE, "
                  f"mean={per_feat.mean():.4f}, min={sorted_err[0]:.4f}, max={sorted_err[-1]:.4f}")
            tag = f"linear_{n_feat}f_{n_neur}n_L{loss_exp}"
            torch.save(model.state_dict(), WEIGHTS_DIR / f"{tag}.pt")

    print("\nDone!")
