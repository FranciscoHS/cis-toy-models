"""
Train models at limit cases:
 - F = n (trivial, expect clean ReLU)
 - F = 2^n - 1 (one codeword per feature, purest regime)
 - F = 2^n - 1 at several n
 - fixed F=20 at larger n (including n=5..8)
 - fixed n=5, varying F

Uses same training recipe as small_models.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


class SimpleMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


def generate_batch(batch_size, n_features, p, device=DEVICE):
    mask = (torch.rand(batch_size, n_features, device=device) < p).float()
    values = torch.rand(batch_size, n_features, device=device) * 2 - 1
    x = mask * values
    y = torch.relu(x)
    return x, y


def train_model(n_features, n_neurons, p, loss_exp, n_batches=10000, seed=0):
    torch.manual_seed(seed)
    model = SimpleMLP(n_features, n_neurons).to(DEVICE)
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


if __name__ == "__main__":
    np.random.seed(42)
    print(f"Device: {DEVICE}")

    # F = 2^n - 1 (one codeword per feature regime)
    configs = [
        # (F, n, p, tag)
        (3, 2, 0.1, "oct_3f_2n"),     # 2^2-1 = 3
        (7, 3, 0.05, "oct_7f_3n"),    # 2^3-1 = 7
        (15, 4, 0.03, "oct_15f_4n"),  # 2^4-1 = 15
        (31, 5, 0.02, "oct_31f_5n"),  # 2^5-1 = 31
        # F = n (trivial)
        (2, 2, 0.1, "trivial_2f_2n"),
        (3, 3, 0.1, "trivial_3f_3n"),
        (5, 5, 0.1, "trivial_5f_5n"),
        # varying n at F=20, higher n
        (20, 6, 0.02, "small_20f_6n"),
        (20, 7, 0.02, "small_20f_7n"),
        (20, 8, 0.02, "small_20f_8n"),
        # fixed n=5, varying F (probe alpha scaling)
        (5, 5, 0.02, "fix5n_5f"),     # trivial
        (10, 5, 0.02, "fix5n_10f"),
        (15, 5, 0.02, "fix5n_15f"),
        (20, 5, 0.02, "fix5n_20f"),   # already have as small_20f_5n
        (31, 5, 0.02, "fix5n_31f"),   # same as oct_31f_5n
        (50, 5, 0.02, "fix5n_50f"),   # already have
        (100, 5, 0.02, "fix5n_100f"),
    ]

    for F, n, p, tag in configs:
        path = WEIGHTS_DIR / f"{tag}_L4.pt"
        if path.exists():
            print(f"[skip] {path}")
            continue
        print(f"Training {tag} (F={F}, n={n}, p={p})")
        model = train_model(F, n, p, loss_exp=4, n_batches=10000)
        torch.save(model.state_dict(), path)
        # quick eval
        model.eval()
        with torch.no_grad():
            per_feat = torch.zeros(F, device=DEVICE)
            count = torch.zeros(F, device=DEVICE)
            for _ in range(50):
                x, y = generate_batch(2048, F, p)
                active = (x > 0).float()
                err = ((model(x) - y) ** 2 * active).sum(dim=0)
                per_feat += err
                count += active.sum(dim=0)
            per_feat = (per_feat / count.clamp(min=1)).cpu().numpy()
        print(f"  mean per-feat MSE = {per_feat.mean():.5f}, "
              f"min = {per_feat.min():.5f}, max = {per_feat.max():.5f}")
    print("Done")
