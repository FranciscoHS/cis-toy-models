"""
Train L4 series where p is chosen so k := pF is fixed across (F, n) at
fixed ratio r = F/n. Lets us test per-output error scaling at fixed
co-activation count.

k=2 is the primary setting; we also do k=5 at r=10 for cross-check.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


class SimpleMLP(nn.Module):
    def __init__(self, F, n):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n, F) * 0.01)
        self.W_out = nn.Parameter(torch.randn(F, n) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


def generate_batch(batch, F, p, device=DEVICE):
    mask = (torch.rand(batch, F, device=device) < p).float()
    values = torch.rand(batch, F, device=device) * 2 - 1
    x = mask * values
    return x, torch.relu(x)


def train_model(F, n, p, loss_exp=4, n_batches=10000, seed=0, batch=2048, lr=0.003):
    torch.manual_seed(seed)
    model = SimpleMLP(F, n).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_batches)
    for step in range(n_batches):
        x, y = generate_batch(batch, F, p)
        loss = (torch.abs(model(x) - y) ** loss_exp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def quick_eval(model, F, p, n_batches=50):
    model.eval()
    per_feat = torch.zeros(F, device=DEVICE)
    count = torch.zeros(F, device=DEVICE)
    for _ in range(n_batches):
        x, y = generate_batch(2048, F, p)
        active = (x > 0).float()
        err = ((model(x) - y) ** 2 * active).sum(dim=0)
        per_feat += err
        count += active.sum(dim=0)
    return (per_feat / count.clamp(min=1)).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10000)
    ap.add_argument("--ks", nargs="+", type=float, default=[2.0])
    args = ap.parse_args()

    # At r=10: F = 10n, p = k/F
    r10_configs = [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), (1000, 100)]
    # r=30
    r30_configs = [(30, 1), (60, 2), (150, 5), (300, 10), (600, 20)]
    # r=100
    r100_configs = [(100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20)]

    all_configs = r10_configs + r30_configs + r100_configs

    for k in args.ks:
        for (F, n) in all_configs:
            p = k / F
            if p >= 1.0:
                print(f"[skip] k={k} too large for F={F}")
                continue
            tag = f"fixk{int(k) if k == int(k) else k}_{F}f_{n}n_L4"
            path = WEIGHTS_DIR / f"{tag}.pt"
            if path.exists():
                print(f"[skip] {path}")
                continue
            print(f"Training {tag} (F={F}, n={n}, r={F//n}, p={p:.4f}, k={k})")
            model = train_model(F, n, p, 4, n_batches=args.batches)
            torch.save(model.state_dict(), path)
            per_feat = quick_eval(model, F, p)
            print(f"  mean per-feat MSE = {per_feat.mean():.5f}, "
                  f"min = {per_feat.min():.5f}, max = {per_feat.max():.5f}, std={per_feat.std():.5f}")
    print("Done")
