"""
Train additional r=10 models at n=200, 500 (F=2000, 5000), to verify
whether alpha saturates as n grows at fixed F/n=10.

We already have n=100 (F=1000). Add n=200 and 500 -- n=1000 would be
F=10000, probably too expensive.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = Path("weights")


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--batches", type=int, default=10000)
    args = ap.parse_args()

    configs = [(2000, 200), (5000, 500)]
    for (F, n) in configs:
        tag = f"small_{F}f_{n}n_L4"
        path = WEIGHTS_DIR / f"{tag}.pt"
        if path.exists():
            print(f"[skip] {path}")
            continue
        print(f"Training {tag} (F={F}, n={n}, r={F//n}, p={args.p})")
        model = train_model(F, n, args.p, 4, n_batches=args.batches)
        torch.save(model.state_dict(), path)
        print(f"  done")
