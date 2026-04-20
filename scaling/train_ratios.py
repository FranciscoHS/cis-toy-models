"""
Train L4 scaling series at F/n = 30 and 100, a few n each, to check
per-output cross-talk scaling with r.

Training recipe matches small_models.py (p=0.02, batch=2048, 10k steps).
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
    ap.add_argument("--series", choices=["r30", "r100", "r30_full", "r100_full", "both"], default="both")
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--batches", type=int, default=10000)
    args = ap.parse_args()

    np.random.seed(42)
    print(f"Device: {DEVICE}")

    if args.series in ("r30", "both"):
        r30_configs = [(30, 1), (60, 2), (150, 5), (300, 10), (600, 20)]
    else:
        r30_configs = []
    if args.series == "r30_full":
        r30_configs = [(30, 1), (60, 2), (150, 5), (300, 10), (600, 20), (1500, 50)]
    if args.series in ("r100", "both"):
        r100_configs = [(100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20)]
    else:
        r100_configs = []
    if args.series == "r100_full":
        r100_configs = [(100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20), (5000, 50)]

    for (F, n) in r30_configs + r100_configs:
        r = F // n
        tag = f"small_{F}f_{n}n_L4"
        path = WEIGHTS_DIR / f"{tag}.pt"
        if path.exists():
            print(f"[skip] {path}")
            continue
        print(f"Training {tag} (F={F}, n={n}, r={r}, p={args.p})")
        model = train_model(F, n, args.p, 4, n_batches=args.batches)
        torch.save(model.state_dict(), path)
        per_feat = quick_eval(model, F, args.p)
        print(f"  mean per-feat MSE = {per_feat.mean():.5f}, "
              f"min = {per_feat.min():.5f}, max = {per_feat.max():.5f}, std={per_feat.std():.5f}")
    print("Done")
