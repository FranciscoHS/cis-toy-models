"""
Replace the linear W_out with an MLP decoder. Keep W_in linear + ReLU bottleneck.
Train on L4. Measure loss decomposition by number of active features k.

Prediction:
 - n=2, F=20: k=1 loss plateaus > 0 (codeword ambiguity, 3 codewords for 20 features).
 - n=5, F=20: k=1 loss approaches 0 (codewords nearly unique, MLP can decode).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from small_models import generate_batch, DEVICE

F = 20
P = 0.02
BATCH = 2048
STEPS = 20_000
LR = 3e-3

CACHE = Path("weights")


class BottleneckMLP(nn.Module):
    def __init__(self, n_features, n_neurons, hidden_dim=128, depth=3):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        layers = []
        prev = n_neurons
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_features))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = torch.relu(self.W_in @ x.T).T   # (batch, n_neurons)
        return self.decoder(z)


def train(n_neurons, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = BottleneckMLP(F, n_neurons).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        x, t = generate_batch(BATCH, F, P)
        loss = (torch.abs(m(x) - t) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def decompose(model, n_samples=400_000):
    torch.manual_seed(9999)
    x, t = generate_batch(n_samples, F, P)
    with torch.no_grad():
        err4 = ((model(x) - t) ** 4).mean(dim=1)
    k = (x != 0).sum(dim=1).cpu().numpy()
    err4 = err4.cpu().numpy()

    out = {}
    total = err4.mean()
    for kv in range(0, min(6, int(k.max())) + 1):
        mask = k == kv
        if mask.sum() == 0:
            continue
        p_k = mask.mean()
        mean_loss = err4[mask].mean()
        contribution = p_k * mean_loss
        out[kv] = dict(p_k=float(p_k), mean_loss=float(mean_loss),
                       contribution=float(contribution), frac=float(contribution/total))
    out["total"] = float(total)
    return out


def load_linear_decomp(n):
    """Load corresponding linear-decoder (trained earlier) decomposition for comparison."""
    from small_models import SimpleMLP
    m = SimpleMLP(F, n).to(DEVICE)
    m.load_state_dict(torch.load(CACHE / f"small_20f_{n}n_L4.pt",
                                 map_location=DEVICE, weights_only=True))
    return decompose(m)


if __name__ == "__main__":
    print(f"Device: {DEVICE}  F={F}  p={P}  steps={STEPS}")
    for n in [2, 3, 4, 5]:
        print(f"\n=== n={n} (MLP decoder) ===")
        mlp_path = CACHE / f"mlp_decoder_20f_{n}n_L4.pt"
        if mlp_path.exists():
            m = BottleneckMLP(F, n).to(DEVICE)
            m.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
            print(f"  loaded {mlp_path}")
        else:
            m = train(n)
            torch.save(m.state_dict(), mlp_path)
            print(f"  trained and saved {mlp_path}")
        d_mlp = decompose(m)

        print(f"  MLP-decoder L4 total: {d_mlp['total']:.4e}")
        d_lin = load_linear_decomp(n)
        print(f"  linear-decoder L4 total (ref): {d_lin['total']:.4e}")
        print(f"  {'k':>3} {'P(k)':>7} {'L|k lin':>11} {'L|k MLP':>11} {'ratio MLP/lin':>14}")
        for kv in sorted(k for k in d_mlp.keys() if k != "total"):
            p_k = d_mlp[kv]['p_k']
            l_m = d_mlp[kv]['mean_loss']
            l_l = d_lin.get(kv, {}).get('mean_loss', float('nan'))
            ratio = l_m / l_l if l_l > 0 else float('nan')
            print(f"  {kv:>3} {p_k:>7.4f} {l_l:>11.4e} {l_m:>11.4e} {ratio:>14.3f}")
