"""
Compare L2 vs L4 trained models under two MSE metrics:
  (a) unconditional  = E[(y_hat - y)^2], what L2 was trained to minimize.
  (b) conditional on active  = E[(y_hat - y)^2 | m_j = 1], what was plotted earlier.
Prediction: L2 wins (a), L4 wins (b).
"""

import torch
import numpy as np
from pathlib import Path
from small_models import SimpleMLP, generate_batch, DEVICE

WEIGHTS_DIR = Path("weights")
P = 0.02
CONFIGS = [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), (1000, 100)]


def mse_both(model, n_features, n_batches=200, batch_size=2048):
    model.eval()
    uncond_sum = 0.0
    uncond_n = 0
    act_sq = torch.zeros(n_features, device=DEVICE)
    act_n = torch.zeros(n_features, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = generate_batch(batch_size, n_features, P)
            err2 = (model(x) - y) ** 2
            uncond_sum += err2.sum().item()
            uncond_n += err2.numel()
            mask = (x != 0).float()
            act_sq += (err2 * mask).sum(dim=0)
            act_n += mask.sum(dim=0)
    uncond = uncond_sum / uncond_n
    cond = (act_sq / act_n.clamp(min=1)).cpu().numpy().mean()
    return uncond, cond


def load(n_feat, n_neur, loss_exp):
    m = SimpleMLP(n_feat, n_neur).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_DIR / f"small_{n_feat}f_{n_neur}n_L{loss_exp}.pt",
                                 map_location=DEVICE, weights_only=True))
    return m


if __name__ == "__main__":
    print(f"{'F':>5} {'n':>4}  "
          f"{'L2 uncond':>11} {'L4 uncond':>11}  "
          f"{'L2 cond':>9} {'L4 cond':>9}  "
          f"{'uncond winner':>14} {'cond winner':>12}")
    for n_feat, n_neur in CONFIGS:
        m2 = load(n_feat, n_neur, 2)
        m4 = load(n_feat, n_neur, 4)
        u2, c2 = mse_both(m2, n_feat)
        u4, c4 = mse_both(m4, n_feat)
        uwin = "L2" if u2 < u4 else "L4"
        cwin = "L2" if c2 < c4 else "L4"
        print(f"{n_feat:>5} {n_neur:>4}  "
              f"{u2:>11.6f} {u4:>11.6f}  "
              f"{c2:>9.4f} {c4:>9.4f}  "
              f"{uwin:>14} {cwin:>12}")
