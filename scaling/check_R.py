"""Measure R structure: diag mean/std, mean squared off-diag, vs Welch bound.

For each model, compute rho := (mean off-diag²)/(α² Welch²).  rho = 1 means
'exactly Welch-level interference', rho < 1 means less interference than
Welch predicts (good), rho > 1 means more (bad).
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, F, n):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n, F) * 0.01)
        self.W_out = nn.Parameter(torch.randn(F, n) * 0.01)


def measure(path, F, n):
    m = SimpleMLP(F, n)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    W_in = m.W_in.detach()
    W_out = m.W_out.detach()
    R = (W_out @ torch.relu(W_in)).numpy()
    diag = R.diagonal()
    mean_diag = diag.mean()
    std_diag = diag.std()
    alpha = mean_diag * F / n
    off = R - (R * 0 + 1e0) * 0  # dummy
    off = R.copy()
    off[range(F), range(F)] = 0.0
    off2_mean = (off * off).sum() / (F * F - F)  # mean of off-diag squared
    welch2 = n * (F - n) / (F * F * (F - 1))
    rho = off2_mean / (alpha * alpha * welch2)
    # symmetry
    asymm = abs(R - R.T).sum() / abs(R).sum()
    return dict(F=F, n=n, r=F//n, alpha=alpha, mean_diag=mean_diag, std_diag=std_diag,
                off2_mean=off2_mean, alpha_welch2=alpha*alpha*welch2,
                rho=rho, asymm=asymm)


if __name__ == "__main__":
    # r=10, 30, 100 series + fixk1 series
    configs = []
    for (F, n) in [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50),
                   (1000, 100), (2000, 200), (5000, 500)]:
        configs.append((F, n, f"small_{F}f_{n}n_L4"))
    for (F, n) in [(30, 1), (60, 2), (150, 5), (300, 10), (600, 20)]:
        configs.append((F, n, f"small_{F}f_{n}n_L4"))
    for (F, n) in [(100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20), (5000, 50)]:
        configs.append((F, n, f"small_{F}f_{n}n_L4"))
    # fixk1
    for (F, n) in [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), (1000, 100),
                   (30, 1), (60, 2), (150, 5), (300, 10), (600, 20),
                   (100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20)]:
        configs.append((F, n, f"fixk1_{F}f_{n}n_L4"))

    out = []
    print(f"{'tag':30s} {'F':>5} {'n':>4} {'r':>3} {'α':>7} {'off²_mean':>10} {'α²·Welch²':>10} {'ρ':>5} {'asym':>6}")
    for F, n, tag in configs:
        path = Path("weights") / f"{tag}.pt"
        if not path.exists():
            continue
        res = measure(path, F, n)
        res["tag"] = tag
        out.append(res)
        print(f"{tag:30s} {F:5d} {n:4d} {res['r']:3d} {res['alpha']:7.3f} "
              f"{res['off2_mean']:10.2e} {res['alpha_welch2']:10.2e} {res['rho']:5.2f} {res['asymm']:6.3f}")
    Path("data").mkdir(exist_ok=True)
    # cast floats to native python
    out_py = [{k: (float(v) if hasattr(v, "item") else v) for k, v in d.items()} for d in out]
    with open("data/R_structure.json", "w") as f:
        json.dump(out_py, f, indent=2)
    print(f"\nWrote data/R_structure.json")
