"""
Numerical solver and comparison for the multi-feature L4 closed-form prediction of alpha.

Model (linear-interference, Parseval-projection, Welch off-diagonals):
  x ~ Bernoulli(p) · Uniform(-1, 1) per coordinate (independent).
  y = ReLU(x).
  y_hat_i = beta v_i  +  sum_{j in S\i} R_ij v_j      [linear approx]
  R_ij has variance alpha^2 * Welch^2, Welch^2 = n(F-n)/(F^2(F-1)) ~ 1/(rF).

Then L4 = E[(y_hat - y)^4] has the expansion (derived in SCALING_REPORT):
  L4 = (p/10) A  +  (p^2 r / 3) u^2 B  +  (p^2 r^2 / 3) u^4
  where u = beta, A = (u-1)^4 + u^4, B = (u-1)^2 + u^2, and r = F/n.

So the predicted beta = alpha * n/F depends on (p, r) only (not n).

Minimize numerically.
"""
from __future__ import annotations
import numpy as np


def L4(u, p, r):
    A = (u - 1) ** 4 + u ** 4
    B = (u - 1) ** 2 + u ** 2
    return (p / 10.0) * A + (p * p * r / 3.0) * u * u * B + (p * p * r * r / 3.0) * u ** 4


def predict_beta(p, r):
    us = np.linspace(0.001, 1.0, 2000)
    vals = np.array([L4(u, p, r) for u in us])
    i = int(np.argmin(vals))
    return float(us[i])


def predict_alpha(p, r):
    return predict_beta(p, r) * r


if __name__ == "__main__":
    import json
    from pathlib import Path

    data = []
    for path in ["data/scaling_mse.json", "data/scaling_mse_r30_r100.json",
                 "data/scaling_mse_r10_large.json", "data/scaling_mse_r100_n50.json",
                 "data/scaling_mse_fixk.json"]:
        p = Path(path)
        if p.exists():
            data.extend(json.load(open(p)))

    # For each entry: compute predicted alpha at its training p.
    # Fixed-p models were trained at p=0.02. Fixed-k at p = k/F.
    print(f"{'tag':35s} {'F':>4} {'n':>3} {'r':>3} {'p':>8}  {'α_obs':>7} {'α_pred':>7}  {'ratio':>5}")
    for e in data:
        tag = e["tag"]
        F, n, r = e["F"], e["n"], e["ratio"]
        if tag.startswith("small_"):
            p = 0.02
        elif tag.startswith("fixk"):
            # parse k
            k = e.get("k_trained")
            if k is None:
                continue
            p = k / F
        else:
            continue
        a_pred = predict_alpha(p, r)
        a_obs = e["alpha"]
        print(f"{tag:35s} {F:>4d} {n:>3d} {int(r):>3d} {p:.5f}  {a_obs:7.3f} {a_pred:7.3f}  {a_obs/a_pred:5.2f}")
