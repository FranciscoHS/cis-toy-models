"""
Measure empirical test MSE and error decomposition on trained L4 models,
directly on multi-feature inputs, over a grid of sampling distributions.

For each model (F, n) and sampling scheme:
  - Bernoulli(p) for p in a list
  - fixed exactly-k active features (k in a list)
compute per-output error decomposition:
  * E[(y_hat - y)^2] averaged over all outputs
  * E[(y_hat - y)^2 | feature active]  (on-signal error)
  * E[y_hat^2 | feature inactive]       (cross-talk energy)
  * P(active) for normalisation sanity

Also reports alpha := F/n * mean(diag R) with R the single-feature
response matrix, so we can combine measured alpha with the closed-form
cross-talk prediction   cross_talk_per_output ~ p * alpha^2 / r.

Writes JSON to data/scaling_mse.json.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, n_features, n_neurons):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(n_neurons, n_features) * 0.01)
        self.W_out = nn.Parameter(torch.randn(n_features, n_neurons) * 0.01)

    def forward(self, x):
        return (self.W_out @ torch.relu(self.W_in @ x.T)).T


def load_model(path: Path, F: int, n: int, device) -> SimpleMLP:
    m = SimpleMLP(F, n).to(device)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


def sample_bernoulli(batch, F, p, device):
    mask = (torch.rand(batch, F, device=device) < p).float()
    values = torch.rand(batch, F, device=device) * 2 - 1
    return mask * values


def sample_fixed_k(batch, F, k, device):
    """Exactly k features active uniformly at random, magnitudes Uniform(-1,1)."""
    # for each row, choose k indices without replacement
    rand = torch.rand(batch, F, device=device)
    topk = rand.topk(k, dim=1).indices  # (batch, k)
    mask = torch.zeros(batch, F, device=device)
    mask.scatter_(1, topk, 1.0)
    values = torch.rand(batch, F, device=device) * 2 - 1
    return mask * values


def compute_alpha_and_R(model, device):
    """R[:, j] := W_out @ ReLU(W_in[:, j])."""
    W_in = model.W_in.detach()
    W_out = model.W_out.detach()
    R = W_out @ torch.relu(W_in)  # (F, F)
    F = R.shape[0]
    # alpha from diag
    diag = R.diagonal()
    return R.cpu().numpy(), diag.mean().item(), diag.std().item()


@torch.no_grad()
def eval_model(model, F, sampler, n_batches=200, batch=2048, device="cpu"):
    """Return dict of scalar metrics averaged over n_batches*batch samples."""
    r2_all = 0.0
    n_all = 0
    r2_active = 0.0
    n_active = 0
    r2_inactive = 0.0
    n_inactive = 0
    yhat2_active = 0.0
    yhat2_inactive = 0.0
    # also direct error on active feature magnitudes
    signed_err_active = 0.0
    abs_err_active = 0.0
    y2_active = 0.0
    total_l2 = 0.0
    total_l4 = 0.0
    # per-sample sum of residual energy (||yhat - y||^2)
    sample_err_energy = 0.0
    sample_crosstalk_energy = 0.0  # sum over inactive outputs of yhat^2 per sample
    sample_signal_energy = 0.0     # sum over active outputs of (yhat - y)^2 per sample
    n_samples = 0

    for _ in range(n_batches):
        x = sampler(batch, F)
        y = torch.relu(x)
        yhat = model(x)
        r = yhat - y
        r2 = r * r

        active = (x > 0)
        inactive = ~active
        active_f = active.float()
        inactive_f = inactive.float()

        r2_all += r2.sum().item()
        n_all += r2.numel()
        r2_a = (r2 * active_f).sum().item()
        r2_i = (r2 * inactive_f).sum().item()
        r2_active += r2_a
        n_active += active_f.sum().item()
        r2_inactive += r2_i
        n_inactive += inactive_f.sum().item()

        yhat2 = yhat * yhat
        yhat2_active += (yhat2 * active_f).sum().item()
        yhat2_inactive += (yhat2 * inactive_f).sum().item()

        signed_err_active += (r * active_f).sum().item()
        abs_err_active += (r.abs() * active_f).sum().item()
        y2_active += ((y * y) * active_f).sum().item()

        total_l2 += r2.sum().item()
        total_l4 += (r2 * r2).sum().item()

        # per-sample
        sample_err_energy += r2.sum(dim=1).sum().item()
        sample_crosstalk_energy += (r2 * inactive_f).sum(dim=1).sum().item()
        sample_signal_energy += (r2 * active_f).sum(dim=1).sum().item()
        n_samples += batch

    eps = 1e-30
    return {
        # per-entry means
        "mse_all": r2_all / max(n_all, 1),
        "mse_active": r2_active / max(n_active, 1),
        "mse_inactive": r2_inactive / max(n_inactive, 1),
        "yhat2_active_mean": yhat2_active / max(n_active, 1),
        "yhat2_inactive_mean": yhat2_inactive / max(n_inactive, 1),
        "p_active_empirical": n_active / max(n_all, 1),
        # per-sample energies (sums over F)
        "err_energy_per_sample": sample_err_energy / max(n_samples, 1),
        "crosstalk_energy_per_sample": sample_crosstalk_energy / max(n_samples, 1),
        "signal_energy_per_sample": sample_signal_energy / max(n_samples, 1),
        # per-output cross-talk
        "crosstalk_per_output": sample_crosstalk_energy / (max(n_samples, 1) * F),
        # target energy for normalization
        "y2_active_mean": y2_active / max(n_active, 1),
        # loss proxies (sum over outputs, mean over samples and outputs)
        "L4": total_l4 / max(n_all, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--out", default="data/scaling_mse.json")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Which configs to run, as F,n,tag (tag maps to weights/{tag}.pt)")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights_dir = Path(args.weights_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # default: existing r=10 series + some extras
    if args.configs is None:
        configs = [
            # (F, n, tag)
            (10, 1, "small_10f_1n_L4"),
            (20, 2, "small_20f_2n_L4"),
            (50, 5, "small_50f_5n_L4"),
            (100, 10, "small_100f_10n_L4"),
            (200, 20, "small_200f_20n_L4"),
            (500, 50, "small_500f_50n_L4"),
            (1000, 100, "small_1000f_100n_L4"),
        ]
    else:
        configs = []
        for c in args.configs:
            F, n, tag = c.split(",")
            configs.append((int(F), int(n), tag))

    # sampling schemes. Training used p=0.02.
    p_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    k_list = [1, 2, 5, 10]

    results = []
    for F, n, tag in configs:
        path = weights_dir / f"{tag}.pt"
        if not path.exists():
            print(f"[skip missing] {path}")
            continue
        print(f"=== {tag}  (F={F}, n={n}, r={F/n:.1f}) ===")
        m = load_model(path, F, n, device)
        R_np, diag_mean, diag_std = compute_alpha_and_R(m, device)
        alpha = diag_mean * F / n
        print(f"  alpha = {alpha:.4f}, diag_mean = {diag_mean:.4f}, diag_std = {diag_std:.4f}")

        entry = {
            "tag": tag,
            "F": F,
            "n": n,
            "ratio": F / n,
            "alpha": alpha,
            "diag_mean": diag_mean,
            "diag_std": diag_std,
            "p_sweep": {},
            "k_sweep": {},
        }
        for p in p_list:
            if p * F < 0.05:
                continue
            sampler = lambda b, F_, _p=p: sample_bernoulli(b, F_, _p, device)
            stats = eval_model(m, F, sampler, n_batches=args.batches, batch=args.batch_size, device=device)
            entry["p_sweep"][f"{p}"] = stats
            print(f"  p={p}: crosstalk/output = {stats['crosstalk_per_output']:.5f}, "
                  f"mse_active={stats['mse_active']:.5f}, p_active~{stats['p_active_empirical']:.4f}")
        for k in k_list:
            if k > F:
                continue
            sampler = lambda b, F_, _k=k: sample_fixed_k(b, F_, _k, device)
            stats = eval_model(m, F, sampler, n_batches=args.batches, batch=args.batch_size, device=device)
            entry["k_sweep"][f"{k}"] = stats
            print(f"  k={k}: crosstalk/output = {stats['crosstalk_per_output']:.5f}, "
                  f"mse_active={stats['mse_active']:.5f}, p_active~{stats['p_active_empirical']:.4f}")
        results.append(entry)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
