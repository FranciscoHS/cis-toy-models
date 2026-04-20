"""Run measurement sweep on fixed-k trained models.

Fixed-k models are at weights/fixk{k}_{F}f_{n}n_L4.pt with training
p = k/F. We evaluate at several p's and k's to compare.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))
from measure_mse import SimpleMLP, load_model, compute_alpha_and_R, eval_model, sample_bernoulli, sample_fixed_k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--out", default="data/scaling_mse_fixk.json")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights_dir = Path(args.weights_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_configs = []
    # r=10
    for (F, n) in [(10, 1), (20, 2), (50, 5), (100, 10), (200, 20), (500, 50), (1000, 100)]:
        all_configs.append((F, n, 10))
    for (F, n) in [(30, 1), (60, 2), (150, 5), (300, 10), (600, 20)]:
        all_configs.append((F, n, 30))
    for (F, n) in [(100, 1), (200, 2), (500, 5), (1000, 10), (2000, 20)]:
        all_configs.append((F, n, 100))

    ks_trained = [1, 2, 5]
    p_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    k_list = [1, 2, 5, 10]

    results = []
    for k_trained in ks_trained:
        for (F, n, r) in all_configs:
            tag = f"fixk{k_trained}_{F}f_{n}n_L4"
            path = weights_dir / f"{tag}.pt"
            if not path.exists():
                print(f"[skip missing] {path}")
                continue
            p_train = k_trained / F
            print(f"=== {tag}  (F={F}, n={n}, r={r}, p_train={p_train:.4f}) ===")
            m = load_model(path, F, n, device)
            R_np, diag_mean, diag_std = compute_alpha_and_R(m, device)
            alpha = diag_mean * F / n
            print(f"  alpha = {alpha:.4f}")

            entry = {
                "tag": tag,
                "F": F,
                "n": n,
                "ratio": r,
                "alpha": alpha,
                "diag_mean": diag_mean,
                "diag_std": diag_std,
                "k_trained": k_trained,
                "p_train": p_train,
                "p_sweep": {},
                "k_sweep": {},
            }
            # always include natural training p
            p_combined = sorted(set(p_list + [p_train]))
            for p in p_combined:
                if p * F < 0.05 or p >= 1:
                    continue
                sampler = lambda b, F_, _p=p: sample_bernoulli(b, F_, _p, device)
                stats = eval_model(m, F, sampler, n_batches=args.batches, batch=2048, device=device)
                entry["p_sweep"][f"{p}"] = stats
            for k in k_list:
                if k >= F:
                    continue
                sampler = lambda b, F_, _k=k: sample_fixed_k(b, F_, _k, device)
                stats = eval_model(m, F, sampler, n_batches=args.batches, batch=2048, device=device)
                entry["k_sweep"][f"{k}"] = stats
            results.append(entry)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path} ({len(results)} entries)")
