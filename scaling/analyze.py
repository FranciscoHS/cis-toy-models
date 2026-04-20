"""
Aggregate measurements across all scaling series into plots + a summary
table.

Inputs (JSON):
  data/scaling_mse.json              (r=10 fixed-p series)
  data/scaling_mse_r30_r100.json     (new ratios)
  data/scaling_mse_fixk.json         (fixed-k trained models)
  data/scaling_mse_r10_large.json    (n=200, 500 at r=10)

Outputs:
  figures/scaling_crosstalk_fixed_p.png   crosstalk/output vs n at p=0.02, r=10/30/100
  figures/scaling_crosstalk_fixed_k.png   crosstalk/output vs n at k=2, r=10/30/100
  figures/scaling_alpha.png               alpha vs n at r=10/30/100
  figures/scaling_collapse.png            crosstalk * n / (k alpha^2) vs r
  data/scaling_summary.json               tabular data
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(paths):
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as f:
            out.extend(json.load(f))
    return out


def by_ratio(data, ratio, tag_predicate=None):
    ents = [d for d in data if abs(d["ratio"] - ratio) < 0.5]
    if tag_predicate is not None:
        ents = [d for d in ents if tag_predicate(d["tag"])]
    return sorted(ents, key=lambda d: d["n"])


def plot_crosstalk_fixed_p(data, p_str, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        entries = by_ratio(data, r, lambda t: t.startswith("small_") and "fixk" not in t)
        ns = [e["n"] for e in entries]
        ct = [e["p_sweep"].get(p_str, {}).get("crosstalk_per_output") for e in entries]
        valid = [(n, c) for n, c in zip(ns, ct) if c is not None]
        if not valid:
            continue
        ns_v, ct_v = zip(*valid)
        ax.plot(ns_v, ct_v, "o-", color=color, label=f"r=F/n={r}")
        # prediction: crosstalk ~ p alpha^2 / r
        alphas = [e["alpha"] for e in entries if e["p_sweep"].get(p_str, {}).get("crosstalk_per_output") is not None]
        p = float(p_str)
        pred = [p * a * a / r for a in alphas]
        ax.plot(ns_v, pred, "--", color=color, alpha=0.4, label=f"r={r} theory p α²/r")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(f"crosstalk energy per output @ p={p_str}")
    ax.set_title(f"Cross-talk vs scale at fixed p={p_str}, fixed r=F/n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def plot_crosstalk_fixed_k(data, k_str, out_path, use_fixk_models=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        if use_fixk_models:
            # use models trained with p = k/F
            entries = by_ratio(data, r, lambda t: t.startswith(f"fixk{k_str}_"))
            key = "p_sweep"
            p_keys = None  # use p_active_empirical ~ k/F
        else:
            entries = by_ratio(data, r, lambda t: t.startswith("small_") and "fixk" not in t)
            key = "k_sweep"
        ns = [e["n"] for e in entries]
        if use_fixk_models:
            # these models' natural p is k/F, measured at that p
            ct = []
            for e in entries:
                F, n = e["F"], e["n"]
                p_natural = float(k_str) / F
                # pick closest p measured
                ct_val = None
                for ps, stats in e.get("p_sweep", {}).items():
                    if abs(float(ps) - p_natural) < 1e-6:
                        ct_val = stats["crosstalk_per_output"]
                        break
                ct.append(ct_val)
        else:
            ct = [e[key].get(k_str, {}).get("crosstalk_per_output") for e in entries]
        valid = [(n, c) for n, c in zip(ns, ct) if c is not None]
        if not valid:
            continue
        ns_v, ct_v = zip(*valid)
        ax.plot(ns_v, ct_v, "o-", color=color, label=f"r={r}")
        # prediction: crosstalk ~ C k alpha^2 / n, check shape vs n
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(f"crosstalk energy per output @ k={k_str}")
    title_src = "trained at p=k/F" if use_fixk_models else "evaluated at k from fixed-p models"
    ax.set_title(f"Cross-talk vs scale at fixed k={k_str} ({title_src})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    # overlay 1/n reference line
    xs = np.array([1, 1000])
    for r, c in [(10, "C0"), (30, "C1"), (100, "C2")]:
        ax.plot(xs, 0.01 / xs, ":", color=c, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def plot_alpha_vs_n(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        entries = by_ratio(data, r, lambda t: t.startswith("small_") and "fixk" not in t)
        if not entries:
            continue
        ns = [e["n"] for e in entries]
        alphas = [e["alpha"] for e in entries]
        ax.plot(ns, alphas, "o-", color=color, label=f"r={r}")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("α = (F/n)·mean(diag R)")
    ax.set_title("Projection scale α vs n, fixed r=F/n, p=0.02")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def plot_collapse(data, out_path):
    """Plot crosstalk * n / (k alpha^2) vs r. If theory holds, this is ~const/r^2."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color, marker in [(10, "C0", "o"), (30, "C1", "s"), (100, "C2", "^")]:
        entries = by_ratio(data, r, lambda t: t.startswith("small_") and "fixk" not in t)
        if not entries:
            continue
        xs = []
        ys = []
        for e in entries:
            alpha = e["alpha"]
            n = e["n"]
            F = e["F"]
            for k_str in ["1", "2", "5", "10"]:
                stats = e.get("k_sweep", {}).get(k_str)
                if stats is None:
                    continue
                k = int(k_str)
                if k >= F:
                    continue
                ct = stats["crosstalk_per_output"]
                # normalize: crosstalk * n / (k alpha^2)
                y = ct * n / (k * alpha * alpha)
                xs.append(n)
                ys.append(y)
        if xs:
            ax.scatter(xs, ys, color=color, marker=marker, s=30, label=f"r={r}", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("crosstalk·n / (k·α²)")
    ax.set_title("Collapse: per-output crosstalk normalised.  Theory: 1/(3r²) dashed.")
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        ax.axhline(1.0 / (3.0 * r * r), color=color, linestyle="--", alpha=0.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def write_summary(data, out_path):
    rows = []
    for e in data:
        if "fixk" in e["tag"]:
            continue
        row = dict(tag=e["tag"], F=e["F"], n=e["n"], r=e["ratio"], alpha=e["alpha"],
                   diag_std=e["diag_std"])
        for p_str, stats in e.get("p_sweep", {}).items():
            row[f"ct_p{p_str}"] = stats["crosstalk_per_output"]
            row[f"mse_active_p{p_str}"] = stats["mse_active"]
        for k_str, stats in e.get("k_sweep", {}).items():
            row[f"ct_k{k_str}"] = stats["crosstalk_per_output"]
            row[f"mse_active_k{k_str}"] = stats["mse_active"]
        rows.append(row)
    rows.sort(key=lambda r: (r["r"], r["n"]))
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sources = [
        "data/scaling_mse.json",
        "data/scaling_mse_r30_r100.json",
        "data/scaling_mse_r10_large.json",
        "data/scaling_mse_fixk.json",
    ]
    data = load(sources)
    print(f"Loaded {len(data)} entries from {len(sources)} files")

    Path("figures").mkdir(exist_ok=True)
    plot_crosstalk_fixed_p(data, "0.02", "figures/scaling_crosstalk_fixed_p.png")
    plot_crosstalk_fixed_k(data, "2", "figures/scaling_crosstalk_fixed_k_eval.png", use_fixk_models=False)
    plot_crosstalk_fixed_k(data, "2", "figures/scaling_crosstalk_fixed_k_trained.png", use_fixk_models=True)
    plot_alpha_vs_n(data, "figures/scaling_alpha.png")
    plot_collapse(data, "figures/scaling_collapse.png")
    write_summary(data, "data/scaling_summary.json")
