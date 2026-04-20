"""Produce the final set of figures for the scaling report.

Figures:
  scaling_alpha_sat.png           alpha vs n at p=0.02 for r=10/30/100, with large-n points
  scaling_xtalk_fixed_p.png       crosstalk vs n at fixed p, saturation at r=10 shown
  scaling_xtalk_fixed_k.png       crosstalk vs n at fixed k=2 (eval), 1/n line drawn
  scaling_collapse.png            crosstalk·n/(k α²) vs n, theory dashed at 1/(3r²)
  scaling_alpha_vs_predicted.png  observed alpha vs L4-multi-feature predicted alpha
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alpha_theory import predict_alpha


def load_all():
    data = []
    for path in ["data/scaling_mse.json", "data/scaling_mse_r30_r100.json",
                 "data/scaling_mse_r10_large.json", "data/scaling_mse_r100_n50.json",
                 "data/scaling_mse_fixk.json"]:
        p = Path(path)
        if p.exists():
            data.extend(json.load(open(p)))
    return data


def subseries(data, predicate, sort_key=lambda d: d["n"]):
    return sorted([d for d in data if predicate(d)], key=sort_key)


def fig_alpha_saturation(data, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        ents = subseries(data, lambda d, r=r: d["tag"].startswith("small_") and "fixk" not in d["tag"] and abs(d["ratio"]-r) < 0.5)
        ns = [e["n"] for e in ents]
        alphas = [e["alpha"] for e in ents]
        ax.plot(ns, alphas, "o-", color=color, label=f"r=F/n={r}", markersize=6)
        # L4-multi-feature prediction (constant in n at fixed r, p=0.02)
        pred = predict_alpha(0.02, r)
        ax.axhline(pred, color=color, linestyle="--", alpha=0.4, label=f"L4-multi pred (r={r}): α={pred:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("α = (F/n)·mean(diag R)")
    ax.set_title("α vs n at fixed r=F/n, p=0.02 — L4 prediction is n-independent (dashed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def fig_xtalk_fixed_p(data, out):
    p_str = "0.02"
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        ents = subseries(data, lambda d, r=r: d["tag"].startswith("small_") and "fixk" not in d["tag"] and abs(d["ratio"]-r) < 0.5)
        ns = [e["n"] for e in ents]
        xts = [e["p_sweep"].get(p_str, {}).get("crosstalk_per_output") for e in ents]
        valid = [(n, x) for n, x in zip(ns, xts) if x is not None]
        if valid:
            ns_v, xts_v = zip(*valid)
            ax.plot(ns_v, xts_v, "o-", color=color, label=f"r={r}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel("crosstalk energy per output")
    ax.set_title(f"Cross-talk saturates at fixed p={p_str}, fixed r (r=10 plateau visible n≥100)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def fig_xtalk_fixed_k(data, out, k_str="2"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color in [(10, "C0"), (30, "C1"), (100, "C2")]:
        # use fixed-p models, evaluated at k
        ents = subseries(data, lambda d, r=r: d["tag"].startswith("small_") and "fixk" not in d["tag"] and abs(d["ratio"]-r) < 0.5)
        ns = [e["n"] for e in ents]
        xts = [e.get("k_sweep", {}).get(k_str, {}).get("crosstalk_per_output") for e in ents]
        valid = [(n, x) for n, x in zip(ns, xts) if x is not None]
        if valid:
            ns_v, xts_v = zip(*valid)
            ax.plot(ns_v, xts_v, "o-", color=color, label=f"r={r}")
        # 1/n guideline using the first point
        if valid:
            x0 = ns_v[0]; y0 = xts_v[0]
            xs = np.array(sorted(ns_v))
            ax.plot(xs, y0 * x0 / xs, ":", color=color, alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel(f"crosstalk energy per output @ k={k_str}")
    ax.set_title(f"Cross-talk at fixed k={k_str} (eval on fixed-p models).  dotted = 1/n reference")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def fig_collapse(data, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, color, marker in [(10, "C0", "o"), (30, "C1", "s"), (100, "C2", "^")]:
        ents = subseries(data, lambda d, r=r: d["tag"].startswith("small_") and "fixk" not in d["tag"] and abs(d["ratio"]-r) < 0.5)
        xs = []; ys = []
        for e in ents:
            alpha = e["alpha"]; n = e["n"]; F = e["F"]
            for k_str in ["1", "2", "5", "10"]:
                stats = e.get("k_sweep", {}).get(k_str)
                if stats is None: continue
                k = int(k_str)
                if k >= F: continue
                ct = stats["crosstalk_per_output"]
                y = ct * n / (k * alpha * alpha)
                xs.append(n); ys.append(y)
        if xs:
            ax.scatter(xs, ys, color=color, marker=marker, s=30, label=f"r={r}", alpha=0.7)
        # theory line 1/(3r^2)
        ax.axhline(1.0 / (3.0 * r * r), color=color, linestyle="--", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel("crosstalk·n / (k·α²)  (empirical C(r))")
    ax.set_title("Collapse: all (n, k) at a given r plateau at C(r).  dashed = linear-interference 1/(3r²)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def fig_alpha_vs_predicted(data, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {10: "o", 30: "s", 100: "^"}
    for e in data:
        tag = e["tag"]
        r = e["ratio"]
        if int(r) not in markers: continue
        if tag.startswith("small_"):
            p = 0.02; color_kind = "C0"
        elif tag.startswith("fixk"):
            k = e.get("k_trained")
            if k is None: continue
            p = k / e["F"]
            color_kind = {1: "C3", 2: "C4", 5: "C5"}.get(k, "C6")
        else:
            continue
        a_pred = predict_alpha(p, r)
        a_obs = e["alpha"]
        ax.scatter(a_pred, a_obs, marker=markers[int(r)], color=color_kind, s=35, alpha=0.7)
    m = max(a for e in data for a in [e["alpha"]])
    xs = np.linspace(0, m, 100)
    ax.plot(xs, xs, "k--", alpha=0.4)
    ax.set_xlabel("predicted α (L4 multi-feature, linear interference + Welch off-diag)")
    ax.set_ylabel("observed α")
    ax.set_title("α observed vs L4-multi-feature prediction\nblue=fixed-p; red/purple/brown=fixed-k=1/2/5; marker shape=r")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    data = load_all()
    print(f"Loaded {len(data)} entries")
    Path("figures").mkdir(exist_ok=True)
    fig_alpha_saturation(data, "figures/scaling_alpha_sat.png")
    fig_xtalk_fixed_p(data, "figures/scaling_xtalk_fixed_p.png")
    fig_xtalk_fixed_k(data, "figures/scaling_xtalk_fixed_k.png")
    fig_collapse(data, "figures/scaling_collapse_final.png")
    fig_alpha_vs_predicted(data, "figures/scaling_alpha_vs_pred.png")
