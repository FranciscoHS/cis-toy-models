"""
Visuals for the "matched slide": 6-parameter ansatz reproduces the trained
20f/2n L4 model.

Panel A: ansatz loss vs n_A, with trained-model loss as horizontal baseline
         and empirical n_A=6 as vertical reference.
Panel B: trained archetype values vs ansatz best-fit values, as paired dots.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from ansatz_fit import (F, build_W_from_ansatz, fit_ansatz, forward, mc_loss,
                        trained_loss, device)

FIGURES_DIR = Path("figures")
CACHE = Path("figures") / "slide_match_data.json"


def run_sweep():
    if CACHE.exists():
        with open(CACHE) as f:
            d = json.load(f)
        return d

    L_trained, W_in_t, W_out_t = trained_loss()
    signs = np.sign(W_in_t)
    A_idx = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, 1)]
    B_idx = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, -1)]
    trained_params = dict(
        a_A=float(W_in_t[:, A_idx].mean()),
        a_B1=float(W_in_t[0, B_idx].mean()),
        a_B2=float(W_in_t[1, B_idx].mean()),
        o_A=float(W_out_t[A_idx, :].mean()),
        o_B1=float(W_out_t[B_idx, 0].mean()),
        o_B2=float(W_out_t[B_idx, 1].mean()),
    )
    n_A_empirical = len(A_idx)

    sweep = []
    for n_A in range(0, F + 2, 2):
        params, loss = fit_ansatz(n_A)
        sweep.append(dict(n_A=n_A, loss=float(loss),
                          params=[float(p) for p in params]))
        print(f"n_A={n_A:>3}  loss={loss:.6e}  params={np.round(params, 3).tolist()}")

    d = dict(L_trained=float(L_trained), trained_params=trained_params,
             n_A_empirical=n_A_empirical, sweep=sweep)
    with open(CACHE, "w") as f:
        json.dump(d, f, indent=2)
    return d


def panel_a_loss_vs_nA(d):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ns = [r["n_A"] for r in d["sweep"]]
    losses = [r["loss"] for r in d["sweep"]]
    L_trained = d["L_trained"]
    n_A_emp = d["n_A_empirical"]

    ax.plot(ns, losses, "o-", color="C0", markersize=7, lw=1.6,
            label="best 6-parameter ansatz")
    ax.axhline(L_trained, color="black", linestyle="--", lw=1.2,
               label=f"trained model  ({L_trained:.3e})")
    ax.axvline(n_A_emp, color="gray", linestyle=":", lw=1.2,
               label=f"empirical $n_A = {n_A_emp}$")

    best = min(d["sweep"], key=lambda r: r["loss"])
    ax.plot(best["n_A"], best["loss"], marker="*", markersize=16,
            color="C3", zorder=10, linestyle="none",
            label=f"ansatz minimum at $n_A={best['n_A']}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"$n_A$  (number of features in the A group)")
    ax.set_ylabel(r"$L_{\mathrm{mean}}$  ($L^4$ per-entry)")
    ax.set_title("Ansatz loss vs discrete parameter $n_A$")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "slide_match_loss.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


def panel_b_param_match(d):
    best = min(d["sweep"], key=lambda r: r["loss"])
    labels = [r"$a_A$", r"$a_{B,1}$", r"$a_{B,2}$",
              r"$o_A$", r"$o_{B,1}$", r"$o_{B,2}$"]
    keys = ["a_A", "a_B1", "a_B2", "o_A", "o_B1", "o_B2"]
    trained_vals = [d["trained_params"][k] for k in keys]
    ansatz_vals = best["params"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(labels))

    ax.axhline(0, color="black", lw=0.6, alpha=0.7)
    for xi, tv, av in zip(x, trained_vals, ansatz_vals):
        ax.plot([xi, xi], [tv, av], color="gray", lw=0.9, alpha=0.5, zorder=1)
    ax.scatter(x, trained_vals, s=140, color="C0",
               edgecolor="black", linewidth=0.7, zorder=3,
               label="trained model (group mean)")
    ax.scatter(x, ansatz_vals, s=140, color="C3", marker="D",
               edgecolor="black", linewidth=0.7, zorder=3,
               label=f"ansatz best fit ($n_A={best['n_A']}$)")

    for xi, tv, av in zip(x, trained_vals, ansatz_vals):
        ax.annotate(f"{tv:+.3f}", (xi, tv),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=9, color="C0")
        ax.annotate(f"{av:+.3f}", (xi, av),
                    textcoords="offset points", xytext=(10, -14),
                    fontsize=9, color="C3")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("parameter value")
    ax.set_title("Trained archetype values vs 6-parameter ansatz fit")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "slide_match_params.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    d = run_sweep()
    panel_a_loss_vs_nA(d)
    panel_b_param_match(d)
