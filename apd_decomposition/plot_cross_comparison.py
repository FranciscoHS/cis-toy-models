"""Produce comparative plots across the APD runs: category-count bar chart,
mono-feature coverage, and a side-by-side of scrubbing histograms.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summaries(run_dir_root: Path, names):
    out = {}
    for n in names:
        p = run_dir_root / n / "summary.json"
        if p.exists():
            with open(p) as f:
                out[n] = json.load(f)
    return out


if __name__ == "__main__":
    root = Path("apd_decomposition/out")
    names = ["plain_20f_2n", "plain_20f_5n", "plain_100f_10n",
             "embed_20f_2n_D40", "embed_20f_5n_D80"]
    s = load_summaries(root, names)
    if len(s) == 0:
        print("No summaries found.")
        raise SystemExit(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # panel 1: component category counts per run
    cats = ["dead", "mono", "duo", "poly"]
    colors = ["lightgray", "tab:green", "tab:orange", "tab:red"]
    xs = np.arange(len(s))
    bottoms = np.zeros(len(s))
    labels = list(s.keys())
    for ci, c in enumerate(cats):
        vals = np.array([s[n][c] for n in labels])
        axes[0].bar(xs, vals, bottom=bottoms, label=c, color=colors[ci])
        bottoms += vals
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("component count")
    axes[0].set_title("APD components: dead / mono / duo / poly")
    axes[0].legend()

    # panel 2: mono-feature coverage vs F
    F_vals = [s[n]["F"] for n in labels]
    covered = [s[n]["mono_features_covered"] for n in labels]
    axes[1].bar(xs - 0.2, F_vals, width=0.4, label="F (total features)", color="tab:blue")
    axes[1].bar(xs + 0.2, covered, width=0.4, label="features covered by some mono component",
                color="tab:green")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("count")
    axes[1].set_title("Feature coverage: F vs monosemantic-component coverage")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("apd_decomposition/cross_comparison.png", dpi=150)
    print("Saved apd_decomposition/cross_comparison.png")

    # Also a table summary
    print("\nname                   F   C  dead mono duo poly  features_covered  mse_full    anti/scrub")
    print("-" * 100)
    for n in labels:
        d = s[n]
        print(f"{n:<22} {d['F']:>3} {d['C']:>3}  {d['dead']:>4} {d['mono']:>4} {d['duo']:>3} {d['poly']:>4}  "
              f"{d['mono_features_covered']:>14}/{d['F']:<3}  {d['mse_all_components']:.2e}   {d['mse_ratio_anti_over_scrub']:.1f}")
