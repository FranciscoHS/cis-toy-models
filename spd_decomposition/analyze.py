"""Analyze an SPD decomposition run.

For each parameter component c, we compute:
  - causal importance (ci) on a one-hot input e_j, for each j. This is the
    value the learned gate produces; it's the "is this component active for
    feature j?" metric.
  - per-component alive_both (nonzero ci on both mlp_in and mlp_out side
    simultaneously, with the same dominant feature)
  - scrubbing / anti-scrubbing MSE (same as the APD analysis)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "spd_new"))

from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances

from spd_decomposition.models import PlainCiSModel, EmbeddedCiSModel


def load_run(run_dir: Path):
    """Load the target and SPD models for a run."""
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)

    if meta["variant"] == "plain":
        target = PlainCiSModel.from_our_weights(
            meta["weights"], meta["n_features"], meta["d_mlp"], device="cpu"
        )
    else:
        target = EmbeddedCiSModel.from_our_weights(meta["weights"], device="cpu")
    target.eval()

    # Rebuild the component model
    C = meta["C"]
    spd = ComponentModel(
        base_model=target,
        target_module_patterns=["mlp_in", "mlp_out"],
        C=C,
        n_ci_mlp_neurons=meta["n_ci_mlp_neurons"],
        pretrained_model_output_attr=None,
    )
    # Latest checkpoint
    ckpt = sorted(run_dir.glob("model_*.pth"))[-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    spd.load_state_dict(state)
    spd.eval()
    return target, spd, meta


def compute_ci_one_hot(target, spd, meta):
    """For each feature index j, use x = e_j as input, forward through the target
    model with cache hooks, and compute the causal importance for each component
    at each layer.
    Returns a dict: {module_name: np.ndarray shape (F, C)} where entry (j, c)
    is the ci for component c when input is e_j.
    """
    F_ = meta["n_features"]
    components = {
        k.removeprefix("components.").replace("-", "."): v
        for k, v in spd.components.items()
    }
    gates = {
        k.removeprefix("gates.").replace("-", "."): v
        for k, v in spd.gates.items()
    }

    device = next(target.parameters()).device
    x = torch.eye(F_, device=device)
    target_out, pre_weight_acts = spd.forward_with_pre_forward_cache_hooks(
        x, module_names=list(components.keys())
    )
    As = {name: c.A for name, c in components.items()}
    ci, _ = calc_causal_importances(pre_weight_acts=pre_weight_acts, As=As, gates=gates)
    return {k: v.detach().cpu().numpy() for k, v in ci.items()}


def classify_components(ci_mat: np.ndarray, dead_thresh: float = 0.05,
                        mono_ratio: float = 0.3):
    """Per-component classification from a (F, C) ci matrix.

    A component is:
      - dead if its max over features is < dead_thresh
      - mono if exactly 1 feature has ci > mono_ratio * max
      - duo if 2
      - poly otherwise.
    """
    F_, C = ci_mat.shape
    max_per_c = ci_mat.max(axis=0)
    is_dead = max_per_c < dead_thresh
    ratios = ci_mat / np.maximum(max_per_c[None, :], 1e-12)
    n_active = (ratios > mono_ratio).sum(axis=0)
    cats = np.zeros(C, dtype=int)
    cats[is_dead] = 0
    cats[(~is_dead) & (n_active == 1)] = 1
    cats[(~is_dead) & (n_active == 2)] = 2
    cats[(~is_dead) & (n_active >= 3)] = 3
    return cats, n_active, max_per_c


def scrubbing_experiment(target, spd, meta, ci_in, ci_out, n_samples=2000, device="cpu"):
    """For each sample, identify "relevant" components (those whose dom feature
    is active). Compare two forward passes:
      - scrub: keep relevant components on (mask = ci), ablate others (mask=0)
      - anti-scrub: opposite
    Return MSE vs target for each.
    """
    F_ = meta["n_features"]
    p = meta["p"]
    C = meta["C"]

    # Pick dominant feature per component based on average of in and out ci
    dom_in = ci_in.argmax(axis=0)        # (C,) feature-ids
    dom_out = ci_out.argmax(axis=0)

    torch.manual_seed(123)
    x = torch.zeros(n_samples, F_, device=device)
    mask = torch.rand(n_samples, F_, device=device) < p
    x[mask] = torch.rand(int(mask.sum()), device=device) * 2 - 1
    active = (x != 0)  # (B, F)

    # For each component, is its dominant feature active in this sample?
    # Use union of in-side and out-side dominant features: component is relevant
    # if EITHER its in-dom or out-dom feature is active.
    comp_in_active = active[:, dom_in]     # (B, C)
    comp_out_active = active[:, dom_out]   # (B, C)
    comp_relevant = comp_in_active | comp_out_active

    components = {
        k.removeprefix("components.").replace("-", "."): v
        for k, v in spd.components.items()
    }
    gates = {
        k.removeprefix("gates.").replace("-", "."): v
        for k, v in spd.gates.items()
    }

    with torch.no_grad():
        y_target = target(x)
        # compute ci at actual inputs
        _, pre_acts = spd.forward_with_pre_forward_cache_hooks(
            x, module_names=list(components.keys())
        )
        As = {n: c.A for n, c in components.items()}
        actual_ci, _ = calc_causal_importances(
            pre_weight_acts=pre_acts, As=As, gates=gates
        )
        # masks: keep_active (scrub) or drop_active (anti-scrub)
        mse_results = {}
        for name, keep in [("scrub", True), ("antiscrub", False)]:
            masks = {}
            for module in components:
                ci_tensor = actual_ci[module]   # (B, C)
                if keep:
                    mask_tensor = ci_tensor * comp_relevant.float()
                else:
                    mask_tensor = ci_tensor * (~comp_relevant).float()
                masks[module] = mask_tensor
            y_spd = spd.forward_with_components(x, components=components, masks=masks)
            mse_results[name] = ((y_spd - y_target) ** 2).mean(dim=-1).cpu().numpy()

    return mse_results


def plot_ci_heatmaps(ci_in, ci_out, name: str, out_path: Path, cats_in, cats_out):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Sort components by their dominant feature
    def sort_by_feat(mat):
        max_per_c = mat.max(axis=0)
        alive = max_per_c > 0.05
        dom = mat.argmax(axis=0)
        dom[~alive] = mat.shape[0]  # send dead to the end
        order = np.argsort(dom)
        return mat[:, order]

    for ax, mat, title in [(axes[0], ci_in, f"{name}\nIN-side causal importance (f × c)"),
                           (axes[1], ci_out, "OUT-side causal importance (f × c)")]:
        X = sort_by_feat(mat)
        vmax = max(X.max(), 0.1)
        im = ax.imshow(X.T, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xlabel("feature index")
        ax.set_ylabel("component (sorted by dom feature)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.7)

    cat_names = ["dead", "mono", "duo", "poly"]
    xs = np.arange(4)
    width = 0.35
    in_counts = [(cats_in == k).sum() for k in range(4)]
    out_counts = [(cats_out == k).sum() for k in range(4)]
    axes[2].bar(xs - width/2, in_counts, width, label="in-side", color="tab:blue")
    axes[2].bar(xs + width/2, out_counts, width, label="out-side", color="tab:orange")
    axes[2].set_xticks(xs); axes[2].set_xticklabels(cat_names)
    axes[2].set_ylabel("count"); axes[2].legend()
    axes[2].set_title(f"Component categories (C={len(cats_in)})")
    for i, c in enumerate(in_counts):
        axes[2].text(i - width/2, c + 0.3, str(c), ha="center", va="bottom", fontsize=8)
    for i, c in enumerate(out_counts):
        axes[2].text(i + width/2, c + 0.3, str(c), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scrubbing(mse_results, name: str, out_path: Path, mse_full: float):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(-8, 1, 60)
    ax.hist(np.clip(mse_results["scrub"], 1e-8, 10), bins=bins, alpha=0.6,
            color="tab:pink",
            label=f"scrubbed (keep dom-active)  mean={np.mean(mse_results['scrub']):.1e}")
    ax.hist(np.clip(mse_results["antiscrub"], 1e-8, 10), bins=bins, alpha=0.6,
            color="tab:green",
            label=f"anti-scrubbed (drop dom-active)  mean={np.mean(mse_results['antiscrub']):.1e}")
    ax.axvline(mse_full, color="black", linestyle="--",
               label=f"full reconstruction  mean={mse_full:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("per-sample MSE vs target")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: scrubbing experiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_run(run_dir: Path):
    target, spd, meta = load_run(run_dir)
    F_ = meta["n_features"]
    C = meta["C"]

    ci = compute_ci_one_hot(target, spd, meta)
    ci_in = ci["mlp_in"]    # (F, C)
    ci_out = ci["mlp_out"]

    cats_in, n_act_in, max_in = classify_components(ci_in)
    cats_out, n_act_out, max_out = classify_components(ci_out)

    # Joint: both in and out alive (max > thresh) and with same dominant feature
    alive_in = max_in > 0.05
    alive_out = max_out > 0.05
    alive_both = alive_in & alive_out
    dom_in = ci_in.argmax(axis=0)
    dom_out = ci_out.argmax(axis=0)
    aligned_both = alive_both & (dom_in == dom_out)

    # Joint mono: aligned both + mono on both sides
    joint_mono_mask = aligned_both & (cats_in == 1) & (cats_out == 1)
    joint_mono_feats = dom_in[joint_mono_mask]
    n_joint_mono = joint_mono_mask.sum().item()
    joint_features_covered = len(set(joint_mono_feats.tolist()))

    # Full reconstruction MSE
    torch.manual_seed(77)
    n_samples = 2000
    x = torch.zeros(n_samples, F_)
    mask = torch.rand(n_samples, F_) < meta["p"]
    x[mask] = torch.rand(int(mask.sum())) * 2 - 1
    with torch.no_grad():
        y_target = target(x)
        # components active with full ci
        components = {
            k.removeprefix("components.").replace("-", "."): v
            for k, v in spd.components.items()
        }
        gates = {
            k.removeprefix("gates.").replace("-", "."): v
            for k, v in spd.gates.items()
        }
        _, pre_acts = spd.forward_with_pre_forward_cache_hooks(
            x, module_names=list(components.keys())
        )
        As = {n: c.A for n, c in components.items()}
        actual_ci, _ = calc_causal_importances(
            pre_weight_acts=pre_acts, As=As, gates=gates
        )
        y_spd_full = spd.forward_with_components(
            x, components=components, masks=actual_ci
        )
        mse_full = ((y_spd_full - y_target) ** 2).mean().item()

    mse_res = scrubbing_experiment(target, spd, meta, ci_in, ci_out)
    mean_scrub = float(np.mean(mse_res["scrub"]))
    mean_anti = float(np.mean(mse_res["antiscrub"]))
    anti_over_scrub = mean_anti / max(mean_scrub, 1e-20)

    summary = dict(
        name=run_dir.name, F=F_, C=C,
        in_dead=(cats_in == 0).sum().item(),
        in_mono=(cats_in == 1).sum().item(),
        in_duo=(cats_in == 2).sum().item(),
        in_poly=(cats_in == 3).sum().item(),
        out_dead=(cats_out == 0).sum().item(),
        out_mono=(cats_out == 1).sum().item(),
        out_duo=(cats_out == 2).sum().item(),
        out_poly=(cats_out == 3).sum().item(),
        alive_both=int(alive_both.sum()),
        aligned_both=int(aligned_both.sum()),
        joint_mono=n_joint_mono,
        joint_features_covered=joint_features_covered,
        mse_full=mse_full,
        mse_scrub=mean_scrub,
        mse_antiscrub=mean_anti,
        anti_over_scrub=anti_over_scrub,
    )
    print(f"\n=== {run_dir.name} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    plot_ci_heatmaps(ci_in, ci_out, run_dir.name,
                     run_dir / "ci_heatmap.png", cats_in, cats_out)
    plot_scrubbing(mse_res, run_dir.name, run_dir / "scrubbing.png", mse_full)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    summaries = [analyze_run(Path(r)) for r in args.runs]
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'name':<22} F   C    alive_b aligned joint_mono features  mse_full  anti/scrub")
    print("-" * 110)
    for s in summaries:
        print(f"{s['name']:<22} {s['F']:<3} {s['C']:<4} {s['alive_both']:<7} "
              f"{s['aligned_both']:<7} {s['joint_mono']:<10} "
              f"{s['joint_features_covered']:<8} "
              f"{s['mse_full']:.2e}  {s['anti_over_scrub']:.2f}")
