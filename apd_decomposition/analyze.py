"""Analyze an APD decomposition run.

For each parameter component c, we compute:
  • a vector of per-feature attributions (how strongly component c participates
    in each feature's computation);
  • classify c as 'monosemantic' (one feature), 'dead' (no feature), or
    'polysemantic' (several features);
  • compute reconstruction quality vs target.

Plots:
  • Feature × component attribution heatmap.
  • Bar chart of component categories (dead / mono / poly).
  • 'Neuron contribution' plot matching figure 6 of the APD paper:
    per-feature, show the per-neuron contribution in target vs APD (max over
    components for APD).

Saves figures into the run output directory.
"""
import argparse
import json
from pathlib import Path

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from apd_decomposition.load_our_model import (
    install_no_residual_patch,
    load_plain_model,
    load_embedded_model,
)
from spd.experiments.resid_mlp.models import (
    ResidualMLPSPDModel,
    ResidualMLPSPDConfig,
)


def load_run(run_dir: Path):
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    install_no_residual_patch()

    if meta["variant"] == "plain":
        target_model, _ = load_plain_model(
            meta["weights"], meta["n_features"], meta["d_mlp"], device="cpu"
        )
    else:
        target_model, _ = load_embedded_model(meta["weights"], device="cpu")

    # Load SPD model
    cfg = target_model.config
    spd_cfg = ResidualMLPSPDConfig(
        n_instances=1,
        n_features=cfg.n_features,
        d_embed=cfg.d_embed,
        d_mlp=cfg.d_mlp,
        n_layers=1,
        act_fn_name=cfg.act_fn_name,
        apply_output_act_fn=cfg.apply_output_act_fn,
        in_bias=cfg.in_bias,
        out_bias=cfg.out_bias,
        init_scale=meta["init_scale"],
        C=meta["C"],
        m=meta.get("m"),
    )
    spd = ResidualMLPSPDModel(config=spd_cfg)
    sd = torch.load(run_dir / "spd_model.pt", map_location="cpu", weights_only=True)
    spd.load_state_dict(sd)

    return target_model, spd, meta


def component_contribution_per_feature(target_model, spd):
    """For each component c and feature j, compute the "neuron contribution"
    (W_U @ W_OUT_c ⊙ W_IN_c @ W_E[:, j]) per-neuron, and summarise.
    This is the paper's Fig 6 formula. Returns contribs (F, C, d_mlp).
    """
    cfg = target_model.config
    F = cfg.n_features
    C = spd.C
    d_mlp = cfg.d_mlp

    # target model weights (single instance)
    W_E = target_model.W_E.data[0]             # (F, d_embed)
    W_U = target_model.W_U.data[0]             # (d_embed, F)
    # mlp_in.weight shape (d_embed, d_mlp): forward is einsum("batch di, di do -> batch do")
    W_IN_full = target_model.layers[0].mlp_in.weight.data[0]   # (d_embed, d_mlp)
    W_OUT_full = target_model.layers[0].mlp_out.weight.data[0] # (d_mlp, d_embed)

    # spd component weights: shape (1, C, d_in, d_out) per LinearComponent
    W_IN_c = spd.layers[0].mlp_in.component_weights[0].detach()    # (C, d_embed, d_mlp)
    W_OUT_c = spd.layers[0].mlp_out.component_weights[0].detach()  # (C, d_mlp, d_embed)

    # For each j, compute contrib[j, c, d]:
    #   write_c = W_U[:, j]^T @ W_OUT_c  → (C, d_mlp)   (how much each neuron of c writes to feature j)
    #   read_c  = W_IN_c^T @ W_E[j, :]^T → (C, d_mlp)   (how much each neuron of c reads from feature j)
    #   contrib = write_c ⊙ read_c
    # Vectorize:
    # read[j, c, d] = sum_{di} W_IN_c[c, di, d] * W_E[j, di]
    read = einops.einsum(W_IN_c, W_E, "c di d, j di -> j c d")
    # write[j, c, d] = sum_{do} W_OUT_c[c, d, do] * W_U[do, j]
    write = einops.einsum(W_OUT_c, W_U, "c d do, do j -> j c d")
    contribs = read * write  # (F, C, d_mlp)

    # Summary per (j, c): sum over neurons
    per_feat_per_comp = contribs.sum(dim=-1)   # (F, C)

    # For each component c, find the feature j with the largest |contribution|
    max_vals, max_feat = per_feat_per_comp.abs().max(dim=0)   # (C,), (C,)
    # signed value
    signed_max = per_feat_per_comp[max_feat, torch.arange(C)]
    return contribs, per_feat_per_comp, max_feat, max_vals, signed_max


def component_in_out_per_feature(target_model, spd):
    """Compute per-feature read/write magnitudes for each component INDEPENDENTLY.

    Returns:
      in_mag (F, C):  sum over neurons of |W_IN_c @ W_E[:, j]|
      out_mag (F, C): sum over neurons of |W_U[:, j]^T @ W_OUT_c|
    """
    cfg = target_model.config
    W_E = target_model.W_E.data[0]            # (F, d_embed)
    W_U = target_model.W_U.data[0]            # (d_embed, F)
    cw_in = spd.layers[0].mlp_in.component_weights[0].detach()    # (C, d_embed, d_mlp)
    cw_out = spd.layers[0].mlp_out.component_weights[0].detach()  # (C, d_mlp, d_embed)
    # in_read[j, c, d] = sum_di W_IN_c[c, di, d] * W_E[j, di]
    in_read = einops.einsum(cw_in, W_E, "c di d, j di -> j c d")
    # out_write[j, c, d] = sum_do W_OUT_c[c, d, do] * W_U[do, j]
    out_write = einops.einsum(cw_out, W_U, "c d do, do j -> j c d")
    in_mag = in_read.abs().sum(dim=-1)   # (F, C)
    out_mag = out_write.abs().sum(dim=-1)
    return in_mag, out_mag


def component_classify(per_feat_per_comp: Tensor, cutoff_ratio: float = 0.3,
                       dead_threshold_abs: float = 0.01):
    """Classify each component as dead / mono / poly.
    - Dead if component's max |contribution| is below dead_threshold_abs × the
      global max |contribution|.
    - Otherwise: count features whose |contribution| is above
      cutoff_ratio × the component's own max. Classify by that count.
    """
    F, C = per_feat_per_comp.shape
    abs_ = per_feat_per_comp.abs()
    max_per_c = abs_.max(dim=0).values.clamp_min(1e-12)
    ratios = abs_ / max_per_c
    n_active_per_c = (ratios > cutoff_ratio).sum(dim=0)
    overall_max = abs_.max().item()
    is_dead = max_per_c < dead_threshold_abs * overall_max
    cats = torch.zeros(C, dtype=torch.long)
    cats[is_dead] = 0
    cats[(~is_dead) & (n_active_per_c == 1)] = 1
    cats[(~is_dead) & (n_active_per_c == 2)] = 2
    cats[(~is_dead) & (n_active_per_c >= 3)] = 3
    return cats, n_active_per_c


def plot_contribution_heatmap(per_feat_per_comp: Tensor, ax, title: str,
                              sort_by_feature: bool = True):
    """Heatmap: rows = components, cols = features."""
    F, C = per_feat_per_comp.shape
    X = per_feat_per_comp.detach().cpu().numpy()

    # Sort components by their dominant feature (so the diagonal is visible)
    if sort_by_feature:
        abs_ = np.abs(X)
        dom_feat = abs_.argmax(axis=0)
        order = np.argsort(dom_feat)
        X_sorted = X[:, order]
    else:
        X_sorted = X

    vmax = np.abs(X_sorted).max()
    im = ax.imshow(X_sorted.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel("feature index")
    ax.set_ylabel("component (sorted)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7)
    return im


def scrubbing_experiment(target_model, spd, meta, n_samples: int = 2000,
                         keep_active: bool = True, device: str = "cpu"):
    """Forward passes where for each sample we keep only the components
    corresponding to the currently-active features (keep_active=True, "scrubbed"
    = ablating unrelated components) or only the components NOT corresponding
    to active features (keep_active=False, "anti-scrubbed" = ablating relevant
    components). Report MSE vs target.

    We assign each component a 'dominant feature' (the j maximizing the
    unsigned per-feature contribution).
    """
    with torch.no_grad():
        _, per_feat_per_comp, comp_dom_feat, _, _ = component_contribution_per_feature(
            target_model, spd
        )
        F = target_model.config.n_features
        C = spd.C
        p = meta["p"]

        torch.manual_seed(123)
        x = torch.zeros(n_samples, 1, F, device=device)
        mask = torch.rand(n_samples, 1, F, device=device) < p
        x[mask] = torch.rand(int(mask.sum()), device=device) * 2 - 1

        # Per-sample: which features are active?
        active = (x != 0).squeeze(1)                    # (B, F)
        comp_to_feat = comp_dom_feat.squeeze()          # (C,)

        # Build per-sample topk_mask of shape (B, 1, C)
        # For each sample, set mask[c] = True iff comp_to_feat[c] is in active features.
        feat_active_per_comp = active[:, comp_to_feat]  # (B, C)
        if keep_active:
            topk_mask = feat_active_per_comp.unsqueeze(1)       # (B, 1, C)
        else:
            topk_mask = (~feat_active_per_comp).unsqueeze(1)    # (B, 1, C)

        y_target = target_model(x).squeeze(1)
        y_spd = spd(x, topk_mask=topk_mask).squeeze(1)
        mse = ((y_spd - y_target) ** 2).mean(dim=-1)    # (B,)
        return mse.cpu().numpy(), y_target.cpu().numpy(), y_spd.cpu().numpy()


def sample_forward_compare(target_model, spd, meta, n_samples=1000, device="cpu"):
    """Get full MSE of spd vs target under standard forward (all components on)
    and batch-topk forward (matching training).
    """
    with torch.no_grad():
        F = target_model.config.n_features
        p = meta["p"]
        torch.manual_seed(7)
        x = torch.zeros(n_samples, 1, F, device=device)
        mask = torch.rand(n_samples, 1, F, device=device) < p
        x[mask] = torch.rand(int(mask.sum()), device=device) * 2 - 1

        y_target = target_model(x).squeeze(1)
        y_spd_full = spd(x).squeeze(1)
        mse_full = ((y_spd_full - y_target) ** 2).mean().item()

        return dict(mse_full=mse_full,
                    per_feat_target_mean=(y_target ** 2).mean().item())


def analyze_run(run_dir: Path, save: bool = True):
    target_model, spd, meta = load_run(run_dir)
    cfg = target_model.config
    F = cfg.n_features
    C = spd.C

    contribs, per_feat_per_comp, max_feat, max_vals, signed_max = \
        component_contribution_per_feature(target_model, spd)
    in_mag, out_mag = component_in_out_per_feature(target_model, spd)
    cats, n_active_per_c = component_classify(per_feat_per_comp)

    cat_names = ["dead", "mono", "duo", "poly"]
    cat_counts = [(cats == k).sum().item() for k in range(4)]
    mono_dom_feats = max_feat[cats == 1]
    features_covered = len(set(mono_dom_feats.tolist()))

    # Separate in/out analyses
    in_cats, _ = component_classify(in_mag)
    out_cats, _ = component_classify(out_mag)
    in_cat_counts = [(in_cats == k).sum().item() for k in range(4)]
    out_cat_counts = [(out_cats == k).sum().item() for k in range(4)]
    # For each mono component, its dominant feature
    in_dom = in_mag.argmax(dim=0)
    out_dom = out_mag.argmax(dim=0)
    in_mono_feats = in_dom[in_cats == 1]
    out_mono_feats = out_dom[out_cats == 1]
    # Features "read" / "written" by some mono component
    in_features_read = len(set(in_mono_feats.tolist()))
    out_features_written = len(set(out_mono_feats.tolist()))

    # Alignment: per component c, are in-dom and out-dom the same feature?
    alive_both = ((in_mag.max(dim=0).values > 0.05 * in_mag.max())
                  & (out_mag.max(dim=0).values > 0.05 * out_mag.max())).nonzero().flatten()
    n_alive_both = len(alive_both)
    n_in_out_aligned = (in_dom[alive_both] == out_dom[alive_both]).sum().item() if n_alive_both > 0 else 0

    print(f"\n=== {run_dir.name} ===")
    print(f"  F={F}, C={C}, d_mlp={cfg.d_mlp}, d_embed={cfg.d_embed}")
    print(f"  [Joint (in*out)] categories: dead={cat_counts[0]}, mono={cat_counts[1]}, "
          f"duo={cat_counts[2]}, poly={cat_counts[3]}")
    print(f"  [In-only]       categories: dead={in_cat_counts[0]}, mono={in_cat_counts[1]}, "
          f"duo={in_cat_counts[2]}, poly={in_cat_counts[3]}; reads {in_features_read}/{F} features")
    print(f"  [Out-only]      categories: dead={out_cat_counts[0]}, mono={out_cat_counts[1]}, "
          f"duo={out_cat_counts[2]}, poly={out_cat_counts[3]}; writes {out_features_written}/{F} features")
    print(f"  Joint mono components cover {features_covered}/{F} features")
    print(f"  Components alive on both in+out: {n_alive_both}/{C}; "
          f"in/out aligned to same feature: {n_in_out_aligned}/{n_alive_both}")

    summary = dict(
        name=run_dir.name, F=F, C=C,
        dead=cat_counts[0], mono=cat_counts[1], duo=cat_counts[2], poly=cat_counts[3],
        mono_features_covered=features_covered,
        in_mono=in_cat_counts[1], in_features_read=in_features_read,
        out_mono=out_cat_counts[1], out_features_written=out_features_written,
        n_alive_both=int(n_alive_both), n_in_out_aligned=int(n_in_out_aligned),
    )

    mse_scrub, _, _ = scrubbing_experiment(target_model, spd, meta, keep_active=True)
    mse_antiscrub, _, _ = scrubbing_experiment(target_model, spd, meta, keep_active=False)
    mse_full = sample_forward_compare(target_model, spd, meta)
    summary["mse_all_components"] = mse_full["mse_full"]
    summary["mse_scrubbed_mean"] = float(np.mean(mse_scrub))
    summary["mse_antiscrubbed_mean"] = float(np.mean(mse_antiscrub))
    summary["mse_ratio_anti_over_scrub"] = float(
        np.mean(mse_antiscrub) / max(np.mean(mse_scrub), 1e-20)
    )
    print(f"  MSE (all comps):      {mse_full['mse_full']:.5f}")
    print(f"  MSE (scrubbed):       mean {np.mean(mse_scrub):.5f}, median {np.median(mse_scrub):.5f}")
    print(f"  MSE (anti-scrubbed):  mean {np.mean(mse_antiscrub):.5f}")
    print(f"  anti/scrub ratio:     {summary['mse_ratio_anti_over_scrub']:.2f}")

    # ── plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plot_contribution_heatmap(in_mag, axes[0],
                              f"{run_dir.name}\nIN-side |read| per (feature, component)")
    plot_contribution_heatmap(out_mag, axes[1],
                              f"OUT-side |write| per (feature, component)")
    width = 0.35
    xs = np.arange(4)
    axes[2].bar(xs - width/2, in_cat_counts, width, label="in-only", color="tab:blue")
    axes[2].bar(xs + width/2, out_cat_counts, width, label="out-only", color="tab:orange")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(cat_names)
    axes[2].set_title(f"Component categories (C={C}, F={F})")
    axes[2].set_ylabel("count")
    axes[2].legend()
    for i, c in enumerate(in_cat_counts):
        axes[2].text(i - width/2, c + 0.3, str(c), ha="center", va="bottom", fontsize=8)
    for i, c in enumerate(out_cat_counts):
        axes[2].text(i + width/2, c + 0.3, str(c), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(run_dir / "attribution_heatmap.png", dpi=150)
    plt.close(fig)

    # Scrubbing histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(-8, 1, 60)
    ax.hist(np.clip(mse_scrub, 1e-8, 10), bins=bins, alpha=0.6,
            label=f"scrubbed (keep active)  mean={np.mean(mse_scrub):.1e}",
            color="tab:pink")
    ax.hist(np.clip(mse_antiscrub, 1e-8, 10), bins=bins, alpha=0.6,
            label=f"anti-scrubbed (drop active)  mean={np.mean(mse_antiscrub):.1e}",
            color="tab:green")
    ax.axvline(mse_full["mse_full"], color="black", linestyle="--",
               label=f"all-comps reconstruction  mean={mse_full['mse_full']:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("per-sample MSE vs target")
    ax.set_ylabel("count")
    ax.set_title(f"{run_dir.name}: scrubbing experiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(run_dir / "scrubbing.png", dpi=150)
    plt.close(fig)

    # Save summary
    if save:
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    all_summaries = []
    for r in args.runs:
        s = analyze_run(Path(r))
        all_summaries.append(s)

    # print combined table
    print("\n\n===== Summary table =====")
    for s in all_summaries:
        print(s)
