"""Shuffle-null test for the 'codeword grouping' claim.

For each L^4 target, compute:
  1. The target's codeword partition of features (sign pattern of W_in[:, j]).
  2. For the SPD decomposition, each alive IN-side component c has a set of
     features it fires for (ci[:, c] > 0.5). Score the alignment: for each
     alive component, what is the purity — the fraction of its active features
     that fall into its single best-matching codeword group?
  3. Compare this to a null where the feature-to-codeword mapping is shuffled
     (permuting which features belong to which group, keeping group sizes).

Output: mean purity (observed), null mean ± std, z-score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spd_new"))

from spd_decomposition.analyze import load_run, compute_ci_one_hot


def codeword_groups(weights_path, n_features, d_mlp, variant="plain"):
    """Return list-of-lists: groups[k] = list of feature indices in codeword k.
    For plain, codeword = sign pattern of W_in[:, j] (positive-set).
    For embedded, codeword = sign pattern of (W_in @ E[:, j]).
    """
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    W_in = sd["W_in"].numpy()
    if variant == "embedded":
        E = sd["E"].numpy()
        effective_W_in = W_in @ E          # (n, F)
    else:
        effective_W_in = W_in

    pos_codes = (effective_W_in.T > 0.05 * effective_W_in.T.max(axis=1, keepdims=True)).astype(int)
    from collections import defaultdict
    groups_map = defaultdict(list)
    for j in range(n_features):
        groups_map[tuple(pos_codes[j])].append(j)
    # list of lists
    return list(groups_map.values()), pos_codes


def purity_score(ci_mat, groups, alive_thresh=0.5, dead_thresh=0.1):
    """For each alive component c (max ci > dead_thresh), identify the set of
    'active features' where ci > alive_thresh * max_per_c. Purity = fraction of
    those that lie in the component's best-matching codeword group. Return
    mean purity over alive components, and per-component details.
    """
    F, C = ci_mat.shape
    max_per_c = ci_mat.max(axis=0)
    alive = max_per_c > dead_thresh
    purities = []
    per_comp = []
    for c in np.where(alive)[0]:
        # active features for this component
        ratios = ci_mat[:, c] / max(max_per_c[c], 1e-12)
        active_feats = set(np.where(ratios > alive_thresh)[0].tolist())
        if not active_feats:
            continue
        # best-matching group: maximises overlap
        best_overlap = 0
        best_group = None
        for g in groups:
            overlap = len(active_feats & set(g))
            if overlap > best_overlap:
                best_overlap = overlap
                best_group = g
        purity = best_overlap / len(active_feats)
        purities.append(purity)
        per_comp.append(dict(comp=int(c), n_active=len(active_feats),
                             purity=purity, active=sorted(active_feats),
                             matched_group=best_group))
    return np.array(purities), per_comp


def shuffle_null(ci_mat, groups, n_shuffles=1000, **kwargs):
    """Shuffle the feature-to-group assignment (preserving group sizes), recompute
    purity. Return mean/std over the shuffled distribution.
    """
    F = ci_mat.shape[0]
    sizes = [len(g) for g in groups]
    null_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        perm = rng.permutation(F)
        shuffled_groups = []
        idx = 0
        for s in sizes:
            shuffled_groups.append(list(perm[idx:idx + s]))
            idx += s
        p, _ = purity_score(ci_mat, shuffled_groups, **kwargs)
        if len(p):
            null_means.append(p.mean())
    null_means = np.array(null_means)
    return null_means.mean(), null_means.std()


def run_analysis(run_name, weights_path, n_features, d_mlp, variant="plain"):
    print(f"\n{'='*60}\n{run_name}\n{'='*60}")
    run_dir = REPO / "spd_decomposition" / "out" / run_name
    target, spd, meta = load_run(run_dir)
    ci = compute_ci_one_hot(target, spd, meta)

    groups, pos_codes = codeword_groups(weights_path, n_features, d_mlp, variant)
    print(f"Target has {len(groups)} distinct codewords. Group sizes: "
          f"{sorted([len(g) for g in groups], reverse=True)}")
    print(f"Groups: {groups}")

    for side in ["mlp_in", "mlp_out"]:
        ci_mat = ci[side]
        purities, per_comp = purity_score(ci_mat, groups)
        if len(purities) == 0:
            print(f"\n[{side}] No alive components.")
            continue
        null_mean, null_std = shuffle_null(ci_mat, groups)
        z = (purities.mean() - null_mean) / (null_std + 1e-12)
        print(f"\n[{side}] Alive components: {len(purities)}")
        print(f"  purities per component: {[f'{p:.2f}' for p in purities]}")
        print(f"  observed mean purity: {purities.mean():.3f}")
        print(f"  null mean purity: {null_mean:.3f} ± {null_std:.3f}")
        print(f"  z-score: {z:.2f}  (|z| > 2 ≈ significant)")
        for pc in per_comp[:8]:
            print(f"    comp {pc['comp']}: active {pc['active']} (n={pc['n_active']}), "
                  f"group {pc['matched_group']}, purity {pc['purity']:.2f}")


if __name__ == "__main__":
    configs = [
        ("plain_20f_5n", REPO / "weights/small_20f_5n_L4.pt", 20, 5, "plain"),
        ("plain_20f_2n", REPO / "weights/small_20f_2n_L4.pt", 20, 2, "plain"),
        ("plain_100f_10n", REPO / "weights/small_100f_10n_L4.pt", 100, 10, "plain"),
        ("embed_20f_5n_D80", REPO / "weights/embed_20f_5n_D80_unit_L4.pt", 20, 5, "embedded"),
        ("embed_20f_2n_D40", REPO / "weights/embed_20f_2n_D40_unit_L4.pt", 20, 2, "embedded"),
    ]
    for name, wp, F, n, variant in configs:
        run_analysis(name, wp, F, n, variant)
