"""Load an existing SPD checkpoint and compute codeword analysis.
This reuses the already-run checkpoint from _spd_critique/out_plain_20f_2n_quick.

Also runs full 20f_2n training in a LESS expensive way: fewer training steps,
and uses the summaries for what we DO have.

Key test: shuffle-null for plain_20f_2n. We have the original ci data? No, we
have the heatmap image but no checkpoint saved.
So strategy: use our 5000-step checkpoint as a best-available proxy.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path("/home/francisco/Projects/vibeathon")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spd_new"))

import numpy as np
import torch
from spd_decomposition.models import PlainCiSModel
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances

device = "cpu"

# Load ci_in/out from the completed 5000-step quick run (saved as npy)
ci_in_quick = np.load(REPO / "_spd_critique/ci_in_20f2n.npy")
ci_out_quick = np.load(REPO / "_spd_critique/ci_out_20f2n.npy")

# Codeword groups for plain_20f_2n
codeword_group = {
    **{j: "A" for j in [1, 4, 7, 15, 16, 17, 18]},
    **{j: "B" for j in [2, 3, 5, 8, 9, 11, 12]},
    **{j: "C" for j in [0, 6, 10, 13, 14, 19]},
}
groups = {"A": [1,4,7,15,16,17,18], "B": [2,3,5,8,9,11,12], "C": [0,6,10,13,14,19]}

def analyze_components(ci_in, name):
    max_in = ci_in.max(axis=0)
    alive = np.where(max_in > 0.05)[0]
    thresh = 0.5
    print(f"\n--- {name}: alive IN components ({len(alive)}): ---")
    # Codeword purity
    per_comp_purity = []
    matched_cw = []
    for c in alive:
        active = np.where(ci_in[:, c] > thresh)[0].tolist()
        if not active:
            active = [int(ci_in[:, c].argmax())]  # fall back to argmax
        gs = [codeword_group[f] for f in active]
        best_g, best_count = None, 0
        from collections import Counter
        for g, cnt in Counter(gs).items():
            if cnt > best_count:
                best_count, best_g = cnt, g
        purity = best_count / len(gs)
        per_comp_purity.append(purity)
        matched_cw.append(best_g)
        print(f"  comp {c}: active = {active}  -> {gs} purity={purity:.2f} "
              f"best_g={best_g}")
    print(f"Mean purity (fraction of features in active set matching a single "
          f"codeword): {np.mean(per_comp_purity):.3f}")
    # "alive components = one codeword group" metric
    pure_count = sum(p == 1.0 for p in per_comp_purity)
    print(f"Alive components with 100% purity: {pure_count}/{len(alive)}")
    return np.mean(per_comp_purity), pure_count, len(alive)

mean_p, pure_count, n_alive = analyze_components(ci_in_quick, "plain_20f_2n_quick (5000 steps)")

# Shuffle null: randomly reassign feature labels within the F=20 space, 1000 times,
# and measure purity. This tells us how much signal the "grouping matches codewords"
# claim actually has.
print("\n\n======== SHUFFLE NULL ========")
feats = list(range(20))
max_in = ci_in_quick.max(axis=0)
alive_comps = np.where(max_in > 0.05)[0]
thresh = 0.5

# Active feature sets (per alive component)
active_sets = []
for c in alive_comps:
    active = np.where(ci_in_quick[:, c] > thresh)[0].tolist()
    if not active:
        active = [int(ci_in_quick[:, c].argmax())]
    active_sets.append(active)

def purity_against_cw(feature_assignment, active_sets):
    """feature_assignment is a dict feat->group; measure mean purity."""
    ps = []
    from collections import Counter
    for s in active_sets:
        gs = [feature_assignment[f] for f in s]
        ps.append(Counter(gs).most_common(1)[0][1] / len(gs))
    return np.mean(ps), sum(p == 1.0 for p in ps)

# Real codeword groups:
real_p, real_pure = purity_against_cw(codeword_group, active_sets)
print(f"REAL codeword purity: mean={real_p:.3f}  n_pure={real_pure}/{len(alive_comps)}")

# Shuffle null: randomize the feature labels keeping group sizes
rng = np.random.RandomState(0)
null_ps = []
null_pures = []
for trial in range(1000):
    perm = rng.permutation(20)
    shuf_group = {perm[i]: codeword_group[i] for i in range(20)}
    mp, npure = purity_against_cw(shuf_group, active_sets)
    null_ps.append(mp)
    null_pures.append(npure)
null_ps = np.array(null_ps); null_pures = np.array(null_pures)
print(f"Shuffle NULL mean purity: {null_ps.mean():.3f} +- {null_ps.std():.3f}  "
      f"[5%,95%] = [{np.percentile(null_ps, 5):.3f}, {np.percentile(null_ps, 95):.3f}]")
print(f"Shuffle NULL n_pure: {null_pures.mean():.2f} +- {null_pures.std():.2f}  "
      f"[5%,95%] = [{np.percentile(null_pures, 5):.2f}, {np.percentile(null_pures, 95):.2f}]")
print(f"Real mean purity z-score: "
      f"{(real_p - null_ps.mean()) / (null_ps.std() + 1e-9):.2f}")
