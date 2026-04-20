"""Check: for the embedded runs where alive_both=0, do IN-alive components and
OUT-alive components pair up by their feature-support sets (rather than by
component-index)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spd_new"))

from spd_decomposition.analyze import load_run, compute_ci_one_hot


def active_feats(ci_col, max_val, ratio=0.5):
    return frozenset(np.where(ci_col / max(max_val, 1e-12) > ratio)[0].tolist())


def analyze(run_name):
    run_dir = REPO / "spd_decomposition" / "out" / run_name
    target, spd, meta = load_run(run_dir)
    ci = compute_ci_one_hot(target, spd, meta)

    print(f"\n=== {run_name} ===")
    in_alive, out_alive = {}, {}
    for side, alive_dict in [("mlp_in", in_alive), ("mlp_out", out_alive)]:
        ci_mat = ci[side]
        max_per_c = ci_mat.max(axis=0)
        alive = np.where(max_per_c > 0.1)[0]
        for c in alive:
            fs = active_feats(ci_mat[:, c], max_per_c[c])
            alive_dict[int(c)] = fs
        print(f"  {side}: {len(alive)} alive  component indices: {alive.tolist()}")

    # Pair by feature-set: for each IN component, is there an OUT component with same set?
    out_sets = {fs: c for c, fs in out_alive.items()}
    pairs = []
    for c_in, fs_in in in_alive.items():
        if fs_in in out_sets:
            pairs.append((c_in, out_sets[fs_in], fs_in))
    # Also check near-match (Jaccard >= 0.8)
    near_pairs = []
    for c_in, fs_in in in_alive.items():
        best_jac, best_c_out = 0, None
        for c_out, fs_out in out_alive.items():
            union = fs_in | fs_out
            inter = fs_in & fs_out
            jac = len(inter) / max(len(union), 1)
            if jac > best_jac:
                best_jac, best_c_out = jac, c_out
        near_pairs.append((c_in, best_c_out, best_jac, fs_in))

    print(f"\n  exact feature-set matches: {len(pairs)}")
    for c_in, c_out, fs in pairs:
        print(f"    in_c={c_in}  <->  out_c={c_out}  feats={sorted(fs)}")

    print(f"\n  best jaccard match per IN comp:")
    for c_in, c_out, jac, fs in near_pairs:
        print(f"    in_c={c_in} -> out_c={c_out}, jaccard={jac:.2f}, "
              f"in_feats={sorted(fs)}, out_feats={sorted(out_alive[c_out]) if c_out is not None else []}")


if __name__ == "__main__":
    for name in ["plain_20f_5n", "plain_20f_2n", "embed_20f_5n_D80", "embed_20f_2n_D40"]:
        analyze(name)
