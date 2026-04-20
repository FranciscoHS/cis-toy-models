"""Better scrubbing test: use per-sample causal importance directly to
identify "causally active" components, instead of the dominant-feature
argmax used previously.

For each sample x and each component c, ci[c] is SPD's own estimate of
whether c is active on x. We use ci > threshold as the "relevant" mask.

- scrub   (keep relevant):    mask[c] = ci[c] if ci[c] >  t else 0
- antscrub (drop relevant):   mask[c] = ci[c] if ci[c] <= t else 0
- random scramble (null):     swap which samples each component is "relevant" for,
                              keeping the total active-count the same

If SPD's ci is meaningfully causal:
  MSE(anti-scrub) >> MSE(scrub)
  MSE(scrub) ≈ MSE(full-ci-mask)
  MSE(anti-scrub) ≥ MSE(random-scramble)   (because the kept components are the
                                            ones that don't actually predict this
                                            sample's output)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spd_new"))

from spd_decomposition.analyze import load_run
from spd.models.component_utils import calc_causal_importances


def ci_scrubbing_experiment(target, spd, meta, n_samples=4000,
                            thresholds=(0.1, 0.3, 0.5), device="cpu"):
    F_ = meta["n_features"]
    p = meta["p"]

    torch.manual_seed(777)
    x = torch.zeros(n_samples, F_, device=device)
    mask = torch.rand(n_samples, F_, device=device) < p
    x[mask] = torch.rand(int(mask.sum()), device=device) * 2 - 1

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
        _, pre_acts = spd.forward_with_pre_forward_cache_hooks(
            x, module_names=list(components.keys())
        )
        As = {n: c.A for n, c in components.items()}
        actual_ci, _ = calc_causal_importances(
            pre_weight_acts=pre_acts, As=As, gates=gates
        )
        # Full-ci mask forward: the intended SPD mask
        y_full_ci = spd.forward_with_components(
            x, components=components, masks=actual_ci
        )
        mse_full_ci = ((y_full_ci - y_target) ** 2).mean(dim=-1).cpu().numpy()

        results = {"full_ci": float(np.mean(mse_full_ci))}
        for t in thresholds:
            scrub_masks = {
                m: ci.where(ci > t, torch.zeros_like(ci))
                for m, ci in actual_ci.items()
            }
            anti_masks = {
                m: ci.where(ci <= t, torch.zeros_like(ci))
                for m, ci in actual_ci.items()
            }
            y_scrub = spd.forward_with_components(
                x, components=components, masks=scrub_masks
            )
            y_anti = spd.forward_with_components(
                x, components=components, masks=anti_masks
            )
            mse_scrub = ((y_scrub - y_target) ** 2).mean(dim=-1).cpu().numpy()
            mse_anti = ((y_anti - y_target) ** 2).mean(dim=-1).cpu().numpy()

            # Random-scramble null: for each component, sample new ci values that
            # preserve its per-component distribution (using a random permutation
            # of sample indices per component), then apply scrub/anti with those.
            scrambled_ci = {}
            for m, ci in actual_ci.items():
                B, C = ci.shape
                # permute rows independently per component (column)
                perms = [torch.randperm(B) for _ in range(C)]
                scrambled = torch.stack([ci[perms[c], c] for c in range(C)], dim=-1)
                scrambled_ci[m] = scrambled
            scr_scrub_masks = {
                m: ci.where(ci > t, torch.zeros_like(ci))
                for m, ci in scrambled_ci.items()
            }
            scr_anti_masks = {
                m: ci.where(ci <= t, torch.zeros_like(ci))
                for m, ci in scrambled_ci.items()
            }
            y_rand_scrub = spd.forward_with_components(
                x, components=components, masks=scr_scrub_masks
            )
            y_rand_anti = spd.forward_with_components(
                x, components=components, masks=scr_anti_masks
            )
            mse_rand_scrub = ((y_rand_scrub - y_target) ** 2).mean(dim=-1).cpu().numpy()
            mse_rand_anti = ((y_rand_anti - y_target) ** 2).mean(dim=-1).cpu().numpy()

            results[f"t={t}"] = {
                "scrub_mean": float(np.mean(mse_scrub)),
                "anti_mean": float(np.mean(mse_anti)),
                "rand_scrub_mean": float(np.mean(mse_rand_scrub)),
                "rand_anti_mean": float(np.mean(mse_rand_anti)),
                "anti_over_scrub": float(np.mean(mse_anti) / max(np.mean(mse_scrub), 1e-20)),
                "anti_over_rand_anti": float(np.mean(mse_anti) / max(np.mean(mse_rand_anti), 1e-20)),
            }
    return results


if __name__ == "__main__":
    runs = ["paper_style_20f_5n", "plain_20f_5n", "plain_20f_2n",
            "plain_100f_10n", "embed_20f_5n_D80", "embed_20f_2n_D40"]
    print(f"{'target':<22} {'mse(full ci)':<13} | "
          f"{'scrub':<10} {'anti':<10} {'ratio':<7} | "
          f"{'rand_anti':<10} {'anti/rand':<10}")
    print("-" * 100)

    summary = {}
    for name in runs:
        run_dir = REPO / "spd_decomposition" / "out" / name
        if not run_dir.exists():
            continue
        target, spd, meta = load_run(run_dir)
        results = ci_scrubbing_experiment(target, spd, meta)
        summary[name] = results
        t05 = results["t=0.5"]
        print(f"{name:<22} {results['full_ci']:.2e}      | "
              f"{t05['scrub_mean']:.2e}  {t05['anti_mean']:.2e}  "
              f"{t05['anti_over_scrub']:>6.2f} | "
              f"{t05['rand_anti_mean']:.2e}  {t05['anti_over_rand_anti']:>7.2f}")

    with open(REPO / "spd_decomposition" / "ci_scrub_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved ci_scrub_results.json")
