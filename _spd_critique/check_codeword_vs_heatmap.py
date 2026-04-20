"""For plain_20f_2n: the IN-side heatmap shows 5 rows. The codewords cleanly split
20 features into groups of 7, 7, 6. Here, we eyeball what each IN component's
"active feature set" is (by reading off the displayed heatmap image pixels).

Rather than eyeballing, we re-run SPD cheaply (fewer steps) so we can access
the SPD state_dict and compute ci exactly. We use 3000 steps, enough to
produce a qualitative match and verify whether the "codeword grouping"
pattern is actually real.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/francisco/Projects/vibeathon")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "spd_new"))

import torch
import numpy as np
from spd_decomposition.models import PlainCiSModel
from spd.configs import Config, ResidualMLPTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import optimize
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def quick_spd(weights, n_features, d_mlp, p, C=30, steps=5000, tag="quick"):
    target_model = PlainCiSModel.from_our_weights(
        str(weights), n_features, d_mlp, device=device
    )
    target_model.eval()
    dataset = ResidualMLPDataset(
        n_features=n_features, feature_probability=p, device=device,
        calc_labels=False, label_type=None, act_fn_name=None,
        label_fn_seed=None, label_coeffs=None,
        data_generation_type="at_least_zero_active", synced_inputs=None,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=2048, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=2048, shuffle=False)
    task_cfg = ResidualMLPTaskConfig(
        feature_probability=p, data_generation_type="at_least_zero_active",
    )
    cfg = Config(
        wandb_project=None, wandb_run_name=None, wandb_run_name_prefix="",
        seed=0, C=C, n_mask_samples=1, n_ci_mlp_neurons=16,
        target_module_patterns=["mlp_in", "mlp_out"],
        faithfulness_coeff=1.0, recon_coeff=None,
        stochastic_recon_coeff=1.0, recon_layerwise_coeff=None,
        stochastic_recon_layerwise_coeff=1.0,
        importance_minimality_coeff=1e-5,
        schatten_coeff=None, out_recon_coeff=None, embedding_recon_coeff=None,
        pnorm=2.0, output_loss_type="mse",
        lr=2e-3, steps=steps, batch_size=2048,
        lr_schedule="constant", lr_warmup_pct=0.0,
        n_eval_steps=20, image_freq=None, image_on_first_step=False,
        print_freq=max(1, steps // 10),
        save_freq=steps,  # save at end
        log_ce_losses=False,
        pretrained_model_class="spd_decomposition.models.PlainCiSModel",
        pretrained_model_path=None, pretrained_model_output_attr=None,
        task_config=task_cfg,
    )
    out_dir = REPO / "_spd_critique" / f"out_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    optimize(
        target_model=target_model, config=cfg, device=device,
        train_loader=train_loader, eval_loader=eval_loader,
        n_eval_steps=cfg.n_eval_steps, out_dir=out_dir,
        plot_results_fn=None,
    )
    # Reload the final component model and compute ci
    spd = ComponentModel(
        base_model=target_model,
        target_module_patterns=["mlp_in", "mlp_out"],
        C=C, n_ci_mlp_neurons=16, pretrained_model_output_attr=None,
    )
    ckpt = sorted(out_dir.glob("model_*.pth"))[-1]
    state = torch.load(ckpt, map_location=device, weights_only=True)
    spd.load_state_dict(state)
    spd.eval()
    return target_model, spd

def compute_ci(target, spd, n_features):
    components = {
        k.removeprefix("components.").replace("-", "."): v
        for k, v in spd.components.items()
    }
    gates = {
        k.removeprefix("gates.").replace("-", "."): v
        for k, v in spd.gates.items()
    }
    x = torch.eye(n_features, device=device)
    _, pre = spd.forward_with_pre_forward_cache_hooks(
        x, module_names=list(components.keys())
    )
    As = {name: c.A for name, c in components.items()}
    ci, _ = calc_causal_importances(pre_weight_acts=pre, As=As, gates=gates)
    return {k: v.detach().cpu().numpy() for k, v in ci.items()}

# ---- plain_20f_2n ----
print("\n\n======== plain_20f_2n (30000 steps, matches original) ========")
target, spd = quick_spd(
    REPO / "weights/small_20f_2n_L4.pt", 20, 2, 0.05, C=30, steps=30000,
    tag="plain_20f_2n_repro",
)
ci = compute_ci(target, spd, 20)
ci_in = ci["mlp_in"]; ci_out = ci["mlp_out"]
print(f"\nci_in shape: {ci_in.shape}")
print(f"max_per_component (in): {ci_in.max(axis=0).round(2)}")

thresh = 0.5
print("\nAlive IN components (max > 0.05):")
max_in = ci_in.max(axis=0)
alive_in = np.where(max_in > 0.05)[0]
for c in alive_in:
    active = np.where(ci_in[:, c] > thresh)[0].tolist()
    print(f"  comp {c}: active features (ci>0.5) = {active}  max={max_in[c]:.2f}")

print("\nAlive OUT components:")
max_out = ci_out.max(axis=0)
alive_out = np.where(max_out > 0.05)[0]
for c in alive_out:
    active = np.where(ci_out[:, c] > thresh)[0].tolist()
    print(f"  comp {c}: active features (ci>0.5) = {active}  max={max_out[c]:.2f}")

# Codeword assignment for 20f_2n
codeword_group = {
    **{j: "A" for j in [1, 4, 7, 15, 16, 17, 18]},   # (0,1)
    **{j: "B" for j in [2, 3, 5, 8, 9, 11, 12]},     # (1,0)
    **{j: "C" for j in [0, 6, 10, 13, 14, 19]},       # (1,1)
}
def classify_group(feats):
    gs = [codeword_group[f] for f in feats]
    if len(set(gs)) == 1: return f"one codeword ({gs[0]}, n={len(gs)})"
    return f"mixed ({gs})"

print("\n--- Classification against codewords ---")
for c in alive_in:
    active = np.where(ci_in[:, c] > thresh)[0].tolist()
    if active:
        print(f"  in-comp {c}: {active}  -> {classify_group(active)}")

import json
np.save(REPO / "_spd_critique/ci_in_20f2n.npy", ci_in)
np.save(REPO / "_spd_critique/ci_out_20f2n.npy", ci_out)
