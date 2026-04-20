"""Run APD (Attribution-based Parameter Decomposition) on our L^4 CiS models.

Usage:
    python run_decomposition.py --weights weights/small_20f_5n_L4.pt --variant plain \\
         --n-features 20 --d-mlp 5 --C 40 --steps 10000

The script:
 1. Patches ResidualMLPModel to skip the residual connection (our arch has none).
 2. Loads our trained weights into the patched ResidualMLPModel.
 3. Creates a matching ResidualMLPSPDModel (also patched) with C parameter components.
 4. Copies fixed W_E, W_U from the target into the SPD model.
 5. Runs optimize() from spd.run_spd to train the decomposition.

Saves into apd_decomposition/out/<run_name>/ the trained SPD weights + metadata.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from apd_decomposition.load_our_model import (
    install_no_residual_patch,
    load_plain_model,
    load_embedded_model,
    verify_model_matches,
)

from spd.experiments.resid_mlp.models import (
    ResidualMLPSPDModel,
    ResidualMLPSPDConfig,
)
from spd.run_spd import Config, ResidualMLPTaskConfig, optimize
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.utils import DatasetGeneratedDataLoader, set_seed


def build_spd_model(target_model, C: int, m: int | None = None,
                    init_scale: float = 1.0, device: str = "cpu"):
    """Create a ResidualMLPSPDModel mirroring the target model's architecture,
    with C parameter components per layer. W_E/W_U are copied (frozen)."""
    cfg = target_model.config
    spd_cfg = ResidualMLPSPDConfig(
        n_instances=cfg.n_instances,
        n_features=cfg.n_features,
        d_embed=cfg.d_embed,
        d_mlp=cfg.d_mlp,
        n_layers=cfg.n_layers,
        act_fn_name=cfg.act_fn_name,
        apply_output_act_fn=cfg.apply_output_act_fn,
        in_bias=cfg.in_bias,
        out_bias=cfg.out_bias,
        init_scale=init_scale,
        C=C,
        m=m,
    )
    model = ResidualMLPSPDModel(config=spd_cfg).to(device)

    # Copy fixed W_E, W_U from target
    with torch.no_grad():
        model.W_E.data[:] = target_model.W_E.data.detach().clone()
        model.W_U.data[:] = target_model.W_U.data.detach().clone()
    model.W_E.requires_grad_(False)
    model.W_U.requires_grad_(False)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", choices=["plain", "embedded"], required=True)
    ap.add_argument("--n-features", type=int, default=None,
                    help="Required for plain; inferred for embedded")
    ap.add_argument("--d-mlp", type=int, default=None,
                    help="Required for plain; inferred for embedded")
    ap.add_argument("--C", type=int, default=40)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=float, default=1.28)
    ap.add_argument("--exact-topk", action="store_true")
    ap.add_argument("--data-gen", type=str, default="at_least_zero_active",
                    choices=["at_least_zero_active", "exactly_one_active", "exactly_two_active"])
    ap.add_argument("--attribution-type", type=str, default="gradient",
                    choices=["gradient", "ablation", "activation"])
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="Output subdir name")
    ap.add_argument("--init-scale", type=float, default=2.0)
    ap.add_argument("--param-match-coeff", type=float, default=1.0)
    ap.add_argument("--topk-recon-coeff", type=float, default=1.0)
    ap.add_argument("--act-recon-coeff", type=float, default=1.0)
    ap.add_argument("--schatten-coeff", type=float, default=10.0)
    ap.add_argument("--schatten-pnorm", type=float, default=0.9)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Patch the forward methods to skip residual
    install_no_residual_patch()

    # Load target
    if args.variant == "plain":
        assert args.n_features is not None and args.d_mlp is not None
        target_model, meta = load_plain_model(
            args.weights, args.n_features, args.d_mlp, device=device
        )
    else:
        target_model, meta = load_embedded_model(args.weights, device=device)

    target_model.eval()
    # Note: don't freeze target params — APD needs the grad graph for
    # attribution, even though target params are never optimized.

    # Sanity check
    diff = verify_model_matches(target_model, args.weights, args.variant,
                                p=args.p, device=device)
    print(f"Target model output diff vs reference: {diff:.2e}")
    assert diff < 1e-5, "Target model disagreement"

    # Build SPD model
    spd_model = build_spd_model(target_model, C=args.C, m=args.m,
                                init_scale=args.init_scale, device=device)

    # param_names: which params to decompose
    param_names = ["layers.0.mlp_in", "layers.0.mlp_out"]

    # Dataset (value_range hard-coded to (-1, 1) by ResidualMLPDataset init below)
    dataset = ResidualMLPDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        feature_probability=args.p,
        device=device,
        calc_labels=False,       # labels = target model output (distillation)
        label_type=None,
        act_fn_name=None,
        data_generation_type=args.data_gen,
    )
    # Override value_range to (-1, 1) to match our training data
    dataset.value_range = (-1.0, 1.0)

    dataloader = DatasetGeneratedDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # Build APD Config
    task_cfg = ResidualMLPTaskConfig(
        feature_probability=args.p,
        init_scale=args.init_scale,
        data_generation_type=args.data_gen,
        pretrained_model_path="n/a",
    )
    cfg = Config(
        wandb_project=None,
        seed=args.seed,
        topk=args.topk,
        batch_topk=True,
        exact_topk=args.exact_topk,
        batch_size=args.batch_size,
        steps=args.steps,
        print_freq=max(1, args.steps // 20),
        image_freq=args.steps,
        image_on_first_step=False,
        save_freq=args.steps,
        lr=args.lr,
        param_match_coeff=args.param_match_coeff,
        topk_recon_coeff=args.topk_recon_coeff,
        act_recon_coeff=args.act_recon_coeff,
        schatten_coeff=args.schatten_coeff,
        schatten_pnorm=args.schatten_pnorm,
        lp_sparsity_coeff=None,
        distil_from_target=False,
        pnorm=None,
        C=args.C,
        m=args.m,
        lr_schedule="cosine",
        lr_warmup_pct=0.01,
        unit_norm_matrices=True,
        attribution_type=args.attribution_type,
        task_config=task_cfg,
    )

    tag = args.tag or (Path(args.weights).stem + f"_C{args.C}")
    out_dir = Path("apd_decomposition/out") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(out_dir / "meta.json", "w") as f:
        json.dump({**meta, **vars(args)}, f, indent=2)

    t0 = time.time()
    optimize(
        model=spd_model,
        config=cfg,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        plot_results_fn=None,
        out_dir=out_dir,
    )
    print(f"Done in {time.time()-t0:.1f}s")

    # Save SPD model
    torch.save(spd_model.state_dict(), out_dir / "spd_model.pt")
    # Also save target weight (for easy post-hoc comparison)
    torch.save(target_model.state_dict(), out_dir / "target_model.pt")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
