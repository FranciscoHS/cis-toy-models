"""Driver: run the new SPD (stochastic parameter decomposition) on our CiS
targets. Writes decomposed SPD model + config to out_dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "spd_new"))

from spd.configs import Config, ResidualMLPTaskConfig
from spd.data_utils import DatasetGeneratedDataLoader
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import optimize

from spd_decomposition.models import PlainCiSModel, EmbeddedCiSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", choices=["plain", "embedded"], required=True)
    ap.add_argument("--n-features", type=int, default=None)
    ap.add_argument("--d-mlp", type=int, default=None)
    ap.add_argument("--C", type=int, default=100)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--n-mask-samples", type=int, default=1)
    ap.add_argument("--n-ci-mlp-neurons", type=int, default=16)
    ap.add_argument("--faithfulness-coeff", type=float, default=1.0)
    ap.add_argument("--stochastic-recon-coeff", type=float, default=1.0)
    ap.add_argument("--stochastic-recon-layerwise-coeff", type=float, default=1.0)
    ap.add_argument("--importance-minimality-coeff", type=float, default=1e-5)
    ap.add_argument("--pnorm", type=float, default=2.0)
    ap.add_argument("--data-gen", default="at_least_zero_active",
                    choices=["at_least_zero_active", "exactly_one_active", "exactly_two_active"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Build target model + load weights ──
    if args.variant == "plain":
        assert args.n_features and args.d_mlp
        target_model = PlainCiSModel.from_our_weights(
            args.weights, args.n_features, args.d_mlp, device=device
        )
        n_features = args.n_features
    else:
        target_model = EmbeddedCiSModel.from_our_weights(args.weights, device=device)
        n_features = target_model.n_features
    target_model.eval()

    # ── Dataset ──
    dataset = ResidualMLPDataset(
        n_features=n_features,
        feature_probability=args.p,
        device=device,
        calc_labels=False,             # labels = target model output (distillation)
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=args.data_gen,
        synced_inputs=None,
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # ── Config ──
    task_cfg = ResidualMLPTaskConfig(
        feature_probability=args.p,
        data_generation_type=args.data_gen,
    )
    cfg = Config(
        wandb_project=None,
        wandb_run_name=None,
        wandb_run_name_prefix="",
        seed=args.seed,
        C=args.C,
        n_mask_samples=args.n_mask_samples,
        n_ci_mlp_neurons=args.n_ci_mlp_neurons,
        target_module_patterns=["mlp_in", "mlp_out"],
        faithfulness_coeff=args.faithfulness_coeff,
        recon_coeff=None,
        stochastic_recon_coeff=args.stochastic_recon_coeff,
        recon_layerwise_coeff=None,
        stochastic_recon_layerwise_coeff=args.stochastic_recon_layerwise_coeff,
        importance_minimality_coeff=args.importance_minimality_coeff,
        schatten_coeff=None,
        out_recon_coeff=None,
        embedding_recon_coeff=None,
        pnorm=args.pnorm,
        output_loss_type="mse",
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        lr_schedule="constant",
        lr_warmup_pct=0.0,
        n_eval_steps=100,
        image_freq=None,                        # disable plotting during training
        image_on_first_step=False,
        print_freq=max(1, args.steps // 20),
        save_freq=None,
        log_ce_losses=False,
        pretrained_model_class="spd_decomposition.models.PlainCiSModel",
        pretrained_model_path=None,
        pretrained_model_output_attr=None,
        task_config=task_cfg,
    )

    tag = args.tag or (Path(args.weights).stem + f"_C{args.C}_newspd")
    out_dir = REPO_ROOT / "spd_decomposition" / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({**vars(args), "n_features": n_features,
                   "d_embed": int(target_model.d_embed) if args.variant == "embedded" else n_features,
                   "d_mlp": int(target_model.d_mlp)}, f, indent=2)

    optimize(
        target_model=target_model,
        config=cfg,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=cfg.n_eval_steps,
        out_dir=out_dir,
        plot_results_fn=None,
    )

    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()
