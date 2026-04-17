"""
Decouple identification-bottleneck from generic-difficulty.

Fix n=5 neurons. Sweep F (features) with Fp held constant (Fp = 0.4).
Train BottleneckMLP (linear W_in + MLP decoder) at each (F, p).
Measure single-feature recovery error ||sR - I||_F^2 / F.

Threshold: 2^n - 1 = 31 non-trivial codewords.
Prediction: per-feature error ~flat for F <= 31, rises for F > 31
(pigeonhole-forced codeword collisions prevent identification).
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from small_models import DEVICE, generate_batch
from mlp_decoder import BottleneckMLP

WEIGHTS_DIR = Path("weights")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
DATA_PATH = FIGURES_DIR / "codeword_knee_data.json"

N = 5
FP = 0.4
SWEEP = [15, 20, 25, 31, 35, 50, 100]

BATCH = 2048
STEPS = 20_000
LR = 3e-3


def train_bottleneck(F, p, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = BottleneckMLP(F, N).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        x, t = generate_batch(BATCH, F, p)
        loss = (torch.abs(m(x) - t) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 2000 == 0:
            print(f"    F={F} p={p:.4f} step {step+1}/{STEPS}  L4={loss.item():.4e}")
    return m


def response_matrix(model, F):
    model.eval()
    R = np.zeros((F, F))
    with torch.no_grad():
        for j in range(F):
            x = torch.zeros(1, F, device=DEVICE)
            x[0, j] = 1.0
            R[:, j] = model(x).cpu().numpy()[0]
    return R


def opt_scale_err(R):
    s = np.trace(R) / (R ** 2).sum()
    return float(s), float(((s * R - np.eye(R.shape[0])) ** 2).sum())


def codeword_partition(W_in):
    """Return descending counts of non-empty sign-pattern codewords."""
    signs = np.sign(W_in)
    codewords = {}
    for j in range(W_in.shape[1]):
        cw = tuple(int(s) for s in signs[:, j])
        codewords[cw] = codewords.get(cw, 0) + 1
    return sorted(codewords.values(), reverse=True)


def run_sweep():
    results = []
    for F in SWEEP:
        p = FP / F
        tag = f"mlp_decoder_{F}f_{N}n_p{p:.5f}_L4"
        ckpt = WEIGHTS_DIR / f"{tag}.pt"
        print(f"\n=== F={F}  p={p:.5f}  Fp={F*p:.3f} ===")
        if ckpt.exists():
            m = BottleneckMLP(F, N).to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE,
                                         weights_only=True))
            print(f"  loaded {ckpt}")
        else:
            m = train_bottleneck(F, p)
            torch.save(m.state_dict(), ckpt)
            print(f"  trained and saved {ckpt}")

        R = response_matrix(m, F)
        s, err = opt_scale_err(R)
        W_in = m.W_in.detach().cpu().numpy()
        partition = codeword_partition(W_in)

        n_codewords_used = len(partition)
        n_colliding = sum(c for c in partition if c > 1)
        print(f"  ||sR-I||_F^2 = {err:.3f}   per-feat = {err/F:.4f}")
        print(f"  codewords used: {n_codewords_used}/{2**N - 1}  "
              f"features sharing a codeword: {n_colliding}")
        print(f"  partition (desc): {partition}")

        results.append(dict(
            F=F, p=p, n=N, s=s, err=err, err_per_feat=err / F,
            n_codewords_used=n_codewords_used,
            n_colliding=n_colliding,
            partition=partition,
        ))

    DATA_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved data to {DATA_PATH}")
    return results


def make_figure(results):
    Fs = [r["F"] for r in results]
    err_per_feat = [r["err_per_feat"] for r in results]
    rank_floor = [(F - N) / F for F in Fs]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Fs, err_per_feat, "o-", color="#2ca02c", lw=2, ms=8,
            label="trained MLP decoder", zorder=3)
    ax.plot(Fs, rank_floor, "--", color="#1f77b4", lw=1.4, alpha=0.8,
            label=r"rank-$n$ floor (linear): $(F - n)/F$")

    thresh = 2 ** N - 1
    ax.axvline(thresh, color="red", ls=":", lw=1.6, alpha=0.8,
               label=rf"$2^n - 1 = {thresh}$ codewords")

    for F, e in zip(Fs, err_per_feat):
        ax.annotate(f"{e:.3f}", xy=(F, e), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#2ca02c")

    ax.set_xlabel(r"$F$ (features) — $p$ chosen so $Fp = 0.4$")
    ax.set_ylabel(r"$\|sR - I\|_F^2 / F$  (per-feature identification error)")
    ax.set_title(rf"Identification bottleneck sweep  ($n = {N}$, $Fp$ constant)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    out = FIGURES_DIR / "slide_codeword_knee.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}  n={N}  Fp={FP}  F sweep: {SWEEP}")
    results = run_sweep()
    make_figure(results)
