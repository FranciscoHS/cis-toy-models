"""
Decompose L4 loss by number of active features k for 20f/{2,3,4,5}n L4 models.
Tests the prediction:
 - n=2: loss dominated by k=1 events (within-codeword ambiguity, O(p)).
 - n=5: loss dominated by k=2+ events (unique codes; O(p^2) collisions).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from small_models import SimpleMLP, generate_batch, DEVICE

F = 20
P = 0.02
STEPS = 10000
BATCH = 2048
LR = 3e-3

CACHE = Path("weights")
CACHE.mkdir(exist_ok=True)


def train(n_neurons, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SimpleMLP(F, n_neurons).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        x, y = generate_batch(BATCH, F, P)
        loss = (torch.abs(m(x) - y) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def get_model(n):
    cached = CACHE / f"small_20f_{n}n_L4.pt"
    m = SimpleMLP(F, n).to(DEVICE)
    if cached.exists():
        m.load_state_dict(torch.load(cached, map_location=DEVICE, weights_only=True))
        print(f"  loaded {cached}")
    else:
        m = train(n)
        torch.save(m.state_dict(), cached)
        print(f"  trained and saved {cached}")
    return m


def decompose(model, n_samples=400_000):
    torch.manual_seed(9999)
    x, t = generate_batch(n_samples, F, P)
    with torch.no_grad():
        err4 = ((model(x) - t) ** 4).mean(dim=1)   # per-sample L4 averaged over features
    k = (x != 0).sum(dim=1).cpu().numpy()          # active count per sample
    err4 = err4.cpu().numpy()

    out = {}
    max_k = min(8, int(k.max()))
    total_mean = err4.mean()
    for kv in range(0, max_k + 1):
        mask = k == kv
        p_k = mask.mean()
        if mask.sum() == 0:
            continue
        mean_loss = err4[mask].mean()
        contribution = p_k * mean_loss
        out[kv] = dict(p_k=p_k, mean_loss=mean_loss, contribution=contribution,
                       frac_of_total=contribution / total_mean if total_mean > 0 else 0)
    out["total"] = total_mean
    return out


if __name__ == "__main__":
    results = {}
    for n in [2, 3, 4, 5]:
        print(f"\n=== n={n} neurons ===")
        m = get_model(n)
        d = decompose(m)
        results[n] = d
        print(f"  total L4: {d['total']:.4e}")
        print(f"  {'k':>3} {'P(k)':>7} {'L|k':>11} {'contrib':>11} {'frac':>6}")
        for kv, info in d.items():
            if kv == "total": continue
            print(f"  {kv:>3} {info['p_k']:>7.4f} {info['mean_loss']:>11.4e} "
                  f"{info['contribution']:>11.4e} {info['frac_of_total']:>6.2%}")

    # Plot: fraction of total loss contributed by each k, for each n
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ks = sorted({k for r in results.values() for k in r if k != "total"})
    width = 0.2
    for i, n in enumerate([2, 3, 4, 5]):
        fracs = [results[n].get(k, {}).get("frac_of_total", 0) for k in ks]
        axes[0].bar(np.array(ks) + i * width - 1.5 * width, fracs, width, label=f"n={n}")
    axes[0].set_xlabel("k = active features per sample")
    axes[0].set_ylabel("fraction of total L4 loss")
    axes[0].set_title("Loss contribution by number of active features")
    axes[0].legend()
    axes[0].set_xticks(ks)

    # Plot: L | k (conditional loss) on log scale
    for n in [2, 3, 4, 5]:
        ks_n = [k for k in results[n] if k != "total" and k > 0]
        vals = [results[n][k]["mean_loss"] for k in ks_n]
        axes[1].semilogy(ks_n, vals, "o-", label=f"n={n}")
    axes[1].set_xlabel("k = active features per sample")
    axes[1].set_ylabel("L4 | k (conditional)")
    axes[1].set_title("Loss given k active features")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/loss_by_k_active.png", dpi=120)
    print("\nSaved figures/loss_by_k_active.png")
