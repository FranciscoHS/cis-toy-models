"""
Sweep the A-group diagonal alpha_A = 2 * o_A * a_A over a range of target values.
For each target, re-optimize the remaining 4 continuous parameters of the 5-param
ansatz (with n_A=6 fixed). Plot L4 loss vs alpha_A.

Expected: U-shape, minimum near the observed alpha_A ~ 0.33.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from ansatz_fit import F, p, device, generate_batch, forward

BATCH = 4096
STEPS = 500
LR = 3e-2
LR_FINAL = 1e-4
N_A = 6


def build_W(params, alpha_A, n_A):
    """params = (a_A, a_B1, a_B2, o_B1, o_B2). o_A is derived from alpha_A = 2 o_A a_A."""
    a_A, a_B1, a_B2, o_B1, o_B2 = params[0], params[1], params[2], params[3], params[4]
    o_A = alpha_A / (2.0 * a_A)
    n_B = (F - n_A) // 2
    W_in = torch.zeros(2, F, device=device, dtype=params.dtype)
    W_out = torch.zeros(F, 2, device=device, dtype=params.dtype)
    iA, iB, iC = slice(0, n_A), slice(n_A, n_A + n_B), slice(n_A + n_B, F)
    W_in[0, iA] = a_A; W_in[1, iA] = a_A
    W_in[0, iB] = a_B1; W_in[1, iB] = a_B2
    W_in[0, iC] = a_B2; W_in[1, iC] = a_B1
    W_out[iA, 0] = o_A; W_out[iA, 1] = o_A
    W_out[iB, 0] = o_B1; W_out[iB, 1] = o_B2
    W_out[iC, 0] = o_B2; W_out[iC, 1] = o_B1
    return W_in, W_out


def fit_at_alpha(alpha_A, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    # Init from empirical
    init = torch.tensor([0.46, 0.545, -0.02, 0.60, -0.25], device=device)
    params = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=LR)
    gamma = (LR_FINAL / LR) ** (1.0 / max(steps - 1, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    for step in range(steps):
        x, t = generate_batch(BATCH, F, p)
        W_in, W_out = build_W(params, alpha_A, N_A)
        y = forward(W_in, W_out, x)
        loss = ((y - t) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # Eval on fixed held-out set
    torch.manual_seed(9999)
    x_eval, t_eval = generate_batch(500_000, F, p)
    with torch.no_grad():
        W_in, W_out = build_W(params, alpha_A, N_A)
        L = ((forward(W_in, W_out, x_eval) - t_eval) ** 4).mean().item()
    return L, params.detach().cpu().numpy()


if __name__ == "__main__":
    # Trained-model reference
    sd = torch.load("weights/small_20f_2n_L4.pt", map_location=device, weights_only=True)
    torch.manual_seed(9999)
    x_eval, t_eval = generate_batch(500_000, F, p)
    with torch.no_grad():
        L_trained = ((forward(sd["W_in"].float(), sd["W_out"].float(), x_eval) - t_eval) ** 4).mean().item()
    print(f"Trained L4 loss: {L_trained:.4e}")

    alphas = np.concatenate([np.linspace(0.05, 0.55, 15), np.linspace(0.60, 0.95, 5)])
    losses = []
    for a in alphas:
        L, p_fit = fit_at_alpha(a)
        losses.append(L)
        print(f"alpha_A={a:.3f}  L={L:.4e}  ratio={L/L_trained:.3f}  "
              f"params={np.round(p_fit,3).tolist()}")

    losses = np.array(losses)
    best_i = np.argmin(losses)
    print(f"\nBest alpha_A: {alphas[best_i]:.3f}  L={losses[best_i]:.4e}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(alphas, losses, "o-", label="ansatz, 4 free params re-optimized per alpha_A")
    plt.axhline(L_trained, color="k", ls="--", label=f"trained model ({L_trained:.3e})")
    plt.axvline(alphas[best_i], color="C2", ls=":", alpha=0.6,
                label=f"best alpha_A={alphas[best_i]:.3f}")
    plt.axvline(0.33, color="C3", ls=":", alpha=0.6, label="observed alpha_A~0.33")
    plt.xlabel(r"$\alpha_A = 2 o_A a_A$ (A-group diagonal)")
    plt.ylabel("L_mean (L4)")
    plt.title(r"Loss vs constrained A-diagonal, re-optimizing other params")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figures/sweep_alpha.png", dpi=120)
    print("Saved figures/sweep_alpha.png")
