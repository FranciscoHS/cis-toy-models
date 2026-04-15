"""
Trace the ansatz fit loss over training steps for n_A=6, to check convergence.
Also compare to:
 - the trained model's MC loss (reference),
 - the 'projected' ansatz: take trained weights, average within groups to enforce
   the archetype structure exactly, measure its MC loss. This separates ansatz
   underfit from optimization underfit.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from ansatz_fit import (
    F, p, device, generate_batch, build_W_from_ansatz, forward, mc_loss,
)

BATCH = 4096
STEPS = 500
LR = 3e-2
LR_FINAL = 1e-4
EVAL_EVERY = 10
N_A = 6


def fit_with_trace(n_A, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    init = torch.tensor([0.46, 0.545, -0.02, 0.34, 0.60, -0.25], device=device)
    params = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=LR)
    gamma = (LR_FINAL / LR) ** (1.0 / max(steps - 1, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    # Fixed held-out eval set: use it at every checkpoint so trace reflects
    # parameter change, not evaluation noise.
    torch.manual_seed(9999)
    x_eval, t_eval = generate_batch(200_000, F, p)

    trace_steps, trace_losses = [], []
    for step in range(steps):
        x, t = generate_batch(BATCH, F, p)
        W_in, W_out = build_W_from_ansatz(params, n_A, F)
        y = forward(W_in, W_out, x)
        loss = ((y - t) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % EVAL_EVERY == 0 or step == steps - 1:
            with torch.no_grad():
                W_in, W_out = build_W_from_ansatz(params, n_A, F)
                y_eval = forward(W_in, W_out, x_eval)
                lval = ((y_eval - t_eval) ** 4).mean().item()
            trace_steps.append(step)
            trace_losses.append(lval)
    return params.detach().cpu().numpy(), np.array(trace_steps), np.array(trace_losses)


def project_trained_onto_ansatz():
    sd = torch.load("weights/small_20f_2n_L4.pt", map_location=device, weights_only=True)
    W_in = sd["W_in"].cpu().numpy()
    W_out = sd["W_out"].cpu().numpy()
    signs = np.sign(W_in)
    A = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, 1)]
    B = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, -1)]
    C = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (-1, 1)]
    a_A = W_in[:, A].mean()                  # avg over both entries and features
    a_B1 = W_in[0, B].mean()
    a_B2 = W_in[1, B].mean()
    o_A = W_out[A, :].mean()
    o_B1 = W_out[B, 0].mean()
    o_B2 = W_out[B, 1].mean()
    params = torch.tensor([a_A, a_B1, a_B2, o_A, o_B1, o_B2], device=device, dtype=torch.float32)

    # Build using the SAME ordering as the fit (A, B, C blocks). We need to
    # reorder trained weights to same block ordering to compare losses on equal
    # footing, but for loss under i.i.d. inputs it does not matter.
    W_in_p, W_out_p = build_W_from_ansatz(params, len(A), F)
    loss_projected = mc_loss(W_in_p, W_out_p, n_batches=20, batch=16384)

    W_in_t = torch.from_numpy(W_in).to(device)
    W_out_t = torch.from_numpy(W_out).to(device)

    # Use a single large eval set for both, same samples, for apples-to-apples.
    torch.manual_seed(9999)
    x_eval, t_eval = generate_batch(200_000, F, p)
    with torch.no_grad():
        loss_trained = ((forward(W_in_t, W_out_t, x_eval) - t_eval) ** 4).mean().item()
        loss_projected = ((forward(W_in_p, W_out_p, x_eval) - t_eval) ** 4).mean().item()

    return loss_trained, loss_projected, params.cpu().numpy()


if __name__ == "__main__":
    print(f"Device: {device}")
    L_trained, L_projected, proj_params = project_trained_onto_ansatz()
    print(f"\nTrained MC loss:                 {L_trained:.6e}")
    print(f"Trained-projected-onto-ansatz:   {L_projected:.6e}  "
          f"(ratio {L_projected/L_trained:.4f})")
    print(f"Projected params (a_A, a_B1, a_B2, o_A, o_B1, o_B2): "
          + " ".join(f"{v:+.4f}" for v in proj_params))

    print(f"\nTracing ansatz fit for n_A={N_A} over {STEPS} steps...")
    final_params, steps, losses = fit_with_trace(N_A, steps=STEPS)
    print(f"Final fitted params: " + " ".join(f"{v:+.4f}" for v in final_params))
    print(f"Final fit loss: {losses[-1]:.6e}  (ratio to trained {losses[-1]/L_trained:.4f})")
    print(f"Min fit loss during trace: {losses.min():.6e}")

    # Plot
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(steps, losses, label=f"ansatz fit (n_A={N_A})")
    plt.axhline(L_trained, color="k", ls="--", label="trained model")
    plt.axhline(L_projected, color="C1", ls=":", label="trained, projected onto ansatz")
    plt.xlabel("step")
    plt.ylabel("L_mean (L4 per-entry)")
    plt.title("Ansatz fit convergence (n_A=6)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ansatz_trace.png", dpi=120)
    print("\nSaved figures/ansatz_trace.png")
