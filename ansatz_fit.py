"""
Fit the 5-parameter archetype ansatz to the L4 objective and compare against
the trained 20f/2n L4 model.

Ansatz:
  W_in col j in A: (a_A, a_A),       W_out row j in A: (o_A, o_A)
  W_in col j in B: (a_B1, a_B2),     W_out row j in B: (o_B1, o_B2)
  W_in col j in C: (a_B2, a_B1),     W_out row j in C: (o_B2, o_B1)
With n_B = n_C = (F - n_A) / 2.

For each n_A in {0, 2, ..., F}, optimize (a_A, a_B1, a_B2, o_A, o_B1, o_B2)
directly via gradient descent on a large MC batch (no scale fix; the overall
scale is redundant but the optimum is well-defined on the MSE surface).

Compare the best-ansatz loss to the trained model's loss, and the best-ansatz
archetype weights to the trained archetype weights.
"""

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

F = 20
p = 0.02
BATCH = 4096
STEPS = 800
LR = 3e-2


def generate_batch(batch_size, n_features, p):
    mask = (torch.rand(batch_size, n_features, device=device) < p).float()
    values = torch.rand(batch_size, n_features, device=device) * 2 - 1
    x = mask * values
    t = torch.relu(x)
    return x, t


def build_W_from_ansatz(params, n_A, F):
    """params = (a_A, a_B1, a_B2, o_A, o_B1, o_B2). Returns W_in (2,F), W_out (F,2)."""
    a_A, a_B1, a_B2, o_A, o_B1, o_B2 = params
    n_B = (F - n_A) // 2
    assert n_A + 2 * n_B == F

    W_in = torch.zeros(2, F, device=device, dtype=params.dtype)
    W_out = torch.zeros(F, 2, device=device, dtype=params.dtype)

    idx_A = slice(0, n_A)
    idx_B = slice(n_A, n_A + n_B)
    idx_C = slice(n_A + n_B, F)

    W_in[0, idx_A] = a_A
    W_in[1, idx_A] = a_A
    W_in[0, idx_B] = a_B1
    W_in[1, idx_B] = a_B2
    W_in[0, idx_C] = a_B2
    W_in[1, idx_C] = a_B1

    W_out[idx_A, 0] = o_A
    W_out[idx_A, 1] = o_A
    W_out[idx_B, 0] = o_B1
    W_out[idx_B, 1] = o_B2
    W_out[idx_C, 0] = o_B2
    W_out[idx_C, 1] = o_B1

    return W_in, W_out


def forward(W_in, W_out, x):
    return (W_out @ torch.relu(W_in @ x.T)).T


def mc_loss(W_in, W_out, n_batches=20, batch=BATCH):
    total = 0.0
    n = 0
    for _ in range(n_batches):
        x, t = generate_batch(batch, F, p)
        y = forward(W_in, W_out, x)
        total += ((y - t) ** 4).mean().item() * batch * F
        n += batch * F
    return total / n  # L_mean (per-entry mean)


def fit_ansatz(n_A, seed=0):
    torch.manual_seed(seed)
    # Init from empirical ballpark values
    init = torch.tensor([0.46, 0.545, -0.02, 0.34, 0.60, -0.25], device=device)
    params = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    for step in range(STEPS):
        x, t = generate_batch(BATCH, F, p)
        W_in, W_out = build_W_from_ansatz(params, n_A, F)
        y = forward(W_in, W_out, x)
        loss = ((y - t) ** 4).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    final_loss = mc_loss(*build_W_from_ansatz(params, n_A, F), n_batches=10, batch=16384)
    return params.detach().cpu().numpy(), final_loss


def trained_loss():
    sd = torch.load("weights/small_20f_2n_L4.pt", map_location=device, weights_only=True)
    W_in = sd["W_in"].to(device)
    W_out = sd["W_out"].to(device)
    return mc_loss(W_in, W_out, n_batches=10, batch=16384), W_in.cpu().numpy(), W_out.cpu().numpy()


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"F={F}, p={p}, batch={BATCH}, steps={STEPS}")

    L_trained, W_in_t, W_out_t = trained_loss()
    print(f"\nTrained L4 model MC loss (L_mean): {L_trained:.6e}")

    # Extract trained archetype values for reference
    signs = np.sign(W_in_t)
    A_idx = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, 1)]
    B_idx = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (1, -1)]
    C_idx = [j for j in range(F) if tuple(signs[:, j].astype(int)) == (-1, 1)]
    print(f"Trained groups: |A|={len(A_idx)} |B|={len(B_idx)} |C|={len(C_idx)}")
    a_A_t = W_in_t[:, A_idx].mean()
    a_B1_t = W_in_t[0, B_idx].mean()
    a_B2_t = W_in_t[1, B_idx].mean()
    o_A_t = W_out_t[A_idx, :].mean()
    o_B1_t = W_out_t[B_idx, 0].mean()
    o_B2_t = W_out_t[B_idx, 1].mean()
    print(f"Trained archetypes: a_A={a_A_t:.3f} a_B1={a_B1_t:.3f} a_B2={a_B2_t:.3f} "
          f"o_A={o_A_t:.3f} o_B1={o_B1_t:.3f} o_B2={o_B2_t:.3f}")

    print(f"\nSweeping n_A and fitting ansatz:")
    print(f"{'n_A':>4} {'L_mean':>12} {'rel_to_trained':>15} "
          f"{'a_A':>7} {'a_B1':>7} {'a_B2':>8} {'o_A':>7} {'o_B1':>7} {'o_B2':>8}")

    results = []
    for n_A in range(0, F + 2, 2):
        params, loss = fit_ansatz(n_A)
        ratio = loss / L_trained
        results.append((n_A, loss, params))
        a_A, a_B1, a_B2, o_A, o_B1, o_B2 = params
        print(f"{n_A:>4} {loss:>12.6e} {ratio:>15.4f} "
              f"{a_A:>7.3f} {a_B1:>7.3f} {a_B2:>8.3f} {o_A:>7.3f} {o_B1:>7.3f} {o_B2:>8.3f}")

    best = min(results, key=lambda r: r[1])
    print(f"\nBest n_A: {best[0]}  L_ansatz={best[1]:.6e}  L_trained={L_trained:.6e}  "
          f"ratio={best[1] / L_trained:.4f}")
