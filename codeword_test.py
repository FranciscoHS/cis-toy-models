"""
Train 20f models with n=2, 3, 4, 5 neurons under L4 and inspect the
codewords (sign pattern of each feature's W_in column).

Prediction from the 'ReLU codeword' framing:
- n=2: 2^2 - 1 = 3 useful codewords. 20 features split 6-7-7.
- n=3: 2^3 - 1 = 7 useful codewords. 20/7 ~ 2.86 features per codeword.
- n=4: 2^4 - 1 = 15. 20/15 ~ 1.33.
- n=5: 2^5 - 1 = 31 >= 20. Each feature could get its own codeword.
"""

import torch
import numpy as np
from collections import Counter
from small_models import SimpleMLP, generate_batch, DEVICE

F = 20
P = 0.02
BATCH = 2048
STEPS = 10000
LR = 3e-3


def train(n_neurons, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SimpleMLP(F, n_neurons).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for step in range(STEPS):
        x, y = generate_batch(BATCH, F, P)
        y_hat = m(x)
        loss = (torch.abs(y_hat - y) ** 4).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m


def codewords(W_in, threshold=0.05):
    """For each feature (column of W_in), produce a sign pattern where small
    magnitudes are treated as 0."""
    out = []
    for j in range(W_in.shape[1]):
        col = W_in[:, j]
        pattern = tuple(int(np.sign(v)) if abs(v) > threshold else 0 for v in col)
        out.append(pattern)
    return out


def eval_loss(model, n_samples=200_000):
    torch.manual_seed(9999)
    x, t = generate_batch(n_samples, F, P)
    with torch.no_grad():
        return ((model(x) - t) ** 4).mean().item()


if __name__ == "__main__":
    print(f"Device: {DEVICE}, F={F}, p={P}, steps={STEPS}")
    results = {}
    for n in [2, 3, 4, 5]:
        print(f"\n=== n={n} neurons ===")
        m = train(n)
        L = eval_loss(m)
        W_in = m.W_in.detach().cpu().numpy()
        W_out = m.W_out.detach().cpu().numpy()

        # Magnitudes: what's the typical "big" vs "small"?
        mags = np.abs(W_in).flatten()
        print(f"  W_in |mag| dist: min={mags.min():.3f} med={np.median(mags):.3f} max={mags.max():.3f}")

        # Codewords at a couple of thresholds
        for thr in [0.05, 0.1, 0.2]:
            cw = codewords(W_in, threshold=thr)
            counts = Counter(cw)
            print(f"  thr={thr}: {len(counts)} distinct codewords, dist: {dict(counts)}")

        print(f"  L4: {L:.4e}")
        # Naive baseline: implement first n features
        W_in_naive = np.zeros((n, F)); W_out_naive = np.zeros((F, n))
        for k in range(n):
            W_in_naive[k, k] = 1.0; W_out_naive[k, k] = 1.0
        W_in_naive_t = torch.from_numpy(W_in_naive).float().to(DEVICE)
        W_out_naive_t = torch.from_numpy(W_out_naive).float().to(DEVICE)
        torch.manual_seed(9999)
        x_n, t_n = generate_batch(200_000, F, P)
        with torch.no_grad():
            L_naive = (((W_out_naive_t @ torch.relu(W_in_naive_t @ x_n.T)).T - t_n) ** 4).mean().item()
        print(f"  naive (first {n} features): {L_naive:.4e}  (ratio {L_naive/L:.2f})")

        results[n] = dict(L=L, L_naive=L_naive, W_in=W_in, W_out=W_out)

    print("\n=== Summary ===")
    print(f"{'n':>3} {'L4':>10} {'L_naive':>10} {'naive/L':>8}")
    for n, r in results.items():
        print(f"{n:>3} {r['L']:>10.3e} {r['L_naive']:>10.3e} {r['L_naive']/r['L']:>8.2f}")
