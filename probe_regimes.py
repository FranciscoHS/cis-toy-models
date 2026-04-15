"""
Probe the 20f/2n L4 model as a piecewise-linear map with 4 regions
(both neurons on/off, only 1, only 2).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(0)
W = torch.load("weights/small_20f_2n_L4.pt", map_location="cpu")
W_in = W["W_in"]   # (2, 20)
W_out = W["W_out"] # (20, 2)
n_feat = 20

print("W_in (2 x 20):")
print(np.array2string(W_in.numpy(), precision=3, suppress_small=True))
print("\nW_out (20 x 2):")
print(np.array2string(W_out.numpy(), precision=3, suppress_small=True))

# Per-feature neuron weights and their signs.
print("\nPer-feature encoding (W_in columns) and their signs:")
print(f"{'feat':>4} {'W_in[0,i]':>10} {'W_in[1,i]':>10}  sign-pair")
for i in range(n_feat):
    a, b = W_in[0, i].item(), W_in[1, i].item()
    sp = f"({'+' if a>0 else '-'},{'+' if b>0 else '-'})"
    print(f"{i:>4} {a:>10.4f} {b:>10.4f}  {sp}")

# For a single-feature input x = v * e_i, pre-activations are v * W_in[:, i].
# ReLU fires on neuron j iff v * W_in[j, i] > 0.
#   v>0: neurons fire where W_in[:, i] > 0
#   v<0: neurons fire where W_in[:, i] < 0
print("\nWhich neurons fire for single-feature input x = v * e_i?")
print(f"{'feat':>4}  v>0 fires       v<0 fires")
for i in range(n_feat):
    a, b = W_in[0, i].item(), W_in[1, i].item()
    pos = [j for j, w in enumerate([a, b]) if w > 0]
    neg = [j for j, w in enumerate([a, b]) if w < 0]
    print(f"{i:>4}  {str(pos):<15} {str(neg):<15}")

# The four piecewise-linear maps.
e1 = torch.tensor([[1.0], [0.0]])  # mask neuron 1 on only
e2 = torch.tensor([[0.0], [1.0]])
e12 = torch.tensor([[1.0], [1.0]])
e0 = torch.tensor([[0.0], [0.0]])

def linmap(mask):
    # y_hat = W_out @ diag(mask) @ W_in
    M = W_out @ (mask * W_in)
    return M

A_both = linmap(e12)   # both on
A_n1 = linmap(e1)      # only neuron 1
A_n2 = linmap(e2)      # only neuron 2
A_off = linmap(e0)     # both off -> zero

print("\nDiagonals of the four linear maps (y_hat_i vs x_i):")
for name, A in [("both", A_both), ("only 1", A_n1), ("only 2", A_n2), ("off", A_off)]:
    d = torch.diag(A).numpy()
    print(f"  {name:>8}: mean diag = {d.mean():.3f}, min={d.min():.3f}, max={d.max():.3f}")

# Empirical: scan v for each single-feature input, record which neurons fire.
print("\nSingle-feature scan v in [-1,1], regime hit for each feature (symbolic):")
# Already derived above from signs; double-check with forward pass.
scan_vs = torch.linspace(-1, 1, 11)
for i in range(n_feat):
    x = torch.zeros(len(scan_vs), n_feat)
    x[:, i] = scan_vs
    pre = (W_in @ x.T).T  # (V, 2)
    fires = (pre > 0).int().numpy()
    # regimes: (0,0) off, (1,0) n1, (0,1) n2, (1,1) both
    regs = ["".join(map(str, r)) for r in fires]
    print(f"  feat {i:>2}: " + " ".join(regs) + f"  (W_in col = {W_in[:,i].numpy().round(3)})")

# Regime distribution on the training distribution.
N = 100_000
p = 0.02
mask = (torch.rand(N, n_feat) < p).float()
vals = torch.rand(N, n_feat) * 2 - 1
x = mask * vals
pre = (W_in @ x.T).T  # (N, 2)
fires = (pre > 0).int()
codes = fires[:, 0] * 2 + fires[:, 1]  # 0 off, 1 only-n2, 2 only-n1, 3 both
counts = torch.bincount(codes, minlength=4).numpy()
print(f"\nRegime frequencies over {N} training samples (p={p}):")
labels = ["off (0,0)", "only n2 (0,1)", "only n1 (1,0)", "both (1,1)"]
for lbl, c in zip(labels, counts):
    print(f"  {lbl:>14}: {c:>7} ({100*c/N:5.2f}%)")

# Fraction of samples with 0, 1, 2+ active features
n_active = mask.sum(dim=1).int()
for k in range(4):
    frac = (n_active == k).float().mean().item()
    print(f"  samples with {k} active feat{'s' if k!=1 else ''}: {100*frac:.2f}%")
print(f"  samples with 3+ active feats: {100*(n_active >= 3).float().mean().item():.2f}%")
