"""
Measure on-feature vs off-feature outputs for the real model,
and compare against a hypothetical all-group-A version.
"""
import torch
import numpy as np

W = torch.load("weights/small_20f_2n_L4.pt", map_location="cpu")
W_in, W_out = W["W_in"], W["W_out"]   # (2,20), (20,2)
n_feat = 20

def summarize(A, name):
    A = A.numpy()
    diag = np.diag(A)
    off = A - np.diag(diag)
    off_rms_per_row = np.sqrt((off ** 2).sum(1) / (n_feat - 1))
    print(f"\n[{name}]")
    print(f"  diagonal:   mean={diag.mean():.3f}  min={diag.min():.3f}  max={diag.max():.3f}")
    print(f"  off-diag:   RMS per row mean={off_rms_per_row.mean():.4f}  max_abs={np.abs(off).max():.4f}")
    print(f"  ratio RMS(off) / mean(diag) = {off_rms_per_row.mean() / diag.mean():.3f}")

# Real "both on" map
A_both = W_out @ W_in
summarize(A_both, "REAL both-on (rank 2)")

# Real single-neuron maps
A_n1 = W_out @ (torch.tensor([[1.0],[0.0]]) * W_in)
A_n2 = W_out @ (torch.tensor([[0.0],[1.0]]) * W_in)
summarize(A_n1, "REAL only-n1 (rank 1)")
summarize(A_n2, "REAL only-n2 (rank 1)")

# Hypothetical: all features group A. Construct W_in' with both rows equal to
# the mean of rows 0 and 1 restricted to their positive entries (~0.5).
row_mag = 0.5
W_in_A = row_mag * torch.ones(2, n_feat)
# Pick the best W_out for this W_in to min ||W_out W_in_A - 0.5 I||_F
# (so we give the hypothetical its best shot at matching 0.5 I).
# With W_in_A = row_mag * 1 1^T, rank 1, W_out @ W_in_A = (W_out @ 1) (1^T) * row_mag.
# That's a rank-1 matrix c 1^T where c = row_mag * W_out.sum(1).
# Best rank-1 approx to 0.5 I has singular values sorted; best rank-1 is 0.5/n * 1 1^T.
# So the best rank-1 map has uniform entries ~0.5/20 = 0.025. Let's just compute it.
I = torch.eye(n_feat) * 0.5  # target (what "both on" in real model ~ achieves)
U, S, Vh = torch.linalg.svd(I)
best_rank1 = (U[:, :1] * S[:1]) @ Vh[:1, :]
summarize(best_rank1, "HYPOTHETICAL best rank-1 approx to 0.5 I (all-group-A ceiling)")

# Single-active-feature probe: inject x = 1.0 * e_j, show what all 20 outputs are.
print("\nReal model, x = +1.0 on feature j, outputs (diag entry, RMS off-diag):")
for j in [0, 2, 13]:  # one from each of A, B, A
    x = torch.zeros(n_feat); x[j] = 1.0
    y = (W_out @ torch.relu(W_in @ x))
    y = y.numpy()
    rms_off = np.sqrt(((y ** 2).sum() - y[j] ** 2) / (n_feat - 1))
    print(f"  feat {j:>2} (group {'A' if j in [0,6,10,13,14,19] else 'B' if j in [2,3,5,8,9,11,12] else 'C'}):"
          f"  y[{j}]={y[j]:+.3f}  RMS(others)={rms_off:.4f}")

# Same probe with best rank-1 map (the "all-group-A ceiling").
print("\nHypothetical rank-1 linear map (best all-group-A ceiling), x = +1.0 on feature j:")
for j in [0, 2, 13]:
    x = torch.zeros(n_feat); x[j] = 1.0
    y = (best_rank1 @ x).numpy()
    rms_off = np.sqrt(((y ** 2).sum() - y[j] ** 2) / (n_feat - 1))
    print(f"  feat {j:>2}:  y[{j}]={y[j]:+.3f}  RMS(others)={rms_off:.4f}")
