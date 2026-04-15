import torch, numpy as np

W = torch.load("weights/linear_20f_2n_L4.pt", map_location="cpu")
A = (W["W_out"] @ W["W_in"]).numpy()  # rank 2

# Each column j is A @ e_j = the network's signature for feature j active at +1.
# These 20 vectors live in a 2D subspace (the column space of A). Project them
# into 2D coordinates via SVD.
U, S, Vh = np.linalg.svd(A, full_matrices=False)
print(f"singular values: {S[:5].round(3)}")  # should show rank 2

# 2D coordinates of each column in col(A) basis
coords = (U[:, :2].T @ A).T  # (20, 2)
# Normalize by singular scale for visualization
norms = np.linalg.norm(coords, axis=1)
angles = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
print(f"\nColumn norms (20 signatures, should be ~equal):")
print(f"  mean={norms.mean():.3f}  std={norms.std():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")
print(f"\nColumn angles in col(A) (degrees), sorted:")
sorted_idx = np.argsort(angles)
for i in sorted_idx:
    print(f"  feat {i:>2}: angle={angles[i]:+7.1f}°  norm={norms[i]:.3f}")

# Pairwise angular gap
sorted_angles = np.sort(angles)
gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360]]))
print(f"\nAngular gaps between consecutive features (sorted):")
print(f"  {np.sort(gaps).round(1)}")
print(f"  mean={gaps.mean():.1f}°  min={gaps.min():.1f}°  max={gaps.max():.1f}°")

# Pairwise cosines
from itertools import combinations
coss = []
for i, j in combinations(range(20), 2):
    c = np.dot(coords[i], coords[j]) / (norms[i] * norms[j])
    coss.append(c)
coss = np.array(coss)
print(f"\nPairwise cosines of column signatures:")
print(f"  max = {coss.max():.3f}  (closest pair)")
print(f"  min = {coss.min():.3f}")
