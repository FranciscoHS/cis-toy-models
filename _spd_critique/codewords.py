"""Analyze codeword structure of plain_20f_5n and plain_20f_2n target models."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import numpy as np

REPO = Path("/home/francisco/Projects/vibeathon")

def codewords(weights_path, n_features, d_mlp):
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    W_in = sd["W_in"].numpy()   # (n, F)
    W_out = sd["W_out"].numpy() # (F, n)
    assert W_in.shape == (d_mlp, n_features), W_in.shape
    assert W_out.shape == (n_features, d_mlp), W_out.shape

    # For each feature j: which neurons does it activate (positively)?
    # Codeword for feature j = sign pattern of W_in[:, j]
    # Normalize per-feature: magnitude doesn't matter, just sign pattern of positive entries
    print(f"\n=== {weights_path} ===")
    print(f"W_in shape {W_in.shape}, W_out shape {W_out.shape}")
    print(f"W_in per-feature L2 norms: {np.linalg.norm(W_in, axis=0).round(3)}")

    # Codeword: sign(W_in[:, j]) with small threshold
    thresh = 0.1
    codes_signed = np.sign(W_in) * (np.abs(W_in) > thresh * np.abs(W_in).max())
    codes_signed = codes_signed.T  # (F, n)
    # Unsigned codeword: positive-part support pattern (which neurons does feat j
    # activate through the ReLU positively?)
    # Only positive values matter after ReLU. But the ReLU activates when W_in @ x > 0.
    # For one-hot x=e_j, h = ReLU(W_in[:, j]). So the active neurons are those where
    # W_in[:, j] > 0.
    pos_codes = (W_in.T > 0.05 * W_in.T.max(axis=1, keepdims=True)).astype(int)
    # Full signed codeword: useful because features with flipped-sign coefficients
    # still share the same "active neuron set" after ReLU
    print("\nSigned codeword (sign W_in[:, j] > 0.1*max, so +1, 0, or -1):")
    for j in range(n_features):
        print(f"  feat {j:2d}: {codes_signed[j].astype(int).tolist()}  "
              f"pos_codeword: {pos_codes[j].tolist()}  "
              f"||W_in[:,j]||={np.linalg.norm(W_in[:, j]):.3f}")

    # Group features by positive codeword (the set of neurons that activate after ReLU)
    from collections import defaultdict
    groups_pos = defaultdict(list)
    for j in range(n_features):
        groups_pos[tuple(pos_codes[j])].append(j)
    print(f"\nNumber of distinct POSITIVE (post-ReLU) codewords: {len(groups_pos)}")
    for k, v in sorted(groups_pos.items()):
        print(f"  {k}  -> features {v}")

    groups_signed = defaultdict(list)
    for j in range(n_features):
        groups_signed[tuple(codes_signed[j].astype(int))].append(j)
    print(f"\nNumber of distinct SIGNED codewords: {len(groups_signed)}")
    for k, v in sorted(groups_signed.items()):
        print(f"  {k}  -> features {v}")

    return codes_signed, pos_codes, groups_pos, groups_signed

codewords(REPO / "weights/small_20f_5n_L4.pt", 20, 5)
codewords(REPO / "weights/small_20f_2n_L4.pt", 20, 2)
