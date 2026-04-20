"""Extra checks:
1. Is E^T really correct as an "unembedding"? The canonical choice would be
   E^+ (pseudoinverse), not E^T. E^T equals E^+ only when columns of E are
   orthonormal, i.e., E^T E = I_F. Check how well this holds for each config.
2. Compare sorted-angle distributions (the 'shape' of W_in_eff as a 2-frame).
3. Verify that E has full column rank (so D>=F matters).
4. Check: codeword partition from sign of W_in @ E[:, j]. For D=40 model the
   first few angles show a spread (35-50deg vs 43-46deg in others); see if it
   corresponds to one or more features being only marginally above threshold.
"""
import torch
import numpy as np
from collections import Counter

def load(p):
    return torch.load(p, map_location="cpu")

configs = [
    "embed_20f_2n_D20_orth_L4",
    "embed_20f_2n_D40_unit_L4",
    "embed_20f_2n_D80_unit_L4",
]

for c in configs:
    d = load(f"weights/{c}.pt")
    E = d["E"].numpy()
    D, F = E.shape
    ETE = E.T @ E
    # Deviation from identity
    dev = np.linalg.norm(ETE - np.eye(F), "fro")
    # Column norms
    col_norms = np.linalg.norm(E, axis=0)
    # Rank
    sv = np.linalg.svd(E, compute_uv=False)
    print(f"\n{c}: E shape {E.shape}")
    print(f"  ||E^T E - I||_F = {dev:.4f}")
    print(f"  col norms: min={col_norms.min():.4f} max={col_norms.max():.4f}")
    print(f"  singular values of E: min={sv.min():.4f} max={sv.max():.4f}  "
          f"cond={sv.max()/sv.min():.2f}")
    # cond number tells us how far from orthonormal E is.

    # How close is the effective solution to the plain model's mechanism?
    # Check the magnitudes of W_in_eff columns vs sign-pattern margin
    W_in = d["W_in"].numpy()
    W_in_eff = W_in @ E  # n x F
    # Margin: min over j of min |W_in_eff[k, j]|
    min_abs = np.min(np.abs(W_in_eff), axis=0)
    print(f"  min |W_in_eff[k,j]| over features: {np.sort(min_abs)}")

# Gauge-equivalence sanity check:
# If we set W_in = W_in_eff_plain @ E^+ and W_out = E^+.T @ W_out_plain,
# then W_in @ E = W_in_eff_plain @ E^+ @ E. If E has orthonormal columns, E^+ @ E = I.
# Otherwise, only the projection onto Row(E) is preserved.
# Let's check: does W_in_eff_plain @ E^+ @ E == W_in_eff_plain?
# E^+ @ E = P_{row(E)} (projects onto the row space of E, which is all of R^F if rank(E) = F).
# So *if rank(E) = F*, this holds. Gauge equivalence requires full col rank (D >= F).
# For D < F, it would fail.
print("\n=== Gauge equivalence exactness (E^+ E = I_F requires rank F) ===")
for c in configs:
    d = load(f"weights/{c}.pt")
    E = d["E"].numpy()
    D, F = E.shape
    Epinv = np.linalg.pinv(E)
    back = Epinv @ E
    print(f"{c}: ||E^+ E - I_F||_F = {np.linalg.norm(back - np.eye(F), 'fro'):.2e}")
