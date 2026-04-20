# Step 2: CiS model with fixed random embedding

## Setup

Add a random embedding E (fixed, D × F) and unembedding E^T (F × D). The model becomes

    r   = E @ x            (B, D)
    z   = W_in @ r         (B, n)
    h   = ReLU(z)
    r'  = W_out @ h        (B, D)
    y   = E^T @ r'         (B, F)

Trainable: W_in (n × D), W_out (D × n). E is fixed at initialisation.

Tested:
- D = F, E random orthogonal (square invertible E)
- D = 2F, 4F, 10F with E having random unit-norm *columns*
- Smallest non-trivial case first: F=20, n=2 (as in the plain-model analysis)
- Also F=20, n=5 to confirm
- L^2 and L^4 loss

## Mathematical note on equivalence

If E has full column rank F (true w.p. 1 for Gaussian and D ≥ F), the
embedded architecture is **function-class equivalent** to the plain model:
any W_in_eff (n × F) can be realised as W_in @ E with W_in = W_in_eff @ E^+,
and any W_out_eff (F × n) can be realised as E^T @ W_out with
W_out = ((E^T)^+) @ W_out_eff = E @ (E^T E)^{-1} @ W_out_eff. So the set of
functions representable is identical to that of the plain model.

Note the subtlety: the "unembedding = E^T" choice is strictly the gauge
inverse of the embedding only when E has orthonormal columns (E^T E = I),
i.e. when D = F with random-orthogonal E. For D > F with unit-*column* E,
E^T E ≠ I and the transpose-as-unembedding is a specific structural
constraint rather than a pure basis change — the optimizer must navigate
a non-trivial metric twist to reach the plain-model optimum. Empirically
it does, to within ~1% on α and partition structure.

## Result (F=20, n=2, L^4)

Per-feature MSE (when feature is active):

| config                       | mean    | std/mean |
|------------------------------|---------|----------|
| plain `small_20f_2n_L4`       | 0.152   | small    |
| embed D=F=20 orthogonal       | 0.158   | 2%       |
| embed D=40 unit-cols          | 0.158   | 2%       |
| embed D=80 unit-cols          | 0.157   | 2%       |

All within ~5% of the plain model's performance. Uniformity across features is
preserved.

Effective single-feature response `R_eff[:, j] = E^T @ W_out @ ReLU(W_in @ E[:,j])`:

| config                       | α (mean_diag × F/n) | ||sR-I||² | F-n |
|------------------------------|---------------------|-----------|-----|
| plain `small_20f_2n_L4`       | 3.243               | 18.00     | 18  |
| embed D=20 orth               | 3.242               | 18.00     | 18  |
| embed D=40 unit               | 3.239               | 18.01     | 18  |
| embed D=80 unit               | 3.247               | 18.00     | 18  |

α agrees to 3 decimals. All embedded models saturate the rank-n Frobenius
bound identically to the plain one.

**Effective codeword partition.** Each feature's effective codeword is the
sign pattern of `W_in @ E[:, j]` (an n-dim vector). For F=20, n=2:

| config         | (1,1) count | (1,0) count | (0,1) count | (0,0) count |
|----------------|-------------|-------------|-------------|-------------|
| plain          | 6           | 7           | 7           | 0           |
| embed D=20 orth | 6           | 7           | 7           | 0           |
| embed D=40 unit | 7           | 6           | 7           | 0           |
| embed D=80 unit | 6           | 7           | 7           | 0           |

**The exact same (6, 7, 7) partition emerges**, independent of D and E_kind.
This is the partition derived in closed form in `closed_form.tex` for the
plain model. The embedded model reaches the same solution expressed in
rotated coordinates.

## Result (F=20, n=5, L^4)

Per-feature MSE ~0.089 across plain and embedded (up to D=200); α = 2.03–2.04
(plain: 2.045, within 1%). Frobenius bound saturated (F-n=15, ||sR-I||² = 15.0).

Effective codeword count: 19–20 unique codewords (embedded) vs 18 (plain).
Slight variation; all configs fall in the "roughly one codeword per feature"
regime.

## L^2 baseline

At F=20, n=2, L^2:
- per-feat MSE mean 0.27, min 0.11, max 0.34 (naive-like, very non-uniform)
- diag(R_eff) mean 0.10, std 0.12 (not uniform)
- effective codeword distribution is heavily imbalanced (e.g., 15, 3, 2)
- α = 1.02 (the L^2-optimal single-feature scale, as expected)

Same as plain-model L^2: embedded L^2 is naive-like, embedded L^4 is CiS.

## Conclusion

Adding the random embedding does not materially change the mechanism. The
L^4-trained embedded model recovers the combinatorial ReLU coding of the plain
model, expressed in rotated coordinates via the effective weights `W_in @ E`
and `E^T @ W_out`. This is the "clean result" the user flagged as a possible
outcome.

**Sharper statement.** In the n=2 case, the sorted angles of `W_in_eff`
columns match the plain-model sorted angles to ~0.01 rad for D=F orthogonal
and D=80 Gaussian unit-col, and to ~0.1 rad for D=40 Gaussian unit-col (worse
conditioning). The embedded model reaches the same 2-frame (same three
clusters of column directions), just expressed in a randomized basis of R^n.

**Caveats**:
- "Same mechanism" is verified at the level of α, Frobenius-bound saturation,
  codeword-partition sizes, and (for n=2) sorted-angle geometry. I did not
  Procrustes-align and compare full Gram matrices between plain and embedded,
  so I cannot rule out that the networks find different tight frames that
  happen to produce identical α and partition counts. My prior is that they
  don't (the plain-model optimum is plausibly unique up to gauge, per the
  closed-form ansatz), but this is not proven.
- D=40 unit-col is noticeably less clean than D=F orth or D=80. `diag(R_eff)`
  std is ~0.011 vs ~0.005 for the other two. This is consistent with
  worse-conditioned E making optimisation slightly harder, but I didn't
  investigate in detail.
- Single seed per config.

**Practical implication for "toy model reliability".** If one wants a toy model
of CiS robust to embedding, the D×F-embedded model with L^4 loss works. Its
mechanism is essentially identical to the plain model's — the embedding
doesn't help distinguish mechanisms, nor does it break them.

## Open / unsure

- I did not run a seed sweep on embedded models (only one seed per config).
  The partition (6,7,7) for F=20,n=2 reproduces in every seed I looked at, but
  I did not test systematically. The closed-form ansatz in `closed_form.tex`
  predicts this partition is the optimum, so robustness across seeds is
  expected but unverified here.
- I did not test D << F (dimension reduction). That would break gauge
  equivalence and could change the story, but it is not what the user asked.
- I did not investigate whether larger D hurts optimization (it would have
  more parameters in the null space of E, which train but are functionally
  irrelevant). At D=200 (10×F), training still works fine; could become an
  issue at much larger D.

## Files

- `embedded_train.py` — training script
- `analyze_embedded.py` — analysis script
- `weights/embed_20f_{2,5}n_D{20,40,80,200}_{orth,unit}_L{2,4}.pt`
