# Autonomous investigation report (2026-04-17)

## Tasks

1. Characterise the single-feature projection implemented by the L^4-trained
   CiS model at F=20, n=5 (and more broadly). Scan (F, n); look for a pattern
   and, if possible, a formula.
2. Add a fixed random Gaussian embedding/unembedding and see if L^4 still
   finds a good solution. Interpret if so.

Full step-wise findings are in `step1_findings.md` and `step2_findings.md`.
Code: `probe_onehot.py`, `analyze_R_structure.py`, `analyze_all.py`,
`visualize_R.py`, `train_limit_configs.py`, `embedded_train.py`,
`analyze_embedded.py`.

## Step 1: what the projection is

Let `R[:, j] := W_out @ ReLU(W_in[:, j])` be the single-feature response
matrix. Claim, supported by all L^4-trained configs I measured:

> **R ≈ α P**, where P is a rank-n orthogonal projection of R^F with
> **uniform diagonal P_jj = n/F**. Off-diagonals with distinct codewords
> cluster near ±(α × Welch), where Welch = sqrt(n(F-n) / (F²(F-1))) is the
> Welch lower bound on frame coherence.

Evidence:
- `||s R − I||_F² ≈ F − n` to within 0.5% for every trained config (this is
  the rank-n Frobenius lower bound; attaining it means R is as close as
  possible to I given the rank constraint).
- Top-n singular values of R are nearly equal (std/mean < 5%) and the rest
  are ~0, i.e., R is approximately rank-n and proportional to a projection.
- `diag(R)` has std/mean ~1–4% (for most configs), so the projection is
  approximately Parseval (all features project with equal squared norm into
  the image subspace).
- Off-diagonal magnitudes for different-codeword pairs match the Welch bound
  to ~10% (bimodal distribution peaked at ±α · Welch).

For F=20, n=5 specifically: diag(R) mean 0.511 (predicted α·n/F = 0.511 with
α=2.045); off-diagonals cluster at ±0.185 (predicted α·Welch = 0.203);
||s R − I||² = 15.02 vs F−n = 15.

Caveat: R is not exactly a scaled projection. It has systematic asymmetry
`||R − R^T||/||R|| ≈ 4–13%` and slight positive off-diagonal bias, which the
pure scaled-projection ansatz misses.

### Limits

- **F = n (trivial):** α = 1, R ≈ I, per-feature MSE ~10^{-4}. Model
  implements clean ReLU.
- **F = 2^n − 1 (one codeword per feature):** off-diagonals saturate the
  Welch bound to ~10%. Note real ETFs don't exist for (7, 3), (15, 4),
  (31, 5) — Gerzon bound F ≤ n(n+1)/2 fails — so the network approximates
  but cannot attain Welch.
- **F > 2^n − 1:** codewords shared; same-codeword features share a
  projection direction.

### Scaling of α

Define α := mean(diag R) · F/n. Single-feature L^4 + ETF assumption gives
the closed form

    β := α·n/F = 1 / (1 + [γ²/(F-1)]^{1/3}),   γ := (F-n)/n.

This is a clean upper bound on observed α and qualitatively tracks the
scaling, but systematically overpredicts α (by ~15% for F=2^n−1 configs, by
up to ~50% at F/n = 10, F large), because multi-feature inputs make the
interference term grow — so the optimiser settles at a smaller α than the
single-feature calculation suggests. Empirical p-sweep on (F=31, n=5) shows α
decreasing monotonically from 2.76 at p=0.005 to 1.44 at p=0.2, consistent
with this story.

A multi-feature extension of the analytical loss is tractable but I did not
derive it.

### Why "uniform diagonal" (i.e., Parseval)?

L^4 penalises outliers. If the projection had some P_jj larger than others
(some features over-represented), those would get low per-feature error,
but the features with small P_jj would get large error — and the L^4 cost
would be dominated by the worst-off feature. Uniformising the diagonal
minimises the worst case. L^2, in contrast, rewards any rank-n projection
equally (all give Frobenius cost F − n), so it's free to concentrate on a
subset — this is the naive / bimodal solution seen in L^2-trained models.

## Step 2: random embedding

Architecture:
    y = E^T @ W_out @ ReLU(W_in @ E @ x),     E fixed D × F,
with W_in (n × D) and W_out (D × n) trainable.

If D ≥ F and E has full column rank (trivial for Gaussian), the parameter
space is **function-class equivalent** to the plain model — any effective
W_in_eff = W_in @ E is realisable. So the embedded model cannot represent
anything the plain model can't.

**Result (L^4, F=20, n=2, D ∈ {20, 40, 80}):** α matches the plain model to
3 decimals, ||sR − I||² = F − n = 18 to identical precision, and the
effective codeword partition is the same (6, 7, 7) as the plain model
(matching `closed_form.tex`'s ansatz). Sorted angles of `W_in @ E` columns
align with the plain-model sorted angles to ≤0.01 rad for D=F orthogonal
and D=80 Gaussian unit-col (worse for D=40 — probably conditioning).

**Result (L^4, F=20, n=5):** α = 2.03–2.04 (plain 2.045), Frob bound
saturated identically, ~19–20 effective codewords (plain: 18). Essentially
the same mechanism.

**Result (L^2):** naive-like in both plain and embedded (non-uniform
per-feature error, a few features well-computed, most ignored).

**Conclusion.** Adding the embedding does not change the mechanism. The
L^4-trained embedded model recovers the combinatorial ReLU coding of the
plain model, expressed in rotated coordinates. This matches the "clean
result" the user flagged as one possible outcome.

Caveats:
- Single seed per config; I didn't run a seed sweep.
- Did not Procrustes-match full Gram matrices of plain vs embedded — only α,
  partition, sorted-angle comparisons for n=2. "Same tight frame" is not
  fully pinned down but is consistent with every comparison I did.
- E^T as unembedding is the true gauge inverse only for orthogonal E. For
  Gaussian unit-col with D > F, the architecture imposes a non-trivial
  metric twist; the optimiser navigates it.

## What I did not do / remaining questions

1. **Derive multi-feature correction to α.** Tractable; not done. Would
   sharpen the α formula significantly and probably explain the F-large
   saturation in the 10:1 expansion series.
2. **Is the projection subspace / tight frame determined uniquely by the
   task?** Seed sweep + Procrustes comparison would say so. I see suggestive
   evidence (the (6, 7, 7) partition is hit reliably for F=20, n=2) but
   didn't prove uniqueness.
3. **D < F (information-lossy embedding).** Not tested. Would break
   function-class equivalence. Interesting follow-up.
4. **Refined R ansatz** with asymmetric and rank-1 correction terms:
   R = α P + β · (u 1^T) + skew might close the gap to trained R exactly.
5. **Why α saturates at F/n = 10.** Scaling series up to 1000/100 shows α
   plateauing near 5; whether this is a real saturation or just slow growth
   I did not resolve.

## Confidence calibration

- **High confidence (very well-supported):**
  Rank-n Frobenius bound saturation. Top-n equal singular values. Uniform
  diagonal. The L^4-vs-L^2 difference (uniform vs bimodal per-feature error).
  The embedded model's α and partition matching the plain model.

- **Medium confidence (supported but not fully proven):**
  The "Parseval frame" label (works to ~2% for diagonal, 10% for off-diag
  magnitudes, modulo R asymmetry). The closed-form α formula (correct
  algebra for single-feature L^4; overshoots due to unmodelled multi-feature
  terms). Embedded ≡ rotated plain (matches on everything I checked; not
  fully nailed down).

- **Speculation / flagged open:**
  α saturation at fixed F/n. Uniqueness of the tight frame. Whether refined
  ansatz would fully close gap to trained R.
