# Per-layer error scaling in the L⁴ CiS toy model (2026-04-17)

## Goal

For the L⁴-trained toy model `y = W_out @ ReLU(W_in @ x)` with sparse input
`x_j ~ Bernoulli(p) · Uniform(-1,1)`, measure directly how the per-layer
per-output error scales with (F, n, p) at fixed expansion r = F/n. The
claim being tested is that this error stays bounded (so a downstream
corrector could in principle recover features).

Two scaling regimes, from the brief:

- **Fixed p (sparsity fraction).** Cross-talk energy per output was
  predicted to saturate to an r-dependent constant as n → ∞ (does not
  vanish).
- **Fixed k = pF (co-activation count).** Predicted to decay as 1/n.

Both are tested directly by sampling multi-feature inputs and computing
residual energy, rather than inferring from R.

## Data

All trained at L⁴, 10k Adam steps batch 2048 cosine LR, on an RTX 4090.
Single seed per config; no seed sweep (caveat).

| series | (F, n) pairs | p |
|---|---|---|
| r=10 fixed-p | (10,1) ... (1000,100) + (2000,200), (5000,500) | 0.02 |
| r=30 fixed-p | (30,1), (60,2), (150,5), (300,10), (600,20) | 0.02 |
| r=100 fixed-p | (100,1) ... (2000,20), (5000,50) | 0.02 |
| fixed-k series (k ∈ {1, 2, 5}) | all the above (F, n) | p = k/F |

For each model, two measurements:

1. **R structure:** diag mean → α := (F/n) · mean(diag R); mean squared
   off-diag → ρ := off²_mean / (α²·Welch²), where Welch² = n(F-n)/[F²(F-1)].
2. **Direct test MSE:** for sampling schemes Bernoulli(p') and exactly-k,
   measure crosstalk_per_output := E_{x,i}[(ŷ_i)² · 1{i inactive}], and
   per-active-feature MSE. 200 batches × 2048 samples per point.

Scripts: `scaling/{measure_mse.py, train_ratios.py, train_fixed_k.py,
train_r10_large.py, measure_fixk.py, check_R.py, alpha_theory.py,
final_plots.py}`. Data: `data/scaling_mse*.json`, `data/R_structure.json`.
Figures: `figures/scaling_{alpha_sat, xtalk_fixed_p, xtalk_fixed_k,
collapse_final, alpha_vs_pred}.png`.

## Main empirical law

Measured across all ~70 trained models (`figures/scaling_collapse_final.png`):

> **crosstalk_per_output ≈ C(r) · k · α² / n**

when samples have exactly k active features (or k = pF for Bernoulli).
Normalising `(crosstalk_per_output · n) / (k · α²)` collapses to a
ratio-only constant C(r) with scatter < 30% across (n, k) within one r,
excluding the n=1 outliers where the ansatz R ≈ αP fails by architecture
(one-neuron model, rank-1 response).

Empirical C(r):

| r | C_obs | 1/(3r²) | ρ̄ · 1/(3r²) | ratio obs/(ρ̄/(3r²)) |
|---|---|---|---|---|
| 10 | 1.5e-3 | 3.3e-3 | 3.3e-3 | 0.45 |
| 30 | 3.0e-4 | 3.7e-4 | ~4.5e-4 | 0.67 |
| 100 | 5.0e-5 | 3.3e-5 | ~5.7e-5 | 0.88 |

ρ̄ is the average off²_mean/(α²·Welch²) for the fixed-p models at that r.
The residual ratio (0.45 → 0.88 as r grows) is attributed to ReLU gating:
at small r the hidden layer has neurons to spare and ReLU attenuates
interference more aggressively; at large r, the network is closer to
operating-capacity and attenuation is weaker.

## Regime-specific scaling

### Fixed p, fixed r (flattening, `figures/scaling_alpha_sat.png`,
`figures/scaling_xtalk_fixed_p.png`)

α and crosstalk stop growing at large n:

| r=10, p=0.02, n | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| α | 3.04 | 3.24 | 3.62 | 3.99 | 4.34 | 4.63 | 4.74 | 4.76 | 4.70 |
| crosstalk/output ×10³ | 2.73 | 3.08 | 3.84 | 4.53 | 5.17 | 5.53 | 5.66 | 5.71 | 5.66 |

At n = 100, 200, 500 the crosstalk values differ by < 1% and α by < 2%,
consistent with either true saturation at ~4.7 or slow log-like drift.
Single-seed data is insufficient to distinguish these; I did not run a
seed sweep. "Saturation" here means "no growth observable over 5× change
in n," not a proven asymptote.

### Fixed k, fixed r (decay, `figures/scaling_xtalk_fixed_k.png`)

Evaluating each fixed-p model on exactly-k sampling (k=2 shown, others
similar). log–log plot is a clean 1/n line at each r.

Concretely, `crosstalk × n / α²` is constant across n within one r:

| r=10, k=1, n | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| (crosstalk·n)/α² × 10³ | 1.50 | 1.51 | 1.53 | 1.54 | 1.54 | 1.52 | 1.51 | 1.50 | 1.49 |

Essentially constant (<5% variation). This directly verifies the Elhage
fixed-co-activation scaling: holding k co-active features fixed, per-layer
per-output error decays as 1/n and so vanishes with width.

### α saturation and deviation from α·P

At r=10 and r=30 fixed-p, R is tightly Welch-bounded (ρ ≈ 1.00-1.28).
At r=100 fixed-p, ρ grows to ≈ 2 because pF > n throughout (the network
cannot achieve Parseval with more co-active features than neurons). That
is the regime where the α·P ansatz first breaks down.

Fixed-k training at any r gives ρ = 1.01 universally (Welch-saturated).
This means the α·P characterisation from the prior report holds exactly
when the training p is small enough that pF < n; it degrades when
pF ≫ n.

## Closed-form α prediction

Minimising the L⁴ loss under the assumptions

- linear forward pass: ŷ_i = β v_i + Σ_{j ∈ S\i} R_ij v_j,
  *This is an approximation: R is defined via R[:,j] = W_out · ReLU(W_in[:,j])
  and the true single-feature response for x_j = v_j is odd in v_j only if
  ReLU(W_in[:,j]) and ReLU(−W_in[:,j]) are related by reflection, which is
  not generally true. The linear-odd-in-v assumption is baked into the
  derivation below.*
- R = α P with P Parseval (diag = n/F),
- off-diagonals at Welch level (R_ij² ≈ α²·Welch²),
- ξ_i = Σ_j R_ij v_j approximately Gaussian with E[ξ⁴] ≈ 3σ⁴.
  *This is asymptotic in k = pF; at finite k the true ratio
  E[ξ⁴]/σ⁴ = 3 + 9/(5k) exceeds 3 by 9/(5k). For k=1 the correction is
  60%, and my formula under-weights the σ⁴ term by that factor. This is
  relevant for the fixk1 comparison.*

gives (derivation in `scaling/alpha_theory.py`; u = β = αn/F, r = F/n):

> **L⁴(u; p, r) = (p/10) A(u) + (p²r/3) u² B(u) + (p²r²/3) u⁴**
>
> A(u) = (u−1)⁴ + u⁴, B(u) = (u−1)² + u²

Minimise numerically → β\*(p, r), hence α\*(p, r) = r·β\*. The prediction
is **independent of n** (only p and r).

Accuracy vs observed (`figures/scaling_alpha_vs_pred.png`):

- **Best apparent match:** fixk1 at r=100, n=10, 20 → predicted 23.26,
  27.55 vs observed 23.42, 27.88 (≤ 1% error). Caveat: this match is
  partly coincidental, a cancellation between (a) the Gaussian ξ⁴ approx
  underweighting the σ⁴ penalty (under-predicting α), and (b) ReLU
  gating in reality reducing σ² below the Welch level (over-predicting α).
- **Good match:** fixed-p at r=30 → 0.99-1.05 ratio, no single
  confounding regime.
- **Poorer match:** fixed-p at r=10 (1.0-1.5 ratio, observed α higher
  than predicted, consistent with ReLU attenuation reducing effective
  interference more at small r where the hidden layer has capacity to
  spare).
- **Breaks:** r=100 fixed-p at larger n (0.62-0.79 ratio), where ρ > 1
  violates the Welch-off-diag assumption; and all n=1 cases (W_in has
  rank 1, can't implement α·P ansatz at all).

## Summary: per-layer error at fixed r

- **Fixed p:** crosstalk_per_output → pα² · C(r) · r at large n, measured
  flat to ≤ 1% across n=100, 200, 500 at r=10. α approaches a finite
  limit α\*(p, r). C(r) ≈ ρ(r)·η(r)/(3r²) with ρ(r) ≳ 1, η(r) ∈ [0.5, 1]
  the residual factor (interpreted as ReLU gating, not independently
  verified).
- **Fixed k = pF:** crosstalk_per_output ≈ C(r) · k · α² / n → 0 as
  n → ∞. At r=10 k=1, (ct·n)/α² is constant to ≤ 5% across n=1..500
  (table above). At r=100, α is not saturated and the constancy is
  looser (~30% scatter across n on the collapse plot).

For the error-correction story: if the downstream computation sees a fixed
fraction p of co-active features, per-output error does not shrink with
width. If the downstream sees a fixed *count* of co-active features,
per-output error shrinks as 1/n. Only the second is a scale-friendly
regime.

## What I did not establish / open

1. **Seed robustness.** Single seed per config. No error bars. The clean
   1/n collapse across 70 models suggests seed-robustness but I have not
   run a sweep.
2. **η(r) is inferred, not measured.** I decompose C(r) = ρ(r)·η(r)/(3r²)
   with ρ directly measured from R, and η backed out by arithmetic. I do
   not independently verify that η is "ReLU gating" rather than some
   other unmodelled factor (e.g. non-Gaussian ξ moments, or correlation
   structure in R_ij beyond off²_mean). The η interpretation is
   speculative.
3. **Gaussian ξ⁴ approximation.** The L⁴ derivation uses E[ξ⁴] ≈ 3σ⁴.
   The true value is (3 + 9/(5k)) σ⁴; at k=1 the correction is 60% and
   materially changes the σ⁴ penalty in L4(u). Part of the reason my
   fixk1 prediction matches observed to 1% is a cancellation between this
   under-weighting and the ReLU-gating under-count in σ² (both are
   non-trivial errors that happen to point opposite). A clean rederivation
   with κ = 3 + 9/(5k) is straightforward and would shift the story.
4. **n-dependence of α at r=100 fixed p.** Empirically α drops from 11.2
   to 5.7 over n=1..50 at r=100, p=0.02, while the formula predicts a
   constant 9.2. This is the "overcrowded" pF > n regime where the α·P
   ansatz breaks (ρ → 2). A theory covering this needs an ansatz beyond
   "R = αP + Welch".
5. **Why ρ > 1 for large-p + large-r.** The optimiser cannot
   simultaneously achieve Welch packing for pF > n active features; it
   trades excess off-diagonal magnitude for something else. I did not
   look at the singular spectrum of the trained R to pin down what.
6. **Stacked-layer error.** Everything here is a single layer. The
   question of whether "per-layer error C(r)·pα²·r" is *small enough* for
   a downstream corrector is not answered by this work; it requires a
   model of the downstream corrector.

## Confidence

- **High:** 1/n decay of per-output cross-talk at fixed k, fixed r (clean
  log-log line, <5% scatter at r=10); flat plateau at r=10 fixed p for
  n=100..500; Welch saturation ρ≈1 for fixed-k=1 training at all r.
- **Medium:** the α\*(p, r) closed-form prediction (≤5% error in roughly
  half the tested configs; accidentally 1% in fixk1 r=100 due to error
  cancellation); the ρ·η/(3r²) decomposition (ρ measured directly, η
  inferred, interpretation as ReLU-gating is speculative).
- **Low / speculation:** that the "saturation regime" cross-talk
  ~5.7e-3 per output at r=10 p=0.02 is small enough for downstream
  correction to recover features. Per-active-feature MSE ~0.13 is ~40%
  of signal energy (v² = 1/3 conditional on active), so the noise is
  nontrivial.
