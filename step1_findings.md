# Step 1: Characterizing the single-feature projection in L^4-trained CiS models

## Setup

Model: `y = W_out @ ReLU(W_in @ x)` with `W_in ∈ R^{n×F}`, `W_out ∈ R^{F×n}`,
trained under L^4 loss on sparse inputs (`x_j ~ Bernoulli(p) × Uniform(-1,1)`),
target `y = ReLU(x)`.

Single-feature response matrix:
    R[:, j] := W_out @ ReLU(W_in[:, j])

Perfect recovery would mean R = I_F, but rank(R) ≤ n < F.

## Main empirical finding

R has a **rank-n near-projection structure with approximately uniform diagonal**,
so it is close to `α · P` where P is a rank-n orthogonal projection with
`P_jj = n/F`:
- `diag(R)` is tightly clustered at `α · n/F` (std/mean ~1–4%).
- Off-diagonals with distinct codewords cluster near ± (the Welch lower bound)
  `α · sqrt(n(F-n) / (F²(F-1)))`, indicating an approximate (not exact)
  equiangular tight frame.
- For same-codeword pairs, R_ij tracks R_jj (they share a projection direction).
- R is NOT exactly a scaled projection: `||R - R^T||/||R|| ≈ 4–13%`
  (systematic asymmetry) and off-diagonal mean is mildly positive. These
  deviations are small relative to the diagonal and singular-value structure
  but are consistent across configs, so "R = α P exactly" is an idealisation.

## Evidence

### Frobenius bound saturation
For every trained config, after optimal uniform rescale `s = tr(R)/||R||_F^2`:
    ||sR - I||_F^2 ≈ F - n
to within relative error < 0.005. Example values (ratio ||sR - I||²/(F-n)):

| name              | F   | n  | ratio  |
|-------------------|-----|----|--------|
| small_20f_5n_L4   | 20  | 5  | 1.0012 |
| small_50f_5n_L4   | 50  | 5  | 1.0007 |
| small_100f_10n_L4 | 100 | 10 | 1.0006 |
| small_500f_50n_L4 | 500 | 50 | 1.0013 |
| small_1000f_100n_L4 | 1000 | 100 | 1.0009 |
| oct_31f_5n_L4     | 31  | 5  | 1.0009 |

The rank-n bound is saturated essentially everywhere.

### Rank and spectrum
Singular values of R show a sharp drop at index n. The top-n singular values are
nearly equal (std/mean < 0.05 in most configs). This is the signature of α × P
with P an orthogonal projection — all nonzero singular values equal to α.

### Uniform diagonal
`diag(R)` has std/mean ~ 0.01–0.04 for most configs. E.g. for 20f/5n:
diag mean 0.511, std 0.013. That is, all 20 features get the same
single-feature-active response magnitude. This is the defining property of a
Parseval frame.

### Off-diagonal distribution (20f/5n)
The histogram of off-diagonals with distinct codewords is bimodal, peaked
near ±0.203, which is α × Welch = α × sqrt(n(F-n)/(F²(F-1))) = 2.045 × 0.0993.
Spread ~0.08 (std). This matches the ETF off-diagonal magnitude to within 10%.
The signs are not uniformly split (≈60/40 positive/negative) but the sign
matrix does not have a two-eigenvalue Seidel-spectrum signature, so I would
not call the frame an "ETF+Seidel" beyond "approximately equiangular."

## The scale α

Define α := mean(diag R) × F/n. Empirically:

| (F, n) | α obs | α single-feat L4+ETF | regime |
|--------|-------|----------------------|--------|
| (5, 5)  | 0.975 | 1.000 | F=n: trivial, α=1 ✓ |
| (3, 2)  | 1.006 | 1.000 | F=2^n-1=3: ETF-optimal single-feat |
| (7, 3)  | 1.324 | 1.400 | F=2^n-1=7 |
| (15, 4) | 1.845 | 2.068 | F=2^n-1=15 |
| (31, 5) | 2.684 | 3.153 | F=2^n-1=31 |
| (20, 5) | 2.045 | 2.256 | 18/20 features get unique codewords |
| (10, 1) | 3.044 | 3.244 | 1 codeword, heavy sharing |
| (100, 10) | 3.987 | 5.168 | 97/100 unique codewords |

**Single-feature L^4 ansatz.** Assuming (a) R = αP with P a uniform projection
P_jj = n/F and all |off_ij| equal to Welch magnitude, (b) loss dominated by
single-feature inputs, and (c) target-magnitudes v ~ Uniform(0,1) on active
entries, the per-feature L^4 error is

    v^4 · [(α n/F − 1)^4 + (F-1) · (α · |P_ij|)^4].

Minimising over α gives

    β := α · n/F  =  1 / (1 + [γ² / (F-1)]^{1/3}),   γ := (F-n)/n.

This is a clean closed form but **overpredicts α** because it ignores
multi-feature interference. With p > 0 and two features active at once, the ReLU
couples them, and large α amplifies cross-talk penalty. The match is good when:
1. F = n (γ = 0, β = 1, α = 1): both observation and prediction ≈ 1.
2. F = 2^n - 1 with modest p: ansatz predicts α within ~15%.
3. Fixed F = 20, varying n: overpredicts by ~10% (see figures/alpha_scaling.png).

For fixed expansion F/n = 10 (10f/1n through 1000f/100n), observed α grows from
3.04 → 4.74 and appears to saturate; single-feature ETF prediction would
diverge, confirming multi-feature effects matter at scale.

## Which projection (i.e., which subspace)?

R factors through Col(W_out) (an n-dim subspace of R^F). Network training
picks *both*:
1. The subspace ⊆ R^F (sits in Col(W_out)).
2. The frame packing (rows of V for V an orthonormal basis of Col(W_out) with
   ||V[j,:]||^2 = n/F for all j — i.e., Parseval frame structure).

The second requirement (uniform diagonal) is what distinguishes the L^4
solution from an arbitrary rank-n projection. L^2 would reward any rank-n
projection equally (all give Frobenius bound F-n), so it prefers to ignore
some features (bimodal "naive"-style solution). L^4 forces uniform per-feature
error, which forces the Parseval structure.

## Limit cases — verified
- **F = n (trivial):** R ≈ I, α ≈ 1, per-feature MSE ~10^{-4}. Clean ReLU.
- **F = 2^n - 1 (one codeword per feature):** Each feature gets a unique
  sign pattern; off-diagonals match Welch bound to within ~10%. Note that
  real ETFs only exist up to the Gerzon bound `F ≤ n(n+1)/2`, which fails
  for (F, n) = (7, 3), (15, 4), (31, 5); in these cases no *real* ETF exists,
  and the network is approximating the Welch lower bound, not attaining it.
- **F > 2^n - 1:** codewords shared; same-codeword features have matched
  R columns (redundant); behavior is "codeword-grouped" + within-group
  interference.

## Open / unsure
- Exact α: the single-feature L^4 ansatz is a clean upper bound, but the real
  α is lower due to multi-feature inputs. Accounting for p requires integrating
  over the active-set distribution with ReLU nonlinearity, which I have not
  derived in closed form. Empirically, a p-sweep on the (31, 5) model shows
  α dropping monotonically with p (α = 2.76 at p = 0.005 down to 1.44 at
  p = 0.2), confirming the multi-feature hypothesis.
- α appears to saturate at fixed F/n = 10 as (F, n) grows (3.04 at 10/1 up
  to 4.74 at 1000/100) but this is based on limited large-scale data; I have
  not verified true saturation.
- R has systematic asymmetry (`||R - R^T||/||R|| ≈ 4–13%`) not captured by the
  scaled-projection ansatz. A refined ansatz of the form
  `R = α P + β · (mean-outer-product term) + skew` may fit better but is
  beyond this step's scope.
- Whether the subspace chosen is determined by the task beyond the uniform-
  diagonal constraint. Multiple tight frames exist for any (F, n); I have not
  investigated which one the network finds or whether it is unique across
  seeds.

## Files produced
- `probe_onehot.py`, `analyze_R_structure.py`, `analyze_all.py`, `visualize_R.py`
- `train_limit_configs.py`
- `data/onehot_probe.json`, `data/onehot_probe_R.npz`, `data/R_structure_summary.json`
- `figures/alpha_scaling.png`, `figures/R_heatmaps.png`, `figures/R_20f_5n_detail.png`
- weights: `weights/oct_*.pt`, `weights/trivial_*.pt`, `weights/fix5n_*.pt`, `weights/small_20f_{6,7,8}n_L4.pt`
