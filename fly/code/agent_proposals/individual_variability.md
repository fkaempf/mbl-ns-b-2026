# Proposal: split-half reliability of per-fly metrics, before any typology claim

## 1. The question

Per-fly numbers vary wildly: inter-wall heading concentration **R 0.21-0.96**
(`intertrial_persistence.py`), menotaxis **R 0.23/0.38/0.47** (`heading_persistence.py`),
escape L/R bias **0.50/0.64/0.64 left** (`shock_adaptation.py`). Before calling this a
typology, one question must be settled: is a fly's metric a **stable trait** (the fly scores
the same on an independent stretch of its own run) or just **sampling noise** spread across
15 flies? You cannot cluster or name types until that is answered.

## 2. The method

**Split-half reliability**, not clustering. For each of the 15 runs, split the session
into half A (first ~half) and half B (second ~half) by time, and recompute each per-fly
metric on each half independently: inter-wall heading R, menotaxis R and mean direction
(circular), and escape L/R fraction. Then across the 15 flies:

- **ICC(3,1)** (or circular within-fly correlation for the heading-direction metric)
  between half-A and half-B scores = test-retest reliability. High ICC = a real, stable
  individual difference; ICC near 0 = the spread is within-fly noise.
- **Null benchmark by within-fly relabeling**: shuffle which observations (inter-wall
  intervals / bounces) fall in "half A" vs "half B" *within each fly* and recompute ICC,
  many times. This builds the noise distribution for ICC given each fly's own sample size,
  so the observed ICC is read against the right null, not against 0.
- Only metrics that **pass** (ICC clearly above the shuffle null, with a CI) earn a
  variance-components statement: fit `metric ~ 1 + (1|fly)` and report between-fly vs
  residual variance. No k-means, no named types.

## 3. Why it is rigorous

It tests the actual claim (stability), not a downstream story. Split-half uses each fly as
its own control, so it cannot manufacture structure from 15 noisy points the way clustering
can. The within-fly shuffle null calibrates ICC to each fly's real sample size (8-49
bounces; uneven interval counts), addressing the same pseudoreplication the codebase flags
repeatedly. Circular metrics get circular statistics (cos/sin), consistent with
`heading_persistence.circ`.

## 4. Concrete output

`plots/exploratory/split_half_reliability.png`: one panel per metric, half-A vs half-B
scatter (15 flies, one point each, identity line), titled with ICC, its shuffle-null 95%
band, and p. A summary bar of ICC per metric with the null band shaded.

## 5. Honest caveats

n=15 is small for ICC: the CI will be wide, and a single metric may land ambiguously. Halving
a 60-75 min run halves the per-fly sample, inflating within-half noise and biasing ICC
**down** (conservative). Time-split confounds stability with drift (`heading_persistence`
shows drift is small, +4 to +32 deg, so acceptable; an odd/even-interval split is the
drift-free cross-check). **If most metrics fail, the honest conclusion is that at n=15 the
per-fly spread cannot be distinguished from sampling noise, and no typology should be
claimed.** Say that plainly rather than clustering anyway.
