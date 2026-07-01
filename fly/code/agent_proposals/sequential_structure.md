# Proposal: does one wall presentation's OUTCOME predict the next? (lag-1 Markov test, non-heading)

## 1. The question

`intertrial_persistence.py` already tests whether the *walking-direction vector*
carries across inter-wall gaps (heading autocorrelation). The untested history
question is whether the **engagement outcome** of presentation n predicts n+1,
beyond heading: does a fly that hit wall n hard (many bounces, fast
time-to-first-bounce) stay engaged on n+1, or reset? Is engagement first-order
Markov, or are presentations independent given the slow heading drift already
characterised? No script asks this.

## 2. The method

One row per wall presentation (`bo.wall_trials`, paired CREATE->DESTROY): 242
presentations / 15 flies, 227 consecutive within-fly (n, n+1) pairs (verified;
median 18 walls/fly). Two non-heading outcomes: `nb` = bounces in [t_on,t_off] and
`ttf` = time-to-first-bounce, both already in `wall_trials.py:trials()` (verified
spread: nb 0..>=3, 44 zero / 51 with 3+; ttf median 5 s, IQR 3-15 s). Optional
third: escape handedness sign from `shock_adaptation.py`'s signed net-turn.

Test: a **lag-1 mixed-effects regression** `y_{n+1} ~ y_n + (1|fly)` over consecutive
pairs (logistic for hit/no-hit, Poisson for `nb`, Gaussian on log-`ttf`). The `y_n`
coefficient is the carry-over, judged against the shuffle null below.

## 3. Why rigorous + the shuffle null (REQUIRED)

A raw lag-1 coefficient is confounded by **session drift** (slow fatigue/arousal
autocorrelates any metric with zero true memory) and **between-fly** spread. The
null kills both: **within each fly, permute the order of its presentation
outcomes**, recompute the coefficient, x5000. This destroys sequential structure
while exactly preserving each fly's outcome distribution and identity, so the p is
immune to between-fly variance and to the metric's marginal drift. Because plain
permutation also breaks drift *order*, pair it with a stricter **circular-shift**
null that keeps the trend: surviving circular shift means genuine adjacency memory,
not drift. Not double-counting: heading is excluded by construction; nb/ttf/
handedness are engagement, not direction.

## 4. Output / plot

`plots/exploratory/sequential_structure.png`: (a) y_n vs y_{n+1} scatter, per-fly
colour + fit; (b) observed coefficient over the within-fly shuffle null, p; (c) same
vs circular-shift null. Print coefficient, CI, both p, per-fly slopes.

## 5. Caveats

- 2 flies have <5 pairs; partial-pooling handles them, but show per-fly slopes.
- `nb`/`ttf` partly reflect wall placement vs the fly's held heading, so a positive
  carry-over could be heading persistence in disguise: add the n->n+1 heading change
  (from `intertrial_persistence`) as a covariate; the residual is the non-heading claim.
- If only the drift signal survives (fails circular shift), or nothing does, the
  honest answer is **no sequential structure beyond the heading persistence already
  shown** — and say so.
