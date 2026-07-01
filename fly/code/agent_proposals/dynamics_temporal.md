# Proposal: fit the bounce->escape->recovery loop as a latent-state sequence (HMM), measure the recovery as a transition rate

## 1. The question it answers

EXPLORATORY_FINDINGS #4 states a *sequence* model in words: menotaxis -> approach -> fast
random escape turn -> slow ~10 s drift back -> repeat. But "recovery" has only ever been
measured as a deviation magnitude in hand-picked windows (`heading_recovery.py`,
`escape_correction.py`: -3..-1 s vs +1..+3 s). The unanswered dynamics question is:
**after a bounce, is the return to menotaxis a graded continuous relaxation, or a discrete
switch back into a "fixating" state with a measurable per-second transition probability and
dwell time? And do escapes share one latent "escape" state regardless of incidence?**
That is a question about *state occupancy and transition timing*, which no current script asks.

## 2. The method

Fit a Gaussian HMM (or, as the honest fallback, a multivariate change-point model) on the
**continuous** 2-D emission stream `[forward speed, |turn rate|]` (both already computed in
`shock_response.py`), at the native ~100 Hz, across each whole run. Start with k=2-3 states
chosen by held-out log-likelihood / BIC, not assumed. Expect states like *fixating-walk*
(moderate speed, low turn) and *escape/reorient* (speed surge + high turn). Then, **without
using the laser times in the fit**, overlay the bounce times and measure: (a) P(escape state
| time-since-bounce), (b) the escape-state dwell-time distribution, (c) the recovery as the
hazard rate of escape->fixate transitions (its 1/e time = the real recovery constant), and
(d) the escape-state transition matrix early vs late in the session.

## 3. Why it is rigorous / what it adds over event-triggered averages

ETAs assume the response is time-locked and stereotyped *in lab time*; they smear any
trial-to-trial jitter in onset and offset. The recovery especially is not an average over a
window - it is a return process with its own clock. An HMM (i) defines the escape state from
the *data*, so it can validate or falsify the assumed sequence rather than presupposing it;
(ii) turns "no heading recovery" into a *quantified* dwell time and hazard rate with units;
(iii) detects escapes the laser-locked average misses (spontaneous reorientations vs
laser-driven ones - a built-in control the current pipeline lacks). The data supports it:
~15 runs x ~60-75 min at ~100 Hz is hundreds of thousands of samples and ~30 bounces/fly,
far above what HMM fitting needs.

## 4. The concrete output / plot

`plots/exploratory/state_dynamics.png`: (a) a raster of the inferred state along one run with
bounce times marked; (b) P(escape state) vs time-since-bounce, pooled (the recovery curve,
with its fitted 1/e constant); (c) the escape-state dwell-time histogram; (d) per-fly
transition-matrix heatmaps. Per-fly stats (n~3 curated flies) to avoid the pseudoreplication
flagged throughout ANALYSIS_DECISIONS.

## 5. Honest caveats

Only ~3 hand-curated flies, so state *parameters* are per-fly but cross-fly claims stay n=3.
A 2-feature HMM with Gaussian emissions and geometric dwell times is a crude generative model
(real dwell times may not be geometric - check empirically; if not, an HSMM or simple
change-point segmentation is the more honest tool). State labels are interpretive, not
ground truth. If k=1 wins on held-out likelihood, that is the honest answer: the stream is one
graded process and the discrete-state framing adds nothing.
