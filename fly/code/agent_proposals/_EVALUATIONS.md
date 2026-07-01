# Agent proposal evaluations (auto research-loop)

Loop keeps ~3 investigation agents running and culls non-rigorous proposals.
Rule: a proposal must be concrete, grounded in the actual data, and not over-derived
fishing. If it isn't, it's deleted and the agent is told to be rigorous.

## Kept
- **rigor_stats.md** — per-fly LMM `dev ~ phase + (1|eid)` replacing the pooled,
  pseudoreplicated Wilcoxon behind "escape disrupts heading". Correct fix to a problem
  the docs already admit 3x; now powered (15 flies / 418 bounces, not the n=3 the docs
  still assume — flag: update EXPLORATORY_FINDINGS/ANALYSIS_DECISIONS). Solid.
- **dynamics_temporal.md** — Gaussian HMM on [speed, |turn|] to recast
  bounce->escape->recovery as a latent state sequence; recovery = escape-state dwell /
  hazard rate. Well-guarded (k by held-out/BIC, k=1 = honest null, change-point
  fallback, spontaneous-vs-laser control). Defensible; a bit more speculative.

- **trajectory_geometry.md** — signed perpendicular penetration profile through the
  soft hurt zone (max depth past the wall plane, dwell time, laser re-fire count).
  Exploits both data quirks (soft-trigger -> depth is real graded geometry; heading
  decoupled -> path coordinate only); premise verified vs raw data (57% cross, 2.75 s
  median dwell). Captures the axis the incidence/reflect-around-through plots collapse.
  Strongest of gen-1.

- **avoidance_learning.md** — per-fly mixed model of time-to-first-bounce + per-trial
  cross-probability vs wall-onset time, with pre-bounce speed as a covariate and an
  off-wall ITI spontaneous-crossing baseline to separate learning from fatigue/slowing.
  Honestly predicts a likely null. Well-controlled; lower priority (likely-null test).
- **individual_variability.md** — split-half test-retest reliability of per-fly metrics,
  gated on ICC before any clustering, vs a within-fly relabel null, honest
  "indistinguishable from noise at n=15" fallback. Correct reliability framing.

- **sequential_structure.md** — within-fly lag-1 mixed-effects Markov test (does one
  presentation's bounce-count / time-to-first-bounce predict the next), guarded by TWO
  nulls (trial-shuffle + circular-shift) to separate adjacency memory from drift. Real
  gap beyond the heading autocorrelation; premise verified. Solid.
- **reproducibility_batch.md** — per-date forest plot + date x phase interaction-LRT on
  the headline effects. KEY catch: the good set spans 4 dates (4/1/5/5 flies) and LASER
  POWER is confounded with date (255 vs 125) -> a "batch effect" could be a laser effect.
  Honest that 1 single-fly date is underpowered. Important caveat.

- **vrh_validation.md** — RESOLVED, not just proposed: proper_rotation_z = the genuine
  integrated FicTrac ball-yaw heading (circ corr -0.998 with FicTrac heading on run 72),
  not a VR artifact. The heading<->walking decoupling (R~0.04) is intrinsic to FicTrac
  (its own integrated_heading_lab vs animal_movement_direction_lab corr -0.005), so vrh
  is a real signal and the facing-vs-walking dissociation is a true property of the data.
  Only unverifiable: absolute body-angle accuracy (no camera ground truth). High value.

- **through_predictors.md** — per-fly mixed logistic of through/bounce (0/1) on approach
  speed, incidence, depth-at-laser-on (NOT max penetration -> avoids outcome leakage),
  laser power, within-wall bounce order; intercept-only null + WAIC/LRT honesty gate.
  Careful (leakage-aware) and distinct from the 8 existing. Solid.
- **spontaneous_baseline.md** — within-fly matched event-null: rate-matched random
  pseudo-events in the laser-free ITIs (edge-trimmed for leftover escape / next approach),
  identical peri-event extraction, every wall-evoked effect reported as a per-fly paired
  delta from spontaneous behaviour. The matched null the pipeline lacks. Broadly useful.
- **avoidance_selection_redteam.md** — red-teams "the fly walks away from the wall":
  the wall-off is distance-triggered (~200 mm cap, median 166 mm, CV 0.22 verified), so
  "ends up far" is partly a selection artifact. Test: recompute wall-behind in a fixed
  uncensored post-onset window + per-fly time-to-off-distance hazard vs laser-off ITI
  null; honest "needs a fixed-timer paradigm" fallback. Important confound check.

- **vr_fidelity.md** — RESOLVED: measured the closed loop from the logs. Software
  latency median 12.4 ms (jitter 3 ms, 0% negative), ball->VR gain exactly 1
  (proper_position step = consumed last_delta), fictrac/vrpos timestamps aligned <5 us.
  VR is fast + faithful -> spatial claims are sound. Only optical/display latency
  unmeasurable (no photodiode). High value, grounded.

- **power_sensitivity.md** — per-claim power audit: one scalar/fly, fly-clustered
  bootstrap + sign-flip CIs, MDE/required-n, TOST equivalence bound on the no-recovery
  null, pseudoreplication inflation for the pooled 62%/48% fractions (which carry no
  inferential test now). Meta-level, honest about the single-fly date. Solid.
- **menotaxis_avoidance_link.md** — within-fly epoch link: regress per-bounce avoidance
  / heading-disruption on local goal R_local (27 s pre-bounce), z-scored, (1|eid),
  block-shuffle null; cross-fly trait correlation honestly relegated as underpowered.
  Clever (avoids the weak n=15 cross-fly test). Solid.

## Culled
- (none yet)  -- 14/14 kept; the rigor-forcing prompts (mandatory honest nulls) mean the
  agents self-censor, so there has been nothing over-derived to cut. Space saturated.

## Verification round (auditing the 3 implemented analyses)
- **meta_completeness.md** (KEEP) -- flagged that the laser-on timestamp (t=0 for every
  peri-event finding) was never validated against the position clock. CHECKED: the laser
  timestamp matches a position sample to ~1 ms (NO clock skew -> peri-event timing sound),
  but the fly sits a median 7 mm (p90 31 mm) from the wall plane at laser-on, only 28%
  inside the +/-3.5 mm config zone -> the trigger geometry is wider/looser than assumed
  (affects depth-at-laser-on, not timing).
- **review_stats_impl.md** (KEEP) -- found real bugs, now FIXED:
  * avoidance_learning: `appsp` averaged over an outcome-dependent window (endogenous) ->
    fixed to a fixed 3 s pre-window; "ttf decreases" is survivorship (bounced trials only)
    -> reframed, headline is now the flat bounce-rate (no learning); off-wall 2% baseline
    is a distance-trigger confound (noted, not a clean control).
  * escape_lmm: random-slope `converged` flag can't catch a singular RE covariance ->
    added a cov_re singularity check; headline reframed to the robust per-fly Wilcoxon
    (Gaussian LMM p anticonservative on bounded dev). The claim still holds.
- **review_hmm_impl.md** -- agent died when the process exited; findings lost. Re-run the
  HMM audit (BIC param count, k robustness, fit-on-bounce leakage, 0.3 s dwell resolution).

## Generations
- gen-1: stats, dynamics, geometry  -> 3 kept.
- gen-2: variability, learning, sequential  -> 3 kept.
- gen-3: reproducibility, vrh_validation kept; path-integration held back.
- TOTAL: 8 proposals, 8 kept, 0 culled. Analysis space well-covered; vrh question
  resolved. Loop consolidating - not spawning redundant work.
