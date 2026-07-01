# Proposal: power / sensitivity audit of the five headline claims at n=15 flies

## 1. The question

Every per-claim stats proposal fixes *how* to test an effect. None asks the prior
question: at **15 flies (4/1/5/5 across 4 dates, one a single-fly date), ~28 bounces/fly,
418 bounces**, which headline claims does this n actually support, and which are riding on
pooled-bounce pseudoreplication or on the lone 20260626 fly? The five headlines:
(a) escape disrupts heading (dev 53->103 deg, per-bounce paired), (b) ~62% wall-behind /
reverse-walking (per-bounce binary), (c) no heading recovery (a *null*), (d) ~48%
walk-through (per-bounce binary), (e) partial inter-wall heading persistence (per-fly R,
0.21-0.96). These are not interchangeable: (b), (c) and (d) are **pooled descriptive numbers with no
inferential test at all** (the 62% and 48% are bare percentages over 418 correlated
bounces; "no recovery" is a pooled median); (a) has a per-fly loop but a pseudoreplicated
pooled Wilcoxon headline; (e) is the one already aggregated per fly. So this audit is the
first fly-level CI three of the five claims have ever had.

## 2. The method

Collapse each claim to **one scalar per fly** (the only honest unit), then quantify what
n=15 can resolve:

- **Fly-level effect size + bootstrap CI.** For each fly compute its own statistic
  (median dev_post-dev_pre; reverse fraction; through fraction; inter-wall R; pre/post
  recovery slope). The effect is the across-fly mean/median; CI by **bootstrapping the 15
  flies** (cluster bootstrap, fly = unit), reported as Cohen's dz for paired continuous and
  as a fly-level proportion for the binary claims.
- **Sign-flip / sign-test CI** as a distribution-free cross-check on the paired claims
  (a, c): how many of 15 flies move the claimed way, exact binomial CI. 15/15 vs 9/15 is
  the difference between a robust and a coin-flip effect.
- **Minimum detectable effect (MDE) and required n.** For each claim, given the observed
  between-fly SD, report the smallest effect a one-sample/paired test on n=15 detects at
  80% power, alpha 0.05, and invert it to the **n needed** for the *observed* effect. The
  null claim (c) gets a **TOST equivalence** bound: the smallest non-zero recovery these 15
  flies could have hidden.
- **Pseudoreplication inflation factor.** For (b) and (d), pooled-bounce CI width vs
  fly-clustered CI width (design-effect ~ 1+(m-1)*ICC): how much the 418-vs-15 gap has
  overstated precision.
- **Leave-one-date-out.** Refit each effect dropping each date; flag any claim that
  collapses without one date, confirming the single-fly 20260626 date is not load-bearing.

## 3. Why it is rigorous

It is not a generic power lecture: every input (effect size, between-fly SD, fly count per
date) is measured from this dataset, and the output is per-claim, not global. It uses the
fly as the unit throughout, so it cannot be gamed by bounce count. It is purely diagnostic
- it reorders existing claims by evidential strength and states equivalence bounds for the
null; it cannot manufacture a new finding.

## 4. Concrete output

`plots/exploratory/power_sensitivity.png`: a five-row forest plot of fly-level effect +
bootstrap CI per claim, with sign-test fraction annotated; a companion table of MDE,
required-n, sign count, pseudoreplication inflation factor, and leave-one-date-out delta.
A one-line verdict per claim: **adequately powered / underpowered / single-date-dependent**.

## 5. Caveats

n=15 makes the bootstrap CIs themselves wide and MDE estimates noisy - the audit reports
ranges, not point certainties, and a "powered" verdict means "powered for an effect this
size," not "true." The single-fly date cannot be split from its date effect (per
`reproducibility_batch`), so leave-one-date-out for 20260626 is informative only as a
sanity drop, not a contrast. Equivalence bounds on the null (c) will likely be loose:
expect the honest reading to be "no recovery *detectable*," not "no recovery."
