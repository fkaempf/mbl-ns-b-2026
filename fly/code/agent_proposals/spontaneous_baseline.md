# Spontaneous within-fly event-null baseline

## Question
Every wall-evoked effect (turn rate 91->471 deg/s, speed 6.9->13.7 mm/s, post-bounce
tortuosity, heading change) is reported as an **absolute** peri-event value. But a fly
that simply walks fast and turns a lot anyway would produce large peri-bounce numbers
with no escape at all. What does the **same metric, aligned to a random moment of the
same fly's own laser-free walking**, look like? Each effect should be reported as a
difference from that matched spontaneous null, not as an absolute.

## Method
Use the inter-wall intervals (wall-off -> next wall-on) as the spontaneous reservoir:
they are genuinely **laser-absent** (the laser only fires while a wall is up; `owner`
map in `wall_response.py`) and abundant (~60 s median ITI, 79 ITIs in runs 72/55/46
alone). Per fly:
1. **Carve clean ITI windows.** Drop the first `EDGE=10 s` after wall-off (carries the
   escape from the wall just removed) and the last `EDGE=5 s` before the next wall-on
   (possible approach to it). Keep only ITIs with >= `2*WIN` of clean span left.
2. **Place matched pseudo-events** in the clean span: draw `K` random times per ITI
   (Poisson-thinned to >= `2*WIN` apart so windows do not overlap), giving one pseudo
   "bounce time" per draw. Match the count to that fly's real bounce count so the null
   is event-rate-matched.
3. **Run the identical peri-event extraction** used for real bounces (`shock_response`'s
   speed + unit-vector turn rate; `wall_response`'s pre/post speed, |heading change|,
   tortuosity) on the pseudo-events. This yields each fly's *spontaneous* peri-event
   distribution for every metric.
4. **Report the contrast per fly:** real peri-bounce metric minus that fly's spontaneous
   pseudo-event mean (paired across 15 flies, Wilcoxon signed-rank on the per-fly
   deltas). Bootstrap the pseudo-event draw (>= 200 reps) so the null has a CI, not a
   single realisation.

## Why rigorous
The null is built from the **same fly, same session, same laser-free walking**, so it
controls for each fly's idiosyncratic speed/turn/menotaxis (finding #3) rather than an
arbitrary zero. Pairing per fly and testing on 15 per-fly deltas avoids the
pseudoreplication the docs flag 3x. It is the matched event-null that
`shock_response`'s within-trace "far baseline" only approximates (that baseline is still
wall-present approach, not spontaneous).

## Output / plot
`plots/exploratory/spontaneous_baseline.png`: per metric (speed, turn rate,
|heading change|, tortuosity), the real peri-bounce mean trace overlaid on the
spontaneous pseudo-event mean +/- bootstrap band; inset = per-fly paired delta with the
Wilcoxon p. One row per metric.

## Caveats
- If the cleaned ITIs are too short/contaminated (large `EDGE` eats most of the 60 s
  gap), the null collapses; report n of usable ITIs and the median clean span, and say
  so honestly rather than forcing it.
- Heading change is direction-symmetric here, so use **|change|**; a non-zero
  spontaneous |change| is expected and is exactly the point of subtracting it.
- Not a test of *whether* bounces evoke a response (#1 already shows that); it
  recalibrates the **magnitude** against the fly's own spontaneous behaviour.
