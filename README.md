# Forecast how many GTA VI players earn each trophy in week one

Entry for make.fun's `gta6-trophy-rarity-forecast-01`. `main.py`
(Python 3.13, standard library only) reads each case in the input
directory — the house's reads of GTA VI's Exophase PS5 trophy page up to
the case's checkpoint — and writes, for every trophy, five quantiles
(5/25/50/75/95%) of its earn rate at the case's target. Scoring is mean
pinball loss on the logit scale against the page's rates at the target;
a prize needs a lower loss than the best of three house entries (a
persistence baseline, a curve baseline, and a house model forecast).

## The model

Everything happens on the logit scale, in `ln(1 + hours since release)`
time — the clock trophy curves run on. A big release's earn rates climb
steeply in the first days and flatten as ln-time grows, so a line in
ln-time is the natural first model of a trophy's week one, and the logit
scale is the scale the contest scores on.

1. **Refresh points only.** Exophase republishes a game's counts in
   steps; between republications the page is byte-identical. Consecutive
   reads with the same players-tracked count and the same rates are one
   refresh, not new evidence, so the history is collapsed to refresh
   points before anything is fit. (The kit's own notes show the launch
   page frozen at 638 tracked players from 23:48 to 04:00 UTC, then one
   refresh to 3,868 players that moved every rate at once.)
2. **Per-trophy weighted fit.** For each trophy, a weighted least
   squares line of logit(rate) on ln(1 + hours), weights
   sqrt(players_tracked): a rate printed from 300 tracked players is
   quantized to whole earners and deserves less trust than one printed
   from 300,000.
3. **Slopes are pooled by tier.** Fifty-five short, noisy series
   extrapolate badly alone. Each trophy's slope is shrunk toward the
   information-weighted mean slope of its tier (platinum, gold, silver,
   bronze), with the shrinkage weight set by the trophy's own fit
   information against the field's median — a trophy with a long,
   well-spread history keeps its own slope; a trophy with two refreshes
   borrows its tier's.
4. **The median respects the newest level.** The forecast median is the
   average of the fitted line's value at the target and the last
   observed logit carried forward at the shrunken slope. A level shift
   at the newest refresh (the launch-night refresh cut most rates by
   diluting the player pool) is real information about where the series
   now sits, and a pure line fit averages it away.
5. **An honest spread, priced from the data.** The band's width combines
   four terms: pooled refresh-to-refresh logit volatility across all
   trophies, scaled by the square root of the horizon (a random walk's
   honest growth, with a floor so a frozen history cannot claim a
   razor-thin band); the trophy's own fit residuals; a fixed
   refresh-jump term, because the next Exophase republication can move
   every rate in one step; and counting noise, 1/projected earners +
   1/projected non-earners at the target, with the player count itself
   projected in ln-time and clamped. On the practice cases — whose
   histories are the frozen launch page and whose answers are the first
   refresh — no band earns a good score, and this one does not pretend
   otherwise; the final's 2.5 days of real reads are where the fit, the
   pooling and the priced volatility earn their keep.

Trophies never observed (null in every read) get a deliberately small
constant forecast; a case with no history at all still produces a
valid file. The program is deterministic (pure float arithmetic, files
in name order, no clock, no randomness, no network) and runs in well
under a second per case.

## Assumptions, stated plainly

- The tracked-player pool keeps growing through week one, and its
  growth dilutes rates even as earners accumulate; the model sees both
  effects only through the printed rates and the player counts, which
  is why the newest level is anchored rather than smoothed away.
- Week-one trophy behaviour of comparable big releases (steep early
  story-progression climb, slower grind/collectible climb, platinum
  near zero for weeks) is used only as a sanity shape: the tier pooling
  lets the case's own data decide how steep each tier actually is,
  instead of hard-coding another game's curve into this one.
- Exophase's refresh cadence inside the final case (reads every
  10 minutes) means the last read before the checkpoint is close to the
  checkpoint; no staleness correction is applied beyond the jump term.

## Running it

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

reads every `*.in` in the input directory and writes `<id>.out` beside
the requested output directory, in the exact shape the brief requires:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
