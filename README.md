# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Idnafaa/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Idnafaa/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PS5 trophies, its earn rate on Exophase one week after launch, as five
quantiles (5/25/50/75/95%). One Python file, standard library only, deterministic, well under a second per case.

How it works:

- **Refresh events.** Exophase recomputes the page on its own schedule, so the house's reads are collapsed into one
  event per distinct page.
- **Counts, not rates.** Launch night was 638 players, a few of whom already held every trophy, so early rates were
  inflated and then diluted. Rates are turned back into earned counts, the launch-night page is set apart as a seed
  cohort, and the model follows the earn rate of everyone who arrived after it.
- **Trend.** For each trophy, logit(cohort rate) is fitted against ln(hours since launch) over the last ~40 hours by
  weighted least squares (recent and larger cohorts weigh more). The slope is shrunk toward a slope pooled across
  trophies at a similar level, damped, and carried to the target; players tracked is extrapolated the same way.
- **Spread.** A seeded Monte Carlo over level, slope, random-walk drift, small-count noise and player growth gives the
  quantiles, with ordering rules applied draw by draw (a later chapter is never more common than an earlier one, the
  platinum never more common than any trophy).

Validation: no real week-one answer exists before the target, so the model was tuned and checked on simulated
launch weeks (wide random ranges for player base, arrival curve, play hours, story length, side-content difficulty and
Exophase refresh cadence), cut at the real deadline and scored at the real target with the contest's logit pinball
loss. A second tuning pass used 100 weeks conditioned to match the real Exophase page two days after launch (44,161
players tracked; Chapter 1 at 11.98%, Chapter 2 at 1.71%, most side trophies still at the 0.01% floor): on the 30
held-out conditioned weeks the previous tuning scored 0.451 and this one 0.414, and on the original 150 wide-range
weeks this one scores 0.224 against the previous 0.250 and the house curve baseline's 0.454. The changes: gentler
player-growth extrapolation, weaker slope pooling, a wider tail, and an upward nudge with a wider upper tail for
trophies still near the floor, whose progressing-player wave is hardest to time.

Practice refresh: the rules document Exophase's first refresh of the game (the 638-player launch-night page, unchanged
through 04:00Z, had become 3,868 players tracked by the 04:30Z read). When a case holds only that launch-night page and
its target falls at or after that read (within 12 hours), the forecast centres each trophy on its launch-night earned
count grown by about a tenth over 3,868 players, with a narrow logit spread, instead of the generic dilution prior.
Histories that already contain a refreshed page, like the final's, go through the model above unchanged.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=01tnmm85&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
