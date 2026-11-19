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

Validation: no real week-one answer exists before the target, so the model was tuned and checked on 150 simulated
launch weeks (wide random ranges for player base, arrival curve, play hours, story length, side-content difficulty and
Exophase refresh cadence), cut at the real deadline and scored at the real target with the contest's logit pinball
loss: **0.250** for this model vs **0.737** for the house baseline and **0.454** for the house curve baseline; it beat
the better of the two in 145 of 150 weeks, with 90% coverage of 0.93. Parameters were picked on 100 weeks and checked on
the 50 held out.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=01tnmm85&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
