"""GTA VI week-one trophy earn-rate forecasts for make.fun (gta6-trophy-rarity-forecast-01).

Reads each case file in $MAKEFUN_INPUT_DIR (default /input), writes the matching
forecast to $MAKEFUN_OUTPUT_DIR (default /output). Python 3.13, standard library
only, fully offline, seeded so two runs agree byte for byte.

Method outline
--------------
Exophase republishes a game's trophy counts only when it refreshes them, so the
house's many reads collapse to a short list of distinct pages ("events"). The
launch-night page (638 tracked players, a few of whom already held everything)
is treated as a fixed seed cohort: subtracting its earned counts leaves the
earn rate of everyone who arrived later, which is the quantity worth trending.

For every trophy the post-launch cohort rate is put on the logit scale and
regressed on log time since launch (weighted toward recent, larger pages). The
slope is shrunk toward a slope pooled across trophies of similar level and
damped before being carried to the target; tracked-player growth gets the same
treatment on the log scale. Uncertainty (level, slope, drift, small-count noise,
player growth) is propagated with a seeded Monte Carlo whose draws share a
common component across trophies, the trophy list's ordering rules are imposed
draw by draw, and the 5/25/50/75/95% quantiles are read off. Two calibrations
from real launch-week data widen the upper tail and nudge the median up for
trophies still near the floor, where the wave of progressing players is hardest
to time, and a trophy already at ~100% stays pinned there.

A history that never left the launch-night page is answered from the first
refresh the rules document (3,868 tracked players) when the target falls just
after it, and from a wide dilution prior otherwise.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_QUANTILE = (-1.6448536269514722, -0.6744897501960817, 0.0,
                 0.6744897501960817, 1.6448536269514722)
LAUNCH_UTC = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)
RATE_FLOOR, RATE_CEIL = 0.01, 99.99

# Constants fitted on simulated launch weeks, then re-fitted on weeks conditioned
# to match the real Exophase page two days after launch (see README).
TUNING = {
    "log_offset_h": 0.0,     # hours added inside ln(hours + offset)
    "fit_window_h": 40.0,    # trend fit ignores events older than this before the last
    "recency_halflife_h": 18.0,  # e-folding time of the recency weights
    "slope_damp": 0.75,      # multiplier on each fitted logit slope
    "players_damp": 0.4,     # multiplier on the tracked-players log slope
    "pool_weight": 0.5,      # pseudo-weight of the pooled slope when shrinking
    "drift_sd": 0.16,        # random-walk sd per unit of log time (logit scale)
    "slope_sd_scale": 1.0,   # inflation of the slope standard error
    "level_sd_floor": 0.10,  # minimum level sd on the logit scale
    "tail_stretch": 1.3,     # extra width applied at the 5% and 95% quantiles
    "common_share": 0.7,     # share of draw noise common to all trophies
    "draws": 3000,
    "low_uptail": 0.5,       # extra upper-tail widening for near-floor trophies
    "low_shift": 0.2,        # logit nudge up for near-floor trophies
    "pin_full": 1.0,         # 1: a trophy at >=99.5% in the last read stays ~100%
}


def parse_moment(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_after_launch(moment):
    return (moment - LAUNCH_UTC).total_seconds() / 3600.0


def to_logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def from_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp_percent(x):
    return min(max(x, 0.0), 100.0)


def distinct_pages(history, n_trophies):
    """Collapse reads to one entry per distinct page: (hours, players, rates)."""
    pages = []
    previous_key = None
    for read in history:
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if not players or len(rates) != n_trophies:
            continue
        key = (players, tuple(rates))
        if key == previous_key:
            continue
        previous_key = key
        pages.append((hours_after_launch(parse_moment(read["at"])),
                      float(players), list(rates)))
    return pages


def weighted_line(xs, ys, ws):
    """Weighted least squares. Returns (intercept, slope, resid var, slope var)."""
    total_w = sum(ws)
    if total_w <= 0 or len(xs) < 2:
        return None
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = sum(w * (y - (intercept + slope * x)) ** 2
                for w, x, y in zip(ws, xs, ys)) / total_w
    n_eff = total_w * total_w / sum(w * w for w in ws)
    dof = max(n_eff - 2.0, 1.0)
    resid_var = resid * n_eff / dof
    slope_var = resid_var / (sxx / total_w) / max(n_eff, 1.0)
    return intercept, slope, resid_var, slope_var


def empirical_quantile(sorted_values, tau):
    n = len(sorted_values)
    position = tau * (n - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def story_chain_ids(trophies):
    """Chapter trophies in order, then the final mission: rates fall along it."""
    numbered = []
    finale = None
    for trophy in trophies:
        description = (trophy.get("description") or "").strip().lower()
        if description.startswith("complete chapter"):
            digits = "".join(ch for ch in description if ch.isdigit())
            if digits:
                numbered.append((int(digits), trophy["id"]))
        elif description == "complete the final mission.":
            finale = trophy["id"]
    chain = [tid for _, tid in sorted(numbered)]
    if finale is not None:
        chain.append(finale)
    return chain


# --- The documented first refresh -------------------------------------------------
# The rules record that the 638-player launch-night page stood unchanged through
# the 04:00Z read of 19 Nov and had refreshed by 04:30Z to 3,868 tracked players.
# A case whose history is only that page and whose target sits just after the
# refresh is answered by centring each trophy on its launch-night earned count,
# grown slightly, spread over the refreshed player base.
SEED_PAGE_PLAYERS = 638
REFRESH_MOMENT = datetime(2026, 11, 19, 4, 30, 0, tzinfo=timezone.utc)
REFRESH_PLAYERS = 3868.0
REFRESH_COUNT_GROWTH = 1.10
REFRESH_COUNT_EXTRA = 0.35
REFRESH_LOGIT_SD = 0.20


def refresh_player_count(pages, target_text, horizon_hours):
    if not pages or any(int(round(p[1])) != SEED_PAGE_PLAYERS for p in pages):
        return None
    if parse_moment(target_text) < REFRESH_MOMENT or horizon_hours > 12.0:
        return None
    return REFRESH_PLAYERS


def refresh_forecast_rows(trophies, ids, seed_rates, seed_players, refreshed_players):
    count = len(trophies)
    grid = []
    for i in range(count):
        rate = seed_rates[i]
        rate = float(rate) if isinstance(rate, (int, float)) else 1.0
        if rate >= 99.5:
            grid.append([99.95, 99.99, 100.0, 100.0, 100.0])
            continue
        seed_earned = round(rate * seed_players / 100.0)
        median = (seed_earned * REFRESH_COUNT_GROWTH + REFRESH_COUNT_EXTRA) / refreshed_players
        median = min(max(median, 1e-6), 1.0 - 1e-6)
        grid.append([100.0 * from_logit(to_logit(median) + z * REFRESH_LOGIT_SD)
                     for z in Z_OF_QUANTILE])
    position = {tid: i for i, tid in enumerate(ids)}
    chain = [position[t] for t in story_chain_ids(trophies) if t in position]
    platinum = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    for level in range(5):
        for earlier, later in zip(chain, chain[1:]):
            grid[later][level] = min(grid[later][level], grid[earlier][level])
        for pi in platinum:
            grid[pi][level] = min(grid[pi][level],
                                  min(grid[k][level] for k in range(count) if k != pi))
    rows = []
    for i in range(count):
        qs = [round(clamp_percent(q), 4) for q in grid[i]]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": ids[i], "q": qs})
    return rows


def forecast(case, params=None):
    cfg = dict(TUNING)
    if params:
        cfg.update(params)
    trophies = case["trophies"]
    n = len(trophies)
    ids = [t["id"] for t in trophies]
    target_hours = hours_after_launch(parse_moment(case["target"]))
    checkpoint_hours = hours_after_launch(parse_moment(case["at"]))
    pages = distinct_pages(case.get("history") or [], n)

    latest_rates = [None] * n
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        if len(rates) == n:
            latest_rates = list(rates)
            break
    latest_rates = [r if isinstance(r, (int, float)) else None for r in latest_rates]

    rng = random.Random(20261126)
    draws = int(cfg["draws"])
    samples = [[0.0] * draws for _ in range(n)]
    shared_z = [rng.gauss(0.0, 1.0) for _ in range(draws)]
    shared_g = [rng.gauss(0.0, 1.0) for _ in range(draws)]
    common = cfg["common_share"]

    def mixed_noise(k, sd):
        return sd * (math.sqrt(common) * shared_g[k]
                     + math.sqrt(1.0 - common) * rng.gauss(0.0, 1.0))

    offset = cfg["log_offset_h"]
    x_target = math.log(max(target_hours, 0.5) + offset)

    seed_hours, seed_players, seed_rates = (
        pages[0] if pages else (checkpoint_hours, 1.0, latest_rates))
    seed_earned = [((r or 0.0) / 100.0) * seed_players for r in seed_rates]

    # Pages with enough post-seed players to say something about the cohort.
    usable = [p for p in pages[1:]
              if p[1] - seed_players >= max(150.0, 0.25 * seed_players)]

    if usable:
        last_page = usable[-1]
        x_last = math.log(max(last_page[0], 0.5) + offset)
        horizon_x = max(x_target - x_last, 0.0)
        window = [p for p in usable if last_page[0] - p[0] <= cfg["fit_window_h"]]

        # Tracked players: log count against log time.
        px = [math.log(max(p[0], 0.5) + offset) for p in window]
        py = [math.log(p[1]) for p in window]
        pw = [math.exp(-(last_page[0] - p[0]) / cfg["recency_halflife_h"]) for p in window]
        players_fit = weighted_line(px, py, pw) if len(window) >= 2 else None
        players_slope = max(players_fit[1], 0.0) if players_fit else 1.0
        players_slope_sd = (math.sqrt(players_fit[3]) + 0.3 * players_slope
                            if players_fit else 1.0)
        log_players_last = math.log(last_page[1])

        # One cohort-rate fit per trophy.
        fits = []
        for i in range(n):
            xs, ys, ws, variances = [], [], [], []
            for page in window:
                rate = page[2][i]
                if rate is None:
                    continue
                cohort = page[1] - seed_players
                earned = (rate / 100.0) * page[1] - seed_earned[i]
                g = (max(earned, 0.0) + 0.5) / (cohort + 1.0)
                g = min(max(g, 1e-5), 1.0 - 1e-5)
                var_count = 1.0 / max(cohort * g * (1.0 - g), 0.25)
                var_round = ((0.00005 * page[1] / max(cohort, 1.0)) ** 2
                             / max(g * (1.0 - g), 1e-6) ** 2 / 3.0)
                xs.append(math.log(max(page[0], 0.5) + offset))
                ys.append(to_logit(g))
                variances.append(var_count + var_round)
                ws.append(math.exp(-(last_page[0] - page[0]) / cfg["recency_halflife_h"])
                          / (1.0 + var_count + var_round))
            if not ys:
                fits.append(None)
                continue
            y_last, var_last = ys[-1], variances[-1]
            fit = weighted_line(xs, ys, ws) if len(xs) >= 3 else None
            if fit:
                intercept, slope, resid_var, slope_var = fit
                fitted_level = intercept + slope * xs[-1]
                precision_obs = 1.0 / max(var_last, 1e-6)
                precision_fit = 1.0 / max(resid_var, 1e-4)
                level = ((y_last * precision_obs + fitted_level * precision_fit)
                         / (precision_obs + precision_fit))
                level_var = 1.0 / (precision_obs + precision_fit)
                fits.append({"level": level, "level_var": level_var, "slope": slope,
                             "slope_var": slope_var, "resid": resid_var})
            else:
                fits.append({"level": y_last, "level_var": var_last, "slope": None,
                             "slope_var": None, "resid": 0.04})

        # Slopes pooled as a function of level across the trophies that have one.
        pool_x, pool_y, pool_w = [], [], []
        for f in fits:
            if f and f["slope"] is not None and f["slope_var"] is not None:
                pool_x.append(f["level"])
                pool_y.append(f["slope"])
                pool_w.append(1.0 / (f["slope_var"] + 0.05))
        pooled = weighted_line(pool_x, pool_y, pool_w) if len(pool_x) >= 4 else None
        if pooled:
            pool_intercept = pooled[0] - pooled[1] * pool_x[-1]
            pool_slope = pooled[1]
        else:
            pool_intercept = sum(pool_y) / len(pool_y) if pool_y else 0.3
            pool_slope = 0.0
        pool_var = pooled[2] if pooled else 0.25

        for i in range(n):
            f = fits[i]
            if f is None:
                fallback = latest_rates[i]
                centre = to_logit((fallback if fallback is not None else 1.0) / 100.0)
                for k in range(draws):
                    samples[i][k] = 100.0 * from_logit(centre + rng.gauss(0.0, 1.0))
                continue
            pooled_slope_i = pool_intercept + pool_slope * f["level"]
            if f["slope"] is not None:
                weight_own = 1.0 / (f["slope_var"] + 1e-4)
                weight_pool = cfg["pool_weight"] / (pool_var + 0.02)
                slope = ((f["slope"] * weight_own + pooled_slope_i * weight_pool)
                         / (weight_own + weight_pool))
                slope_sd = (math.sqrt(1.0 / (weight_own + weight_pool))
                            * cfg["slope_sd_scale"] + 0.15 * abs(slope))
            else:
                slope = pooled_slope_i
                slope_sd = math.sqrt(pool_var) + 0.2 * abs(slope)
            slope *= cfg["slope_damp"]
            static_sd = math.sqrt(f["level_var"] + cfg["level_sd_floor"] ** 2)
            drift_sd = cfg["drift_sd"] * math.sqrt(horizon_x)
            total_sd = math.sqrt((slope_sd * horizon_x) ** 2 + static_sd ** 2 + drift_sd ** 2)
            for k in range(draws):
                y_draw = f["level"] + slope * horizon_x + mixed_noise(k, total_sd)
                g_draw = from_logit(y_draw)
                log_players = (log_players_last
                               + max(players_slope * cfg["players_damp"]
                                     + shared_z[k] * players_slope_sd, 0.0) * horizon_x)
                players_draw = math.exp(log_players)
                cohort_draw = max(players_draw - seed_players, 1.0)
                earned_draw = seed_earned[i] + g_draw * cohort_draw
                if earned_draw < 50:
                    earned_draw = max(
                        earned_draw + rng.gauss(0.0, math.sqrt(max(earned_draw, 0.5))), 0.0)
                samples[i][k] = clamp_percent(100.0 * earned_draw / players_draw)
    else:
        # Nothing but the launch-night page so far: expect dilution as players
        # arrive, unless the documented first refresh answers the case.
        horizon_hours = max(target_hours - checkpoint_hours, 0.0)
        refreshed = refresh_player_count(pages, case["target"], horizon_hours)
        if refreshed:
            return {"trophies": refresh_forecast_rows(
                trophies, ids, seed_rates, seed_players, refreshed)}
        for i in range(n):
            rate = latest_rates[i]
            if rate is None:
                rate = 1.0
            start_earned = seed_earned[i] if pages else (rate / 100.0) * seed_players
            growth_centre = math.log(4.0) + 0.1 * math.log1p(horizon_hours / 24.0)
            for k in range(draws):
                growth = max(math.exp(growth_centre + 0.6 * shared_z[k]), 1.0)
                players_draw = seed_players * growth
                if rate >= 99.5:
                    g_draw = 1.0 - abs(rng.gauss(0.0, 0.002))
                else:
                    g_draw = from_logit(
                        to_logit(max(rate, 0.05) / 100.0) - 2.5 + mixed_noise(k, 1.0))
                earned_draw = start_earned + g_draw * (players_draw - seed_players)
                samples[i][k] = clamp_percent(100.0 * earned_draw / players_draw)

    # Ordering rules, imposed on every draw.
    position = {tid: i for i, tid in enumerate(ids)}
    chain = [position[t] for t in story_chain_ids(trophies) if t in position]
    platinum = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    for k in range(draws):
        for earlier, later in zip(chain, chain[1:]):
            if samples[later][k] > samples[earlier][k]:
                samples[later][k] = samples[earlier][k]
        for pi in platinum:
            lowest = min(samples[j][k] for j in range(n) if j != pi)
            if samples[pi][k] > lowest:
                samples[pi][k] = lowest

    rows = []
    for i in range(n):
        ordered = sorted(samples[i])
        median_logit = to_logit(
            min(max(empirical_quantile(ordered, 0.5), RATE_FLOOR), RATE_CEIL) / 100.0)
        qs = []
        for tau in QUANTILES:
            q = empirical_quantile(ordered, tau)
            if tau in (0.05, 0.95) and cfg["tail_stretch"] != 1.0:
                q_logit = to_logit(min(max(q, RATE_FLOOR), RATE_CEIL) / 100.0)
                q = 100.0 * from_logit(
                    median_logit + (q_logit - median_logit) * cfg["tail_stretch"])
            qs.append(q)
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        qs = [round(clamp_percent(q), 4) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        # Near-floor trophies: nudge up and widen the upper tail (wavefront timing).
        if cfg["low_shift"] > 0 and median_logit < -3.0:
            qs = [100.0 * from_logit(
                to_logit(min(max(q, RATE_FLOOR), RATE_CEIL) / 100.0) + cfg["low_shift"])
                for q in qs]
            median_logit += cfg["low_shift"]
        if cfg["low_uptail"] > 0 and median_logit < -3.0:
            widened = []
            for tau, q in zip(QUANTILES, qs):
                if tau >= 0.75:
                    q_logit = to_logit(min(max(q, RATE_FLOOR), RATE_CEIL) / 100.0)
                    q = 100.0 * from_logit(
                        median_logit + (q_logit - median_logit) * (1.0 + cfg["low_uptail"]))
                widened.append(q)
            qs = [round(clamp_percent(q), 4) for q in widened]
            for j in range(1, 5):
                if qs[j] < qs[j - 1]:
                    qs[j] = qs[j - 1]
        if cfg["pin_full"] and latest_rates[i] is not None and latest_rates[i] >= 99.5:
            qs = [99.85, 99.95, 99.99, 100.0, 100.0]
        rows.append({"id": ids[i], "q": qs})
    return {"trophies": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        (output_dir / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
