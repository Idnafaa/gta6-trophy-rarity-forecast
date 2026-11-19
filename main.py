"""Week-one trophy earn-rate forecaster for GTA VI on Exophase (make.fun gta6-trophy-rarity-forecast-01).

Python 3.13, standard library only, no network. Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output).

The model, in four steps:

1. Refresh events. Exophase recomputes a game's counts on its own schedule, so consecutive house reads are often the
   same page. Reads are collapsed into "events": one per distinct page (players tracked + every rate).

2. Counts, not rates. A rate is earned / tracked. Launch night was a few hundred players, a handful of whom (review or
   early copies) already held every trophy, so early rates were inflated and then diluted as players arrived. Each
   event's rate is turned into an earned count, and the launch-night page is kept apart as a fixed "seed" cohort.
   What is modelled is the cohort rate g of everyone who arrived after it: (earned - seed earned) / (tracked - seed).

3. Trend. For each trophy, logit(g) is fitted against ln(hours since launch + c) over the recent events by weighted
   least squares (recent events and bigger cohorts weigh more), the slope is shrunk toward a slope pooled across
   trophies at a similar level, damped, and carried to the target. Players tracked is extrapolated the same way on
   ln N. The forecast rate mixes the seed cohort back in: (seed earned + g * (N - seed)) / N.

4. Spread. Quantiles come from a seeded Monte Carlo over the uncertain pieces: the trend's level and slope, a
   random-walk term that grows with the horizon, the count noise of rare trophies, and the growth of players tracked.
   Ordering rules of the trophy list are applied to the draws (the platinum needs every trophy, a later chapter
   can not be more common than an earlier one), then the 5/25/50/75/95% points are read off.

When the history holds no usable growth yet (every read is the launch-night page), a dilution prior is used instead.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
LAUNCH = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)  # midnight in New Zealand, first region to unlock
LO, HI = 0.01, 99.99

# Tuned on simulated launch weeks (see README); kept as plain constants so the run is deterministic.
PARAMS = {
    "c_hours": 3.0,        # offset in ln(hours + c)
    "window_h": 40.0,      # events older than this before the last one are ignored by the trend fit
    "decay_h": 18.0,       # recency weight e-folding time
    "damp": 0.80,          # multiplier on the fitted slope
    "n_damp": 0.75,        # multiplier on the players-tracked slope
    "shrink_k": 2.0,       # pseudo-weight of the pooled slope
    "rw": 0.16,            # random-walk sd per unit of ln-time on the logit scale
    "slope_sd_mult": 1.0,  # inflation of the slope's standard error
    "base_sd": 0.10,       # floor sd on the logit scale
    "tail": 1.12,          # extra width at the 5% and 95% levels (fat tails)
    "rho": 0.7,            # share of each draw's noise common to every trophy
    "draws": 3000,
}


def parse_time(text):
    text = text.replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def hours_since_launch(when):
    return (when - LAUNCH).total_seconds() / 3600.0


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def expit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp_pct(x):
    return min(max(x, 0.0), 100.0)


def collapse_events(history, n_trophies):
    """One event per distinct page, in time order. Each event: (hours, players, [rate or None])."""
    events = []
    last_key = None
    for read in history:
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if not players or len(rates) != n_trophies:
            continue
        key = (players, tuple(rates))
        if key == last_key:
            continue
        last_key = key
        events.append((hours_since_launch(parse_time(read["at"])), float(players), list(rates)))
    return events


def wls(xs, ys, ws):
    """Weighted least squares line. Returns (intercept at x_ref=last x, slope, residual var, slope var)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    n_eff = sw * sw / sum(w * w for w in ws)
    dof = max(n_eff - 2.0, 1.0)
    var_res = resid * n_eff / dof
    slope_var = var_res / (sxx / sw) / max(n_eff, 1.0)
    return a, b, var_res, slope_var


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def is_chapter(trophy):
    desc = (trophy.get("description") or "").lower()
    return desc.startswith("complete chapter")


def chapter_chain(trophies):
    """Ids of the story chain in order: Chapter 1..n, then the final mission (rates can only fall along it)."""
    chapters = []
    final = None
    for t in trophies:
        desc = (t.get("description") or "").strip().lower()
        if desc.startswith("complete chapter"):
            digits = "".join(ch for ch in desc if ch.isdigit())
            if digits:
                chapters.append((int(digits), t["id"]))
        elif desc == "complete the final mission.":
            final = t["id"]
    chain = [tid for _, tid in sorted(chapters)]
    if final is not None:
        chain.append(final)
    return chain


def forecast(case, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    trophies = case["trophies"]
    n = len(trophies)
    ids = [t["id"] for t in trophies]
    target_h = hours_since_launch(parse_time(case["target"]))
    at_h = hours_since_launch(parse_time(case["at"]))
    events = collapse_events(case.get("history") or [], n)

    last_rates = [None] * n
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        if len(rates) == n:
            last_rates = list(rates)
            break
    last_rates = [r if isinstance(r, (int, float)) else None for r in last_rates]

    rng = random.Random(20261126)
    draws = int(p["draws"])
    samples = [[0.0] * draws for _ in range(n)]
    # Draws share a common part across trophies (one week, one player base), so the ordering rules below rarely
    # bind by chance; each trophy's marginal spread is unchanged by it.
    shared_z = [rng.gauss(0.0, 1.0) for _ in range(draws)]
    shared_g = [rng.gauss(0.0, 1.0) for _ in range(draws)]
    rho = p["rho"]

    def mix(k, sd):
        return sd * (math.sqrt(rho) * shared_g[k] + math.sqrt(1.0 - rho) * rng.gauss(0.0, 1.0))

    c = p["c_hours"]
    x_target = math.log(max(target_h, 0.5) + c)

    seed_t, seed_n, seed_rates = (events[0] if events else (at_h, 1.0, last_rates))
    seed_counts = [((r or 0.0) / 100.0) * seed_n for r in seed_rates]

    # Events usable for the cohort trend: enough players beyond the seed cohort.
    usable = [e for e in events[1:] if e[1] - seed_n >= max(150.0, 0.25 * seed_n)]

    if len(usable) >= 1:
        last = usable[-1]
        x_last = math.log(max(last[0], 0.5) + c)
        dx = max(x_target - x_last, 0.0)
        window = [e for e in usable if last[0] - e[0] <= p["window_h"]]

        # Players tracked: ln N against ln-time.
        nx = [math.log(max(e[0], 0.5) + c) for e in window]
        ny = [math.log(e[1]) for e in window]
        nw = [math.exp(-(last[0] - e[0]) / p["decay_h"]) for e in window]
        fitn = wls(nx, ny, nw) if len(window) >= 2 else None
        n_slope = max(fitn[1], 0.0) if fitn else 1.0
        n_slope_sd = math.sqrt(fitn[3]) + 0.3 * n_slope if fitn else 1.0
        ln_n_last = math.log(last[1])

        # Per-trophy cohort fits.
        fits = []
        for i in range(n):
            xs, ys, ws, cvars = [], [], [], []
            for e in window:
                r = e[2][i]
                if r is None:
                    continue
                cohort = e[1] - seed_n
                earned = (r / 100.0) * e[1] - seed_counts[i]
                g = (max(earned, 0.0) + 0.5) / (cohort + 1.0)
                g = min(max(g, 1e-5), 1 - 1e-5)
                var_count = 1.0 / max(cohort * g * (1 - g), 0.25)
                # Rounding the printed rate to 2 decimals: +-0.005 percentage points of the whole page.
                var_round = (0.00005 * e[1] / max(cohort, 1.0)) ** 2 / max(g * (1 - g), 1e-6) ** 2 / 3.0
                xs.append(math.log(max(e[0], 0.5) + c))
                ys.append(logit(g))
                cvars.append(var_count + var_round)
                ws.append(math.exp(-(last[0] - e[0]) / p["decay_h"]) / (1.0 + var_count + var_round))
            if not ys:
                fits.append(None)
                continue
            y_last = ys[-1]
            v_last = cvars[-1]
            fit = wls(xs, ys, ws) if len(xs) >= 3 else None
            if fit:
                a, b, var_res, slope_var = fit
                level_fit = a + b * xs[-1]
                # Blend the observed last level with the fitted line by their precisions.
                prec_obs = 1.0 / max(v_last, 1e-6)
                prec_fit = 1.0 / max(var_res, 1e-4)
                level = (y_last * prec_obs + level_fit * prec_fit) / (prec_obs + prec_fit)
                level_var = 1.0 / (prec_obs + prec_fit)
                fits.append({"level": level, "level_var": level_var, "b": b, "b_var": slope_var,
                             "res": var_res, "y": y_last})
            else:
                fits.append({"level": y_last, "level_var": v_last, "b": None, "b_var": None,
                             "res": 0.04, "y": y_last})

        # Pooled slope as a function of level: weighted line of b on level across good fits.
        px, py, pw = [], [], []
        for f in fits:
            if f and f["b"] is not None and f["b_var"] is not None:
                px.append(f["level"])
                py.append(f["b"])
                pw.append(1.0 / (f["b_var"] + 0.05))
        pool = wls(px, py, pw) if len(px) >= 4 else None
        if pool:
            pa, pb = pool[0] - pool[1] * px[-1], pool[1]
        else:
            pa, pb = (sum(py) / len(py) if py else 0.3), 0.0
        pool_var = pool[2] if pool else 0.25

        for i in range(n):
            f = fits[i]
            lr = last_rates[i]
            if f is None:
                base = logit((lr if lr is not None else 1.0) / 100.0)
                for k in range(draws):
                    samples[i][k] = 100.0 * expit(base + rng.gauss(0.0, 1.0))
                continue
            b_pool = pa + pb * f["level"]
            if f["b"] is not None:
                wi = 1.0 / (f["b_var"] + 1e-4)
                wp = p["shrink_k"] / (pool_var + 0.02)
                b = (f["b"] * wi + b_pool * wp) / (wi + wp)
                b_sd = math.sqrt(1.0 / (wi + wp)) * p["slope_sd_mult"] + 0.15 * abs(b)
            else:
                b = b_pool
                b_sd = math.sqrt(pool_var) + 0.2 * abs(b)
            b *= p["damp"]
            sd_static = math.sqrt(f["level_var"] + p["base_sd"] ** 2)
            sd_rw = p["rw"] * math.sqrt(dx) + 0.0
            sd_total = math.sqrt((b_sd * dx) ** 2 + sd_static ** 2 + sd_rw ** 2)
            for k in range(draws):
                yk = f["level"] + b * dx + mix(k, sd_total)
                gk = expit(yk)
                ln_nk = ln_n_last + max(n_slope * p["n_damp"] + shared_z[k] * n_slope_sd, 0.0) * dx
                nk = math.exp(ln_nk)
                cohort_k = max(nk - seed_n, 1.0)
                # Count noise at the target for rare trophies.
                earned = seed_counts[i] + gk * cohort_k
                if earned < 50:
                    earned = max(earned + rng.gauss(0.0, math.sqrt(max(earned, 0.5))), 0.0)
                samples[i][k] = clamp_pct(100.0 * earned / nk)
    else:
        # Launch-night only: the page has not refreshed yet. Players arrive and dilute the early completers.
        horizon_h = max(target_h - at_h, 0.0)
        for i in range(n):
            lr = last_rates[i]
            if lr is None:
                lr = 1.0
            e0 = seed_counts[i] if events else (lr / 100.0) * seed_n
            # Launch night: Exophase's first refresh multiplies players tracked several times over.
            # Its size is not in the history, so it is drawn wide around a fourfold rise.
            grow_mu = math.log(4.0) + 0.1 * math.log1p(horizon_h / 24.0)
            for k in range(draws):
                growth = max(math.exp(grow_mu + 0.6 * shared_z[k]), 1.0)
                nk = seed_n * growth
                if lr >= 99.5:
                    gk = 1.0 - abs(rng.gauss(0.0, 0.002))
                else:
                    # A new arrival rarely holds a trophy within hours of starting.
                    gk = expit(logit(max(lr, 0.05) / 100.0) - 2.5 + mix(k, 1.0))
                earned = e0 + gk * (nk - seed_n)
                samples[i][k] = clamp_pct(100.0 * earned / nk)

    # Ordering rules, applied draw by draw.
    index = {tid: i for i, tid in enumerate(ids)}
    chain = [index[t] for t in chapter_chain(trophies) if t in index]
    plat = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    for k in range(draws):
        for a_i, b_i in zip(chain, chain[1:]):
            if samples[b_i][k] > samples[a_i][k]:
                samples[b_i][k] = samples[a_i][k]
        for pi in plat:
            low = min(samples[j][k] for j in range(n) if j != pi)
            if samples[pi][k] > low:
                samples[pi][k] = low

    rows = []
    for i in range(n):
        vals = sorted(samples[i])
        med_l = logit(min(max(quantile(vals, 0.5), LO), HI) / 100.0)
        qs = []
        for tau in LEVELS:
            q = quantile(vals, tau)
            if tau in (0.05, 0.95) and p["tail"] != 1.0:
                ql = logit(min(max(q, LO), HI) / 100.0)
                q = 100.0 * expit(med_l + (ql - med_l) * p["tail"])
            qs.append(q)
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        qs = [round(clamp_pct(q), 4) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": ids[i], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        (out_dir / (path.stem + ".out")).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
