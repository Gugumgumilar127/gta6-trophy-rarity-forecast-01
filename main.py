"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for every Grand Theft Auto VI PS5 trophy, the quantiles of its
Exophase earn rate (percent of tracked players who have earned it) at the
case's target instant, scored by pinball loss on the logit scale.

Model (see README.md for the full write-up and the evidence behind it):

  * Work on the logit scale in ln(1 + hours since release) time, the clock
    trophy curves actually run on: earn rates climb steeply in the first
    days after launch and flatten as ln-time grows.
  * Collapse the read history to refresh points (consecutive reads with the
    same players-tracked count and the same rates are one Exophase refresh,
    not new information).
  * Per trophy, fit a weighted least squares line of logit(rate) on
    ln(1 + hours), weights sqrt(players_tracked) (bigger samples quantize
    less), then shrink the slope toward the pooled slope of the trophy's
    tier (platinum/gold/silver/bronze): 55 noisy short series borrow
    strength from each other instead of extrapolating alone.
  * The median blends the fitted line with the last observed level carried
    forward at the shrunken slope, so a level shift at the newest refresh
    is respected instead of averaged away.
  * The spread is honest, not decorative: pooled refresh-to-refresh logit
    volatility scaled by the square root of the horizon, plus the fit's
    residual variance, plus a fixed refresh-jump term (Exophase republishes
    counts in steps, and a step can move every rate at once), plus the
    counting noise of the projected earner count at the target.
  * Trophies with too little history take the pooled slope from their last
    observed rate; a case with no history at all still writes a valid,
    deliberately small forecast rather than crashing.

Deterministic: pure float arithmetic on the case file, files read in name
order, no randomness, no clock, no network. Runs in well under a second.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Spread floors and caps (logit units). The volatility floor keeps the band
# honest even when the observed history happens to be frozen; the caps keep
# one wild series from blowing up its own forecast.
SIGMA_RW_FLOOR = 0.30
SIGMA_RW_CAP = 1.50
SIGMA_TOTAL_CAP = 3.00
JUMP_VAR = 0.04  # 0.2 logit of refresh-step risk, whatever the horizon
RESID_FLOOR = 0.02


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid_pct(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (the starter's rounding, kept so our
    numbers print the way the house's do)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation,
    |error| < 1.15e-9 in the central range), so any level the case asks
    for gets its z without a lookup table."""
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def release_origin(case: dict, first_slot: datetime) -> datetime:
    """The clock trophy growth runs on starts at release midnight UTC."""
    try:
        rd = case["game"]["release_date"]
        return datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
    except Exception:
        return first_slot


def hours_since(origin: datetime, when: datetime) -> float:
    return max(0.0, (when - origin).total_seconds() / 3600.0)


class TrophySeries:
    """One trophy's refresh-point series and its fitted forecast parts."""

    def __init__(self, tier: str):
        self.tier = tier
        self.x: list[float] = []       # ln(1 + hours since release)
        self.y: list[float] = []       # logit(rate in percent)
        self.w: list[float] = []       # sqrt(players_tracked)
        self.players: list[float] = [] # players_tracked at each point
        self.last_rate: float | None = None
        # Filled by the fit pass.
        self.has_fit = False
        self.x_bar = 0.0
        self.y_bar = 0.0
        self.slope = 0.0
        self.info = 0.0                # weighted x-variance: fit information
        self.resid_var = RESID_FLOOR


def build_series(case: dict) -> tuple[list[TrophySeries], float, float]:
    """Collapse the history to refresh points and build per-trophy series.

    Returns (series per trophy, x at the checkpoint, x at the target).
    """
    trophies = case["trophies"]
    history = case.get("history") or []
    series = [TrophySeries(t.get("tier", "")) for t in trophies]
    if not history:
        return series, 0.0, 0.0
    slots = [parse_time(read["slot"]) for read in history]
    origin = release_origin(case, slots[0])
    x_at = math.log1p(hours_since(origin, parse_time(case["at"])))
    x_target = math.log1p(hours_since(origin, parse_time(case["target"])))

    prev_key = None
    for read, slot in zip(history, slots):
        rates = read.get("rates") or []
        players = read.get("players_tracked") or 0
        key = (players, tuple(rates))
        if key == prev_key:
            continue  # same Exophase refresh re-read: no new information
        prev_key = key
        x = math.log1p(hours_since(origin, slot))
        w = math.sqrt(players) if players and players > 0 else 1.0
        for i, s in enumerate(series):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            s.x.append(x)
            s.y.append(logit(rate))
            s.w.append(w)
            s.players.append(float(players))
            s.last_rate = rate
    return series, x_at, x_target


def fit_series(s: TrophySeries) -> None:
    """Weighted least squares of logit(rate) on ln(1 + hours)."""
    n = len(s.x)
    if n == 0:
        return
    sw = sum(s.w)
    if sw <= 0:
        return
    s.x_bar = sum(wi * xi for wi, xi in zip(s.w, s.x)) / sw
    s.y_bar = sum(wi * yi for wi, yi in zip(s.w, s.y)) / sw
    sxx = sum(wi * (xi - s.x_bar) ** 2 for wi, xi in zip(s.w, s.x))
    s.info = sxx
    if n >= 2 and sxx > 1e-12:
        sxy = sum(wi * (xi - s.x_bar) * (yi - s.y_bar)
                  for wi, xi, yi in zip(s.w, s.x, s.y))
        s.slope = sxy / sxx
        s.has_fit = True
        s.resid_var = max(
            RESID_FLOOR,
            sum(wi * (yi - (s.y_bar + s.slope * (xi - s.x_bar))) ** 2
                for wi, xi, yi in zip(s.w, s.x, s.y)) / sw,
        )


def pooled_slopes(series: list[TrophySeries]) -> tuple[dict[str, float], float]:
    """Information-weighted mean slope per tier, and overall."""
    by_tier: dict[str, list[TrophySeries]] = {}
    for s in series:
        if s.has_fit:
            by_tier.setdefault(s.tier, []).append(s)

    def pool(group: list[TrophySeries]) -> float:
        info = sum(s.info for s in group)
        if info <= 1e-12:
            return 0.0
        return sum(s.info * s.slope for s in group) / info

    fitted = [s for s in series if s.has_fit]
    global_pool = pool(fitted)
    return {tier: pool(group) for tier, group in by_tier.items()}, global_pool


def pooled_volatility(series: list[TrophySeries]) -> float:
    """Refresh-to-refresh logit variance rate (per day), pooled over all
    trophies: sum of squared logit steps over sum of elapsed days. Steps
    include Exophase's republication jumps, which is the point: the band
    must price the risk that the page moves in steps, not smoothly."""
    num = 0.0
    den = 0.0
    for s in series:
        for j in range(1, len(s.x)):
            dx = s.x[j] - s.x[j - 1]
            if dx <= 0:
                continue
            # x is ln(1 + hours): invert to hours to get elapsed days.
            dt_days = (math.expm1(s.x[j]) - math.expm1(s.x[j - 1])) / 24.0
            if dt_days <= 0:
                continue
            num += (s.y[j] - s.y[j - 1]) ** 2
            den += dt_days
    if den <= 0:
        return SIGMA_RW_FLOOR
    sigma = math.sqrt(num / den)
    return min(SIGMA_RW_CAP, max(SIGMA_RW_FLOOR, sigma))


def projected_players(s: TrophySeries, x_target: float) -> float:
    """Project the tracked-player count at the target by a plain least
    squares line of ln(players) on ln(1 + hours); used only to size the
    counting noise of the forecast rates, and clamped to sane growth."""
    if not s.players:
        return 0.0
    last = s.players[-1]
    n = len(s.x)
    if n < 2:
        return last
    x_bar = sum(s.x) / n
    lp = [math.log(max(1.0, p)) for p in s.players]
    lp_bar = sum(lp) / n
    sxx = sum((xi - x_bar) ** 2 for xi in s.x)
    if sxx <= 1e-12:
        return last
    slope = sum((xi - x_bar) * (li - lp_bar) for xi, li in zip(s.x, lp)) / sxx
    pred = math.exp(lp_bar + slope * (x_target - x_bar))
    return min(max(pred, last), 50.0 * last)


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = [norm_quantile(tau) for tau in levels]
    trophies = case["trophies"]
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    series, _x_at, x_target = build_series(case)

    if not case.get("history"):
        # No reads at all: a valid, deliberately small forecast. There is
        # nothing to fit; crashing would score worse than any guess.
        rows = [{"id": t["id"], "q": [0.1, 0.5, 1.0, 2.0, 5.0]}
                for t in trophies]
        return {"trophies": rows}

    for s in series:
        fit_series(s)
    tier_pool, global_pool = pooled_slopes(series)
    sigma_rw = pooled_volatility(series)
    infos = sorted(s.info for s in series if s.has_fit)
    shrink_k = infos[len(infos) // 2] if infos else 1.0

    rows = []
    for s, trophy in zip(series, trophies):
        pool = tier_pool.get(s.tier, global_pool)
        if s.has_fit:
            w_own = s.info / (s.info + shrink_k) if (s.info + shrink_k) > 0 else 0.0
            slope = w_own * s.slope + (1.0 - w_own) * pool
            line_pred = s.y_bar + slope * (x_target - s.x_bar)
            last_pred = s.y[-1] + slope * (x_target - s.x[-1])
            median_logit = 0.5 * line_pred + 0.5 * last_pred
            resid_var = s.resid_var
        elif s.y:
            # Too few points to fit: pooled growth from the last level.
            median_logit = s.y[-1] + pool * (x_target - s.x[-1])
            resid_var = RESID_FLOOR
        else:
            # Never observed (null in every read): the tier's typical
            # level is unknowable; answer small rather than crash.
            median_logit = logit(0.05)
            resid_var = RESID_FLOOR

        n_proj = projected_players(s, x_target)
        p_med = sigmoid_pct(median_logit) / 100.0
        count_var = 0.0
        if n_proj > 0:
            earners = max(0.5, n_proj * p_med)
            others = max(0.5, n_proj * (1.0 - p_med))
            count_var = min(1.0, 1.0 / earners + 1.0 / others)
        sigma = math.sqrt(sigma_rw ** 2 * horizon_days + resid_var
                          + JUMP_VAR + count_var)
        sigma = min(SIGMA_TOTAL_CAP, sigma)

        q = []
        for z in zs:
            q.append(round4(sigmoid_pct(median_logit + z * sigma)))
        # Rounding can in principle tie neighbours; the contract wants
        # non-decreasing, so enforce it after rounding.
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
