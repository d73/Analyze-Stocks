import json
import os
import sys
from datetime import date

TICKERS = sorted(set(t.replace(".", "-") for t in sys.argv[1:]))
DATAFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datafiles")


def _v(val, default=0):
    """Return val if not None, else default. Prevents TypeError on missing data."""
    return default if val is None else val


def load_market_data() -> dict:
    with open(os.path.join(DATAFILES_DIR, "__market.json")) as f:
        return json.load(f)


def load_tickers_data() -> dict[str, dict]:
    with open(os.path.join(DATAFILES_DIR, "__tickers.json")) as f:
        rows = json.load(f)
    return {row["ticker"]: row for row in rows}


def calculate_regime_score(market: dict) -> dict:
    """
    Score the current market regime on a 0–6 scale.

    +1 if sp500_vs_200d_pct > 0        (price above 200-day SMA)
    +1 if sp500_200d_trend == "up"     (200-day SMA itself is rising)
    +1 if vix < 20                     (low volatility)
    +1 if hyg_trend == "up"            (high-yield credit trending up)
    +1 if fear_greed_score > 35        (sentiment not extreme fear)
    +1 if vvix_percentile < 70         (vol-of-vol not elevated)

    Returns a dict with the total score and a breakdown of each component.
    """
    components = {
        "sp500_above_200d":   _v(market.get("sp500_vs_200d_pct")) > 0,
        "sp500_200d_uptrend": market.get("sp500_200d_trend") == "up",
        "vix_low":            _v(market.get("vix"), 999) < 20,
        "hyg_uptrend":        market.get("hyg_trend") == "up",
        "fear_greed_ok":      _v(market.get("fear_greed_score")) > 35,
        "vvix_calm":          _v(market.get("vvix_percentile"), 100) < 70,
    }

    score = sum(components.values())

    return {
        "regime_score": score,
        "components": {k: int(v) for k, v in components.items()},
    }


def classify_regime(regime_score: int) -> dict:
    """
    Classify the regime score into a label.

    5–6 → Good
    3–4 → Neutral
    0–2 → Bad
    """
    if regime_score >= 5:
        label = "Good"
    elif regime_score >= 3:
        label = "Neutral"
    else:
        label = "Bad"

    return {"regime_label": label}


def score_ticker_trend(ticker: dict) -> dict:
    """
    Score a ticker's trend strength on a 0–25 scale.

    +10 if price_vs_200d_sma_pct > 0   (price above 200-day SMA)
    +5  if sma_200d_direction == "up"  (200-day SMA is rising)
    +5  if price_vs_50d_sma_pct > 0    (price above 50-day SMA)
    +5  if sma_50d_direction == "up"   (50-day SMA is rising)

    Interpretation:
        20–25 → strong uptrend
        <10   → weak / avoid
    """
    components = {
        "above_200d_sma":    (_v(ticker.get("price_vs_200d_sma_pct")) > 0, 10),
        "200d_sma_rising":   (_v(ticker.get("sma_200d_direction"), "") == "up", 5),
        "above_50d_sma":     (_v(ticker.get("price_vs_50d_sma_pct")) > 0, 5),
        "50d_sma_rising":    (_v(ticker.get("sma_50d_direction"), "") == "up", 5),
    }

    score = sum(pts for passed, pts in components.values() if passed)

    if score >= 20:
        label = "strong uptrend"
    elif score < 10:
        label = "weak / avoid"
    else:
        label = "moderate"

    return {
        "ticker": ticker["ticker"],
        "trend_score": score,
        "trend_label": label,
        "components": {k: int(passed) * pts for k, (passed, pts) in components.items()},
    }


def score_ticker_momentum(ticker: dict) -> dict:
    """
    Score a ticker's momentum on a 0–20 scale.

    RSI (rsi_14):
        +8 if 50–65   (ideal momentum zone)
        +5 if 65–75   (strong but not overbought)
        +2 if 40–50   (recovering)
         0 otherwise

    MACD (stackable, up to +12):
        +6 if macd_crossover == "bullish"
        +4 if macd_histogram > 0
        +2 if macd > macd_signal_line
    """
    rsi = _v(ticker.get("rsi_14"))

    if 50 <= rsi <= 65:
        rsi_score = 8
    elif 65 < rsi <= 75:
        rsi_score = 5
    elif 40 <= rsi < 50:
        rsi_score = 2
    else:
        rsi_score = 0

    macd_components = {
        "macd_crossover_bullish": (_v(ticker.get("macd_crossover"), "") in ("bullish", "bullish_cross"), 6),
        "macd_histogram_positive": (_v(ticker.get("macd_histogram")) > 0, 4),
        "macd_above_signal": (_v(ticker.get("macd")) > _v(ticker.get("macd_signal_line")), 2),
    }
    macd_score = sum(pts for passed, pts in macd_components.values() if passed)

    score = rsi_score + macd_score

    return {
        "ticker": ticker["ticker"],
        "momentum_score": score,
        "components": {
            "rsi": rsi_score,
            **{k: int(passed) * pts for k, (passed, pts) in macd_components.items()},
        },
    }


def score_ticker_range_position(ticker: dict) -> dict:
    """
    Score a ticker's position within its 52-week range on a 0–15 scale.

    Uses 52w_range_position_pct (0% = 52w low, 100% = 52w high).

        +10 if 40%–80%   (healthy mid-range, not extended)
        +5  if 20%–40%   (recovering from lows)
        +2  if >80%      (extended / near highs)
         0  if <20%      (falling knife risk)
    """
    pos = _v(ticker.get("52w_range_position_pct"))

    if 40 <= pos <= 80:
        score = 10
    elif 20 <= pos < 40:
        score = 5
    elif pos > 80:
        score = 2
    else:
        score = 0

    return {
        "ticker": ticker["ticker"],
        "range_position_score": score,
        "52w_range_position_pct": pos,
    }


def score_ticker_volume(ticker: dict) -> dict:
    """
    Score a ticker's volume confirmation on a 0–10 scale.

    Uses volume_vs_avg_ratio (current volume / 20-day avg volume).

        +10 if > 1.2    (above-average volume, confirms move)
        +5  if 0.8–1.2  (normal volume)
         0  if < 0.8    (low volume, weak conviction)
    """
    ratio = _v(ticker.get("volume_vs_avg_ratio"))

    if ratio > 1.2:
        score = 10
    elif ratio >= 0.8:
        score = 5
    else:
        score = 0

    return {
        "ticker": ticker["ticker"],
        "volume_score": score,
        "volume_vs_avg_ratio": ratio,
    }


def score_ticker_relative_strength(ticker: dict) -> dict:
    """
    Score a ticker's relative strength vs SPY on a 0–10 scale.

    Uses rs_vs_spy_20d (ticker 20d return / SPY 20d return).

        +10 if > 1.02   (outperforming SPY)
        +5  if 0.98–1.02 (in-line with SPY)
         0  if < 0.98   (underperforming SPY)
    """
    rs = _v(ticker.get("rs_vs_spy_20d"))

    if rs > 1.02:
        score = 10
    elif rs >= 0.98:
        score = 5
    else:
        score = 0

    return {
        "ticker": ticker["ticker"],
        "relative_strength_score": score,
        "rs_vs_spy_20d": rs,
    }


def score_ticker_event_risk(ticker: dict, today: date = None) -> dict:
    """
    Apply an event risk penalty based on proximity to earnings date (-10 to 0).

        -10 if earnings_date within 7 days
        -5  if earnings_date within 14 days
         0  otherwise (or if earnings_date is null)
    """
    if today is None:
        today = date.today()

    earnings_str = ticker.get("earnings_date")
    if not earnings_str:
        return {"ticker": ticker["ticker"], "event_risk_penalty": 0, "earnings_date": None}

    earnings = date.fromisoformat(earnings_str)
    days_until = (earnings - today).days

    if 0 <= days_until <= 7:
        penalty = -10
    elif 0 <= days_until <= 14:
        penalty = -5
    else:
        penalty = 0

    return {
        "ticker": ticker["ticker"],
        "event_risk_penalty": penalty,
        "earnings_date": earnings_str,
        "days_until_earnings": days_until,
    }


def score_ticker_total(ticker: dict, today: date = None) -> dict:
    """
    Composite ticker score combining all sub-scores.

    Max possible: 25 (trend) + 20 (momentum) + 15 (range) + 10 (volume) + 10 (RS) = 80
    Event risk penalty: -10 to 0
    Effective range: -10 to 80
    """
    trend     = score_ticker_trend(ticker)
    momentum  = score_ticker_momentum(ticker)
    range_pos = score_ticker_range_position(ticker)
    volume    = score_ticker_volume(ticker)
    rs        = score_ticker_relative_strength(ticker)
    event     = score_ticker_event_risk(ticker, today)

    total = (
        trend["trend_score"]
        + momentum["momentum_score"]
        + range_pos["range_position_score"]
        + volume["volume_score"]
        + rs["relative_strength_score"]
        + event["event_risk_penalty"]
    )

    return {
        "ticker": ticker["ticker"],
        "total_score": total,
        "breakdown": {
            "trend":            trend["trend_score"],
            "momentum":         momentum["momentum_score"],
            "range_position":   range_pos["range_position_score"],
            "volume":           volume["volume_score"],
            "relative_strength": rs["relative_strength_score"],
            "event_risk":       event["event_risk_penalty"],
        },
    }


def decide_ticker_action(ticker: dict, today: date = None) -> dict:
    """
    Final decision for a ticker combining the composite score with hard-rule overrides.

    Primary action (score-based):
        80–100  → Aggressive Buy / Add
        65–79   → Buy
        50–64   → Hold
        35–49   → Trim
        <35     → Exit / Avoid

    Avoid Entry if any of:
        - price_vs_50d_sma_pct < -3 AND sma_50d_direction == "down"
        - volume_vs_avg_ratio < 0.7
        - earnings_date within 7 days

    Force Exit if any of:
        - RSI > 75 AND 52w_range_position_pct > 90
        - macd_histogram < 0 AND macd_crossover in ("bearish", "bearish_cross")
    """
    if today is None:
        today = date.today()

    scored = score_ticker_total(ticker, today)
    score = scored["total_score"]

    if score >= 80:  # 80 is the maximum possible score
        primary_action = "Aggressive Buy / Add"
    elif score >= 65:
        primary_action = "Buy"
    elif score >= 50:
        primary_action = "Hold"
    elif score >= 35:
        primary_action = "Trim"
    else:
        primary_action = "Exit / Avoid"

    # --- secondary filters ---
    avoid_reasons = []
    exit_reasons = []

    if _v(ticker.get("price_vs_50d_sma_pct")) < -3 and _v(ticker.get("sma_50d_direction"), "") == "down":
        avoid_reasons.append("price below 50d SMA and SMA trending down")

    if _v(ticker.get("volume_vs_avg_ratio")) < 0.7:
        avoid_reasons.append("volume too low (< 0.7x avg)")

    earnings_str = ticker.get("earnings_date")
    if earnings_str:
        days_until = (date.fromisoformat(earnings_str) - today).days
        if 0 <= days_until <= 7:
            avoid_reasons.append(f"earnings in {days_until} day(s)")

    if _v(ticker.get("rsi_14")) > 75 and _v(ticker.get("52w_range_position_pct")) > 90:
        exit_reasons.append("RSI > 75 and price near 52w high (>90%)")

    if _v(ticker.get("macd_histogram")) < 0 and _v(ticker.get("macd_crossover"), "") in ("bearish", "bearish_cross"):
        exit_reasons.append("MACD histogram negative with bearish crossover")

    # --- resolve final action ---
    if exit_reasons:
        final_action = "Force Exit"
    elif avoid_reasons and primary_action in ("Aggressive Buy / Add", "Buy"):
        final_action = "Avoid Entry"
    else:
        final_action = primary_action

    return {
        "ticker": ticker["ticker"],
        "total_score": score,
        "primary_action": primary_action,
        "final_action": final_action,
        "avoid_entry_reasons": avoid_reasons,
        "force_exit_reasons": exit_reasons,
        "breakdown": scored["breakdown"],
    }


def calculate_position_size(decision: dict, regime_label: str) -> dict:
    """
    Calculate final position size as a multiplier on 1 base unit.

    Final size = regime_multiplier * score_multiplier

    Regime multiplier (from classify_regime):
        Good    → 1.00x
        Neutral → 0.50x
        Bad     → 0.25x

    Score multiplier (from total_score):
        80+     → 1.25x
        65–79   → 1.00x
        50–64   → 0.50x
        <50     → 0.00x  (no position)
    """
    regime_multipliers = {"Good": 1.0, "Neutral": 0.5, "Bad": 0.25}
    regime_mult = regime_multipliers.get(regime_label, 0.0)

    score = decision["total_score"]
    if score >= 80:
        score_mult = 1.25
    elif score >= 65:
        score_mult = 1.0
    elif score >= 50:
        score_mult = 0.5
    else:
        score_mult = 0.0

    position_size = regime_mult * score_mult

    return {
        "ticker": decision["ticker"],
        "position_size": position_size,
        "regime_multiplier": regime_mult,
        "score_multiplier": score_mult,
        "regime_label": regime_label,
        "total_score": score,
    }


def check_entry_timing(ticker: dict) -> dict:
    """
    Evaluate entry timing triggers. Entry is valid if at least one trigger fires.

    Pullback:  price within -3% to -1% of 50d SMA
    Breakout:  volume_vs_avg_ratio > 1.5 AND pct_from_52w_high >= 0 (new 52w high)
    Momentum:  macd_crossover == "bullish"
               (proxy for crossover within last 3 days — tickers.json is point-in-time)
    """
    pullback = -3.0 <= _v(ticker.get("price_vs_50d_sma_pct")) <= -1.0
    breakout = _v(ticker.get("volume_vs_avg_ratio")) > 1.5 and _v(ticker.get("pct_from_52w_high")) >= 0
    momentum = _v(ticker.get("macd_crossover"), "") in ("bullish", "bullish_cross")

    triggers = {
        "pullback": pullback,
        "breakout": breakout,
        "momentum": momentum,
    }

    return {
        "ticker": ticker["ticker"],
        "entry_valid": any(triggers.values()),
        "triggers": triggers,
    }


def run(tickers: list[str], today: date = None) -> list[dict]:
    """
    Run the full decision engine for each ticker in the list.

    For every ticker:
      1. decide_ticker_action  — score + hard-rule overrides
      2. If action is Buy or Aggressive Buy / Add:
         - calculate_position_size  — regime-adjusted lot size
         - check_entry_timing       — confirm a valid entry trigger exists
    """
    if today is None:
        today = date.today()

    market = load_market_data()
    all_tickers = load_tickers_data()

    regime = classify_regime(calculate_regime_score(market)["regime_score"])
    regime_label = regime["regime_label"]

    BUY_ACTIONS = {"Buy", "Aggressive Buy / Add"}

    results = []
    for symbol in tickers:
        ticker = all_tickers.get(symbol)
        if ticker is None:
            results.append({"ticker": symbol, "error": "not found in tickers.json"})
            continue
        if "error" in ticker:
            results.append({"ticker": symbol, "error": ticker["error"]})
            continue

        decision = decide_ticker_action(ticker, today)
        row = {
            "ticker":        symbol,
            "final_action":  decision["final_action"],
            "total_score":   decision["total_score"],
            "breakdown":     decision["breakdown"],
        }

        if decision["final_action"] in BUY_ACTIONS:
            sizing = calculate_position_size(decision, regime_label)
            timing = check_entry_timing(ticker)
            row["position_size"]    = sizing["position_size"]
            row["regime_mult"]      = sizing["regime_multiplier"]
            row["score_mult"]       = sizing["score_multiplier"]
            row["entry_valid"]      = timing["entry_valid"]
            row["entry_triggers"]   = timing["triggers"]

        if decision["avoid_entry_reasons"]:
            row["avoid_entry_reasons"] = decision["avoid_entry_reasons"]
        if decision["force_exit_reasons"]:
            row["force_exit_reasons"] = decision["force_exit_reasons"]

        results.append(row)

    return results


if __name__ == "__main__":
    if not TICKERS:
        print("Usage: python decision-engine.py TICKER [TICKER ...]")
        sys.exit(1)

    market = load_market_data()
    regime_result = calculate_regime_score(market)
    regime = classify_regime(regime_result["regime_score"])
    print(f"Regime: {regime['regime_label']}  (score {regime_result['regime_score']}/6)\n")

    for row in run(TICKERS):
        if "error" in row:
            print(f"{row['ticker']}: {row['error']}")
            continue

        print(f"{row['ticker']:8}  {row['final_action']:22}  score={row['total_score']}")

        if "position_size" in row:
            print(f"          position size: {row['position_size']:.4f}x  "
                  f"(regime {row['regime_mult']}x × score {row['score_mult']}x)")
            valid = row["entry_valid"]
            triggers = ", ".join(k for k, v in row["entry_triggers"].items() if v) or "none"
            print(f"          entry timing:  {'VALID' if valid else 'NO TRIGGER'}  [{triggers}]")

        for reason in row.get("avoid_entry_reasons", []):
            print(f"          ⚠ avoid: {reason}")
        for reason in row.get("force_exit_reasons", []):
            print(f"          ✖ force exit: {reason}")
        print()


# inspo https://chatgpt.com/c/69de34ad-89a8-83e8-940a-143efed96920