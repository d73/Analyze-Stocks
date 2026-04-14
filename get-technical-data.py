"""
get-technical-data.py
Fetches per-ticker technical indicators and market-wide sentiment data,
writing two separate JSON files intended for downstream AI analysis.

Output files:
  output-tickers.json     — one record per ticker (fetch_ticker_data)
  output-indicators.json  — single record of market-wide conditions (fetch_market_indicators)

--- TICKER DATA (output-tickers.json) ---
Per-ticker fields pulled via yfinance:
  Price & range:
    - Current price, 52W low/high, % from each, range position (0-100)
  Momentum:
    - RSI (14)
    - MACD (12/26/9) with histogram and crossover signal
  Moving averages:
    - Price vs 50D SMA (% above/below + direction)
    - Price vs 200D SMA (% above/below + direction)
  Volume & volatility:
    - Latest volume, 20D avg volume, volume/avg ratio
    - ATR (14)
    - Beta (24M)
  Short interest:
    - Short % of float, days to cover
  Fundamentals:
    - Earnings date

--- MARKET-WIDE DATA (output-indicators.json) ---
Fetched once, not tied to any individual ticker:
  - S&P 500 price, 200D SMA, % above/below, SMA trend (^GSPC)
  - VIX level + 1Y percentile rank (^VIX)
  - VVIX level + 1Y percentile rank (^VVIX)
  - HYG price, 1Y percentile, % vs 20D SMA, trend
  - CNN Fear & Greed Index: composite score, rating, 1W/1M history,
    and all seven sub-indicators (momentum, price strength, price breadth,
    put/call, VIX, junk bond demand, safe haven demand)

Usage:
    pip install yfinance pandas numpy scipy fear-greed
    python get-technical-data.py AAPL MSFT NVDA
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import fear_greed as fg
from datetime import date
from scipy.stats import percentileofscore

warnings.filterwarnings("ignore")
TICKERS = sorted(set(t.replace(".", "-") for t in sys.argv[1:]))

# ---------------------------------------------------------------------------
# Price-history derived indicators
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(close: pd.Series) -> dict:
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    # Crossover: did MACD cross signal line in last 3 bars?
    h = histogram.iloc[-3:]
    if h.iloc[-1] > 0 and h.iloc[0] <= 0:
        crossover = "bullish_cross"
    elif h.iloc[-1] < 0 and h.iloc[0] >= 0:
        crossover = "bearish_cross"
    else:
        crossover = "bullish" if histogram.iloc[-1] > 0 else "bearish"

    return {
        "macd": round(float(macd.iloc[-1]), 4),
        "macd_signal_line": round(float(signal.iloc[-1]), 4),
        "macd_histogram": round(float(histogram.iloc[-1]), 4),
        "macd_crossover": crossover,
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return round(float(atr.iloc[-1]), 4)


def compute_sma_metrics(close: pd.Series, window: int) -> dict:
    if len(close) < window:
        return {
            f"price_vs_{window}d_sma_pct": None,
            f"sma_{window}d_direction": None,
        }
    sma = close.rolling(window=window).mean()
    latest_sma = float(sma.iloc[-1])
    latest_price = float(close.iloc[-1])
    pct_above = round((latest_price / latest_sma - 1) * 100, 2)

    # Direction: is SMA itself rising vs 5 bars ago?
    lookback = min(6, len(sma.dropna()))
    direction = "up" if latest_sma > float(sma.iloc[-lookback]) else "down"

    return {
        f"price_vs_{window}d_sma_pct": pct_above,
        f"sma_{window}d_direction": direction,
    }


def compute_52w_range(close: pd.Series) -> dict:
    w52 = close.tail(252)
    low = float(w52.min())
    high = float(w52.max())
    price = float(close.iloc[-1])
    pct_from_low = round((price / low - 1) * 100, 2)
    pct_from_high = round((price / high - 1) * 100, 2)
    range_position = round((price - low) / (high - low) * 100, 2) if high != low else None
    return {
        "52w_low": round(low, 2),
        "52w_high": round(high, 2),
        "pct_from_52w_low": pct_from_low,   # positive = above low
        "pct_from_52w_high": pct_from_high,  # negative = below high
        "52w_range_position_pct": range_position,  # 0=at low, 100=at high
    }


def compute_volume_ratio(df: pd.DataFrame, avg_window: int = 20) -> dict:
    vol = df["Volume"].squeeze()
    avg_vol = float(vol.tail(avg_window).mean())
    latest_vol = float(vol.iloc[-1])
    ratio = round(latest_vol / avg_vol, 2) if avg_vol else None
    return {
        "volume": int(latest_vol),
        "avg_volume_20d": int(avg_vol),
        "volume_vs_avg_ratio": ratio,
    }


def compute_rs_vs_spy(df_ticker: pd.DataFrame, df_spy: pd.DataFrame, window: int = 20) -> dict:
    t = df_ticker["Close"].squeeze()
    s = df_spy["Close"].squeeze()
    combined = pd.DataFrame({"ticker": t, "spy": s}).dropna()
    if len(combined) < window + 1:
        return {"rs_vs_spy_20d": None, "rs_vs_spy_signal": None}
    t_ret = combined["ticker"].iloc[-1] / combined["ticker"].iloc[-window] - 1
    s_ret = combined["spy"].iloc[-1] / combined["spy"].iloc[-window] - 1
    rs = round((1 + t_ret) / (1 + s_ret), 4) if (1 + s_ret) != 0 else None
    return {"rs_vs_spy_20d": rs}


# ---------------------------------------------------------------------------
# yfinance .info derived
# ---------------------------------------------------------------------------

def get_info_fields(info: dict) -> dict:
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    short_pct = info.get("shortPercentOfFloat")
    return {
        "price": round(float(price), 2) if price else None,
        "beta_24m": info.get("beta"),
        "short_interest_pct_float": round(short_pct * 100, 2) if short_pct else None,
        "days_to_cover": info.get("shortRatio"),
    }


# ---------------------------------------------------------------------------
# Earnings date
# ---------------------------------------------------------------------------

def get_earnings_date(t: yf.Ticker) -> str | None:
    try:
        cal = t.calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed and len(ed) > 0:
                v = ed[0]
                return str(v.date()) if hasattr(v, "date") else str(v)
        if hasattr(cal, "loc"):
            ed = cal.loc["Earnings Date"]
            v = ed.iloc[0]
            return str(v.date()) if hasattr(v, "date") else str(v)
    except Exception:
        pass
    return None



# ---------------------------------------------------------------------------
# Market-wide indicators (fetched once, applied to all records)
# ---------------------------------------------------------------------------

def get_sp500_vs_200ma(period: str = "2y") -> dict:
    """
    Download ^GSPC and compute S&P 500 position relative to its 200D SMA.

    Returns:
      sp500_price         – latest close
      sp500_200d_sma      – current 200D SMA value
      sp500_vs_200d_pct   – % above/below the 200D SMA (positive = above)
      sp500_200d_trend    – 'up' / 'down' based on SMA direction vs 5 bars ago
    """
    df = yf.download("^GSPC", period=period, interval="1d", progress=False)
    if df.empty:
        return {"sp500_price": None, "sp500_200d_sma": None,
                "sp500_vs_200d_pct": None, "sp500_200d_trend": None}

    close = df["Close"].squeeze().dropna()
    sma200 = close.rolling(window=200).mean().dropna()

    if sma200.empty:
        return {"sp500_price": None, "sp500_200d_sma": None,
                "sp500_vs_200d_pct": None, "sp500_200d_trend": None}

    price = round(float(close.iloc[-1]), 2)
    sma_val = round(float(sma200.iloc[-1]), 2)
    vs_pct = round((price / sma_val - 1) * 100, 2)

    lookback = min(5, len(sma200) - 1)
    trend = "up" if float(sma200.iloc[-1]) > float(sma200.iloc[-1 - lookback]) else "down"

    return {
        "sp500_price": price,
        "sp500_200d_sma": sma_val,
        "sp500_vs_200d_pct": vs_pct,
        "sp500_200d_trend": trend,
    }


def get_vix(period: str = "1y") -> dict:
    """Download ^VIX and return current level + 1Y percentile rank."""
    df = yf.download("^VIX", period=period, interval="1d", progress=False)
    if df.empty:
        return {"vix": None, "vix_percentile": None}
    close = df["Close"].squeeze().dropna()
    current = round(float(close.iloc[-1]), 2)
    percentile = round(float(percentileofscore(close, current)), 1)
    return {"vix": current, "vix_percentile": percentile}


def get_vvix(period: str = "1y") -> dict:
    """Download ^VVIX and return current level + 1Y percentile rank."""
    df = yf.download("^VVIX", period=period, interval="1d", progress=False)
    if df.empty:
        return {"vvix": None, "vvix_percentile": None}
    close = df["Close"].squeeze().dropna()
    current = round(float(close.iloc[-1]), 2)
    percentile = round(float(percentileofscore(close, current)), 1)
    return {"vvix": current, "vvix_percentile": percentile}




def get_hyg(period: str = "1y") -> dict:
    """
    Download HYG (iShares HY Corporate Bond ETF) as a credit / risk-sentiment gauge.

    Returns:
      hyg_price         – latest closing price
      hyg_1y_percentile – where today's price sits in the trailing 1Y range (0=low, 100=high)
      hyg_vs_20d_sma    – % above/below 20D SMA (positive = above)
      hyg_trend         – 'up' / 'down' based on 20D SMA direction vs 5 bars ago
    """
    df = yf.download("HYG", period=period, interval="1d", progress=False)
    if df.empty:
        return {"hyg_price": None, "hyg_1y_percentile": None,
                "hyg_vs_20d_sma": None, "hyg_trend": None}

    close = df["Close"].squeeze().dropna()
    price = round(float(close.iloc[-1]), 2)
    percentile = round(float(percentileofscore(close, price)), 1)

    sma20 = close.rolling(window=20).mean()
    latest_sma = float(sma20.iloc[-1])
    vs_sma = round((price / latest_sma - 1) * 100, 2) if latest_sma else None

    lookback = min(5, len(sma20.dropna()) - 1)
    trend = "up" if latest_sma > float(sma20.iloc[-1 - lookback]) else "down"

    return {
        "hyg_price": price,
        "hyg_1y_percentile": percentile,
        "hyg_vs_20d_sma": vs_sma,
        "hyg_trend": trend,
    }


def get_fear_greed() -> dict:
    """
    Fetch CNN Fear & Greed Index via the fear_greed package.

    Returns:
      fear_greed_score   – current score (0 = extreme fear, 100 = extreme greed)
      fear_greed_rating  – text label, e.g. 'Extreme Fear', 'Greed'
    """
    try:
        data = fg.get()
        history = data.get("history", {})
        ind = data.get("indicators", {})

        def _ind(key):
            return ind.get(key, {})

        return {
            "fear_greed_score":                         round(data["score"], 1),
            "fear_greed_rating":                        data["rating"],
            "fear_greed_1w_ago":                        history.get("1w"),
            "fear_greed_1m_ago":                        history.get("1m"),
            # Sub-indicators
            "fear_greed_momentum_sp500_score":          _ind("market_momentum_sp500").get("score"),
            "fear_greed_momentum_sp500_rating":         _ind("market_momentum_sp500").get("rating"),
            "fear_greed_price_strength_score":          _ind("stock_price_strength").get("score"),
            "fear_greed_price_strength_rating":         _ind("stock_price_strength").get("rating"),
            "fear_greed_price_breadth_score":           _ind("stock_price_breadth").get("score"),
            "fear_greed_price_breadth_rating":          _ind("stock_price_breadth").get("rating"),
            "fear_greed_put_call_score":                _ind("put_call_options").get("score"),
            "fear_greed_put_call_rating":               _ind("put_call_options").get("rating"),
            "fear_greed_market_volatility_vix_score":   _ind("market_volatility_vix").get("score"),
            "fear_greed_market_volatility_vix_rating":  _ind("market_volatility_vix").get("rating"),
            "fear_greed_junk_bond_demand_score":        _ind("junk_bond_demand").get("score"),
            "fear_greed_junk_bond_demand_rating":       _ind("junk_bond_demand").get("rating"),
            "fear_greed_safe_haven_demand_score":       _ind("safe_haven_demand").get("score"),
            "fear_greed_safe_haven_demand_rating":      _ind("safe_haven_demand").get("rating"),
        }
    except Exception as e:
        print(f"  fear_greed ERROR: {e!r}")
        return {
            "fear_greed_score": None,                        "fear_greed_rating": None,
            "fear_greed_1w_ago": None,                       "fear_greed_1m_ago": None,
            "fear_greed_momentum_sp500_score": None,         "fear_greed_momentum_sp500_rating": None,
            "fear_greed_price_strength_score": None,         "fear_greed_price_strength_rating": None,
            "fear_greed_price_breadth_score": None,          "fear_greed_price_breadth_rating": None,
            "fear_greed_put_call_score": None,               "fear_greed_put_call_rating": None,
            "fear_greed_market_volatility_vix_score": None,  "fear_greed_market_volatility_vix_rating": None,
            "fear_greed_junk_bond_demand_score": None,       "fear_greed_junk_bond_demand_rating": None,
            "fear_greed_safe_haven_demand_score": None,      "fear_greed_safe_haven_demand_rating": None,
        }


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_ticker_data(tickers: list[str]) -> list[dict]:
    print("Downloading SPY (relative strength baseline)...")
    spy_data = yf.download("SPY", period="1y", interval="1d", progress=False)

    results = []
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            t = yf.Ticker(ticker)
            df = yf.download(ticker, period="1y", interval="1d", progress=False)

            if df.empty:
                print("NO DATA")
                results.append({"ticker": ticker, "error": "no_price_data"})
                continue

            close = df["Close"].squeeze()
            info = t.info

            record = {"ticker": ticker}

            # Price / info fields
            record.update(get_info_fields(info))

            # 52W range
            record.update(compute_52w_range(close))

            # Volume vs avg
            record.update(compute_volume_ratio(df))

            # RSI
            record["rsi_14"] = compute_rsi(close)

            # MACD
            record.update(compute_macd(close))

            # ATR
            record["atr_14"] = compute_atr(df)

            # SMA metrics
            record.update(compute_sma_metrics(close, 50))
            record.update(compute_sma_metrics(close, 200))

            # Relative strength vs SPY
            record.update(compute_rs_vs_spy(df, spy_data))

            # Earnings date
            record["earnings_date"] = get_earnings_date(t)

            print("OK")
            results.append(record)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"ticker": ticker, "error": str(e)})

    return results

def fetch_market_indicators() -> list[dict]:
    print("Downloading S&P 500 vs 200D MA...")
    sp500_data = get_sp500_vs_200ma()

    print("Downloading VIX...")
    vix_data = get_vix()

    print("Downloading VVIX...")
    vvix_data = get_vvix()

    print("Downloading HYG (credit / risk sentiment)...")
    hyg_data = get_hyg()

    print("Fetching Fear & Greed Index...")
    fear_greed_data = get_fear_greed()

    results = []

    try:
        record = {}

        # Market-wide indicators
        record.update(sp500_data)
        record.update(vix_data)
        record.update(vvix_data)
        record.update(hyg_data)
        record.update(fear_greed_data)
        
        print("OK")
        results.append(record)

    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nFetching technicals for: {', '.join(TICKERS)}\n")
    data = fetch_ticker_data(TICKERS)
    f_tickers = "output-tickers.json"
    with open(f_tickers, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved to {f_tickers}")


    print(f"\nFetching market indicators\n")
    data = fetch_market_indicators()
    f_market = "output-market.json"
    with open(f_market, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nSaved to {f_market}")
