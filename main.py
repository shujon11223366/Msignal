# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pandas_ta as ta
import httpx
import math
import time

# -------------------------
# === CONFIG / API KEYS ===
# -------------------------
# Your TwelveData API Key (baked in as requested)
TWELVE_KEY = "0a25bcb593e047b2aded75b1db91b130"

# Default timeframes (Investing.com-style)
TIMEFRAMES = ["1min", "5min", "15min", "1h", "4h", "5h"]

# Binance REST base (no key required for public klines)
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# FastAPI app
app = FastAPI(title="AI Signals Backend - Indicators + ICT/SMC")

# Allow all origins (change to your domain for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# === UTIL / HELPERS  ===
# -------------------------

def normalize_pair(pair: str) -> str:
    """Normalize pair string, e.g. 'EUR/USD' -> 'EUR/USD' and 'BTC/USDT' -> 'BTC/USDT'"""
    return pair.strip().upper().replace(" ", "")

def is_crypto_pair(pair: str) -> bool:
    """A simple heuristic: treat pairs whose quote is USDT/BTC/ETH/BUSD as crypto"""
    p = normalize_pair(pair)
    if "/" in p:
        base, quote = p.split("/")
    else:
        # fallback: last 3-4 chars
        base, quote = p[:-3], p[-3:]
    return quote in {"USDT","BTC","ETH","BUSD","USDC"}

def df_from_binance(pair: str, timeframe: str, lookback=200):
    """
    Fetch klines from Binance REST, return pandas DataFrame with columns: open, high, low, close, volume, t
    timeframe: '1min','5min','15min','1h','4h','5h' — we map to Binance intervals
    For 5h (not native), aggregate 1h candles into 5h.
    """
    p = normalize_pair(pair).replace("/", "")
    # Binance symbol expects e.g. BTCUSDT
    symbol = p
    # Map timeframe -> binance interval
    map_intervals = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "1h": "1h",
        "4h": "4h",
    }
    if timeframe in map_intervals:
        interval = map_intervals[timeframe]
        limit = min(1000, lookback)
        url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        arr = r.json()
        df = pd.DataFrame(arr, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        df = df[["open_time","open","high","low","close","volume"]].astype(float, errors='ignore')
        df.rename(columns={"open_time":"t"}, inplace=True)
        df["t"] = pd.to_datetime(df["t"], unit='ms')
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.reset_index(drop=True)
        return df[["t","open","high","low","close","volume"]]
    elif timeframe == "5h":
        # fetch 1h candles and aggregate by 5
        df_1h = df_from_binance(pair, "1h", lookback=math.ceil(lookback*1.2))
        # take last N*5 rows and group every 5
        n = 5
        df_1h = df_1h.tail( (df_1h.shape[0]//n) * n )
        if df_1h.empty:
            raise ValueError("Not enough data to build 5h candles")
        grouped = []
        for i in range(0, len(df_1h), n):
            chunk = df_1h.iloc[i:i+n]
            t = chunk["t"].iloc[0]
            open_ = chunk["open"].iloc[0]
            high_ = chunk["high"].max()
            low_ = chunk["low"].min()
            close_ = chunk["close"].iloc[-1]
            vol_ = chunk["volume"].sum()
            grouped.append([t, open_, high_, low_, close_, vol_])
        df = pd.DataFrame(grouped, columns=["t","open","high","low","close","volume"])
        return df
    else:
        raise ValueError("Unsupported timeframe for Binance: "+timeframe)

def df_from_twelvedata(pair: str, timeframe: str, outputsize=200):
    """
    Fetch candles from TwelveData time_series endpoint.
    timeframe uses TwelveData naming: 1min,5min,15min,1h,4h,5h
    """
    # Try symbol normalized without slash first (EURUSD) because TwelveData accepts many formats.
    symbol = normalize_pair(pair).replace("/", "")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={timeframe}&apikey={TWELVE_KEY}&outputsize={outputsize}"
    r = httpx.get(url, timeout=10)
    j = r.json()
    if "values" not in j:
        # try with slash (EUR/USD)
        symbol2 = normalize_pair(pair)
        url2 = f"https://api.twelvedata.com/time_series?symbol={symbol2}&interval={timeframe}&apikey={TWELVE_KEY}&outputsize={outputsize}"
        r2 = httpx.get(url2, timeout=10)
        j = r2.json()
        if "values" not in j:
            raise ValueError({"error": j})
    values = j["values"]
    df = pd.DataFrame(values)
    # ensure numeric
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # parse time
    if "datetime" in df.columns:
        df["t"] = pd.to_datetime(df["datetime"])
    else:
        df["t"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')
    df = df[["t","open","high","low","close","volume"]]
    df = df.reset_index(drop=True)
    return df

def fetch_candles(pair: str, timeframe: str):
    """
    Unified fetch: crypto -> Binance, else -> TwelveData
    returns DataFrame with columns t, open, high, low, close, volume
    """
    if is_crypto_pair(pair):
        return df_from_binance(pair, timeframe)
    else:
        return df_from_twelvedata(pair, timeframe)

# -------------------------
# === INDICATORS & SMC ===
# -------------------------

def compute_indicators(df: pd.DataFrame):
    """
    Compute RSI, EMA20, EMA50, MACD (constructed), and produce a simple signal + confidence + tags.
    """
    # Ensure numeric and drop NaNs
    df = df.dropna(subset=["close"]).copy()
    closes = df["close"].astype(float)

    # indicators
    df["rsi"] = ta.rsi(closes, length=14)
    df["ema20"] = ta.ema(closes, length=20)
    df["ema50"] = ta.ema(closes, length=50)
    # compute MACD manually (ema12 - ema26) and signal line
    ema12 = ta.ema(closes, length=12)
    ema26 = ta.ema(closes, length=26)
    macd_line = ema12 - ema26
    macd_signal = ta.ema(macd_line.fillna(0), length=9)

    # latest values
    rsi = float(df["rsi"].iloc[-1]) if not df["rsi"].isna().all() else None
    ema20 = float(df["ema20"].iloc[-1]) if not df["ema20"].isna().all() else None
    ema50 = float(df["ema50"].iloc[-1]) if not df["ema50"].isna().all() else None
    macd_l = float(macd_line.iloc[-1]) if len(macd_line)>0 else None
    macd_s = float(macd_signal.iloc[-1]) if len(macd_signal)>0 else None

    score = 0
    tags = []

    # EMA trend
    if ema20 is not None and ema50 is not None:
        if ema20 > ema50:
            score += 1
            tags.append("EMA Bullish")
        else:
            score -= 1
            tags.append("EMA Bearish")

    # RSI
    if rsi is not None:
        if rsi < 30:
            score += 1
            tags.append("RSI Oversold")
        elif rsi > 70:
            score -= 1
            tags.append("RSI Overbought")
        else:
            tags.append(f"RSI {rsi:.1f}")

    # MACD
    if macd_l is not None and macd_s is not None:
        if macd_l > macd_s:
            score += 1
            tags.append("MACD Bullish")
        else:
            score -= 1
            tags.append("MACD Bearish")

    # derive label
    if score >= 2:
        label = "Strong Buy"
    elif score == 1:
        label = "Buy"
    elif score == 0:
        label = "Neutral"
    elif score == -1:
        label = "Sell"
    else:
        label = "Strong Sell"

    confidence = min(100, (abs(score) / 3) * 100) if score is not None else 0

    return {
        "label": label,
        "confidence": round(confidence, 1),
        "tags": tags,
        "rsi": rsi,
        "ema20": ema20,
        "ema50": ema50,
        "macd": {"macd": macd_l, "signal": macd_s}
    }

def detect_smc(df: pd.DataFrame):
    """
    Lightweight ICT/SMC heuristics:
     - BOS (break of structure): last close breaks previous high/low
     - Simple order block detection: find recent opposite candle before impulsive move
     - Discount/Premium: price relative to EMA50
     - FVG: detect a price imbalance across 3 candles
    These are heuristic approximations and will need tuning for production use.
    """
    out = []
    if len(df) < 6:
        return out
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    t = df["t"]

    # BOS: compare last close with previous candle high/low
    last_close = close.iloc[-1]
    prev_high = high.iloc[-2]
    prev_low = low.iloc[-2]
    if last_close > prev_high:
        out.append("BOS ↑")
    elif last_close < prev_low:
        out.append("BOS ↓")

    # Discount/Premium vs ema50
    ema50 = ta.ema(close, length=50).iloc[-1]
    if pd.isna(ema50):
        pass
    else:
        if last_close < ema50:
            out.append("Discount Zone")
        else:
            out.append("Premium Zone")

    # Simple Order Block detection:
    # Find any bearish candle followed by at least two bullish candles indicating a bullish OB
    ob_found = False
    for i in range(len(df)-6, len(df)-2):
        if i < 0:
            continue
        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])
        # bearish candle before bullish run
        if c < o:
            # check next two candles bullish and make higher close
            if (df["close"].iloc[i+1] > df["close"].iloc[i+0]) and (df["close"].iloc[i+2] > df["close"].iloc[i+1]):
                ob_high = float(df["high"].iloc[i])
                ob_low = float(df["low"].iloc[i])
                # if price has returned to that zone (last_close within)
                if ob_low <= last_close <= ob_high:
                    out.append("Bullish OB Hit")
                    ob_found = True
                    break
    # opposite: bullish order block
    if not ob_found:
        for i in range(len(df)-6, len(df)-2):
            if i < 0:
                continue
            o = float(df["open"].iloc[i])
            c = float(df["close"].iloc[i])
            # bullish candle before bearish run
            if c > o:
                if (df["close"].iloc[i+1] < df["close"].iloc[i+0]) and (df["close"].iloc[i+2] < df["close"].iloc[i+1]):
                    ob_high = float(df["high"].iloc[i])
                    ob_low = float(df["low"].iloc[i])
                    if ob_low <= last_close <= ob_high:
                        out.append("Bearish OB Hit")
                        ob_found = True
                        break

    # Fair Value Gap (simple): check for a gap between consecutive candles bodies
    # If candle n low > candle n-1 high (up gap) or n high < n-1 low (down gap)
    for i in range(len(df)-4, len(df)-1):
        if i < 1:
            continue
        prev_high = float(df["high"].iloc[i-1])
        cur_low = float(df["low"].iloc[i])
        if cur_low > prev_high:
            out.append("FVG ↑")
            break
        prev_low = float(df["low"].iloc[i-1])
        cur_high = float(df["high"].iloc[i])
        if cur_high < prev_low:
            out.append("FVG ↓")
            break

    return out

# -------------------------
# === API ENDPOINTS  ===
# -------------------------

@app.get("/api/summary")
def api_summary(
    pairs: str = Query("EUR/USD", description="Comma separated pairs, e.g. EUR/USD,BTC/USDT"),
    timeframes: str = Query(",".join(TIMEFRAMES), description="Comma separated timeframes (optional)"),
):
    """
    Returns a Investing.com-style technical summary for multiple pairs across multiple timeframes.
    Example: /api/summary?pairs=EUR/USD,BTC/USDT&timeframes=1min,5min,15min,1h,4h,5h
    """
    # parse
    pair_list = [normalize_pair(p) for p in pairs.split(",") if p.strip()]
    tf_list = [t.strip() for t in timeframes.split(",") if t.strip()]
    if not tf_list:
        tf_list = TIMEFRAMES

    results = {}
    for pair in pair_list:
        pair_res = {}
        for tf in tf_list:
            try:
                # throttle a little to avoid rate limits if many requests
                # time.sleep(0.15)
                df = fetch_candles(pair, tf)
                if df is None or df.empty:
                    pair_res[tf] = {"error": "no data"}
                    continue
                indicators = compute_indicators(df)
                smc = detect_smc(df)
                pair_res[tf] = {
                    "signal": indicators["label"],
                    "confidence": indicators["confidence"],
                    "indicators": indicators["tags"],
                    "smc": smc
                }
            except Exception as e:
                pair_res[tf] = {"error": str(e)}
        results[pair] = pair_res

    return {"results": results}
