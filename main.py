from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pandas_ta as ta
import httpx
import itertools

app = FastAPI()

# ======================
# === CONFIGURATION ===
# ======================
TWELVEDATA_KEYS = [
    "0a25bcb593e047b2aded75b1db91b130",
    "b032b8736d00401ab3321a4f7bdcb41b",
    "a53423fb25cd4cddb1a4cef56556cb28",
    "fec80023e7814ff09b28ed7ae3ec60f6",
    "7b9e17a6949148fb9a39d20e55175857",
    "c136460dce8d4f30a509ebfeddd86314",
    "1a796c8dbefc41dc8e5ccd5a1b4a4076",
    "7531386b92d448a4aa786b9d8bd81b41",
    "ad1735526af74714af6ff355495b54f5",
    "29251dc6d59f4cf6b8318347eed7071d"
]

key_cycle = itertools.cycle(TWELVEDATA_KEYS)


def get_twelvedata_key():
    return next(key_cycle)


# ======================
# === CORS CONFIG ===
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "AI Signal Backend Running 🚀"}


@app.get("/api/signal")
def get_signal(pair: str = "EUR/USD", timeframe: str = "5min"):
    try:
        df = fetch_symbol(pair, timeframe)
        signal_data = calculate_indicators(df)
        smc_data = detect_smc(df)

        return {
            "pair": pair,
            "timeframe": timeframe,
            "signal": signal_data["signal"],
            "confidence": signal_data["confidence"],
            "confluence": signal_data["tags"] + smc_data,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/summary")
def get_summary(pairs: str = "EUR/USD,BTC/USD,ETH/USD", timeframes: str = "1min,5min,15min"):
    result = {}
    pair_list = pairs.split(",")
    timeframe_list = timeframes.split(",")

    for pair in pair_list:
        result[pair] = {}
        for timeframe in timeframe_list:
            try:
                df = fetch_symbol(pair, timeframe)
                signal_data = calculate_indicators(df)
                smc_data = detect_smc(df)

                result[pair][timeframe] = {
                    "signal": signal_data["signal"],
                    "confidence": signal_data["confidence"],
                    "confluence": signal_data["tags"] + smc_data
                }

            except Exception as e:
                result[pair][timeframe] = {"error": str(e)}

    return {"results": result}


def fetch_symbol(symbol: str, interval: str):
    api_key = get_twelvedata_key()
    symbol = symbol.replace(" ", "")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=5000"
    r = httpx.get(url).json()

    if "values" not in r:
        raise ValueError(r)

    df = pd.DataFrame(r["values"])
    df.rename(columns={"open": "o", "high": "high", "low": "low", "close": "close"}, inplace=True)

    # ✅ Convert OHLC columns to floats
    for col in ["o", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df


def calculate_indicators(df):
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]

    rsi = df["rsi"].iloc[-1]
    ema20, ema50 = df["ema20"].iloc[-1], df["ema50"].iloc[-1]

    score = 0
    tags = []

    if ema20 > ema50:
        score += 1
        tags.append("EMA Bullish")
    else:
        score -= 1
        tags.append("EMA Bearish")

    if rsi < 30:
        score += 1
        tags.append("RSI Oversold")
    elif rsi > 70:
        score -= 1
        tags.append("RSI Overbought")

    if df["macd"].iloc[-1] > 0:
        score += 1
        tags.append("MACD Bullish")
    else:
        score -= 1
        tags.append("MACD Bearish")

    if score >= 2:
        signal = "Strong Buy"
    elif score == 1:
        signal = "Buy"
    elif score == 0:
        signal = "Neutral"
    elif score == -1:
        signal = "Sell"
    else:
        signal = "Strong Sell"

    return {"signal": signal, "confidence": abs(score) / 3 * 100, "tags": tags}


def detect_smc(df):
    confluence = []
    if len(df) < 3:
        return confluence

    if df["close"].iloc[-1] > df["high"].iloc[-2]:
        confluence.append("BOS ↑")
    elif df["close"].iloc[-1] < df["low"].iloc[-2]:
        confluence.append("BOS ↓")

    if df["close"].iloc[-1] < df["ema50"].iloc[-1]:
        confluence.append("Discount Zone")
    else:
        confluence.append("Premium Zone")

    return confluence