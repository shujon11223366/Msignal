from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

app = FastAPI()

# Allow frontend access from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend index.html
@app.get("/")
async def root():
    return FileResponse("index.html")


# TwelveData API keys (multiple)
API_KEYS = [
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

key_index = 0

def get_next_key():
    global key_index
    key = API_KEYS[key_index]
    key_index = (key_index + 1) % len(API_KEYS)
    return key


# Example: Fetch signal from TwelveData or logic
async def fetch_signal(pair: str, timeframe: str):
    key = get_next_key()
    url = f"https://api.twelvedata.com/time_series?symbol={pair.replace('/','')}&interval={timeframe}&apikey={key}&outputsize=10"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return {"error": resp.text}
        data = resp.json()

    # Dummy signal logic — replace with your real ICT/SMC strategy
    try:
        signal = "Strong Buy" if pair.endswith("USD") else "Strong Sell"
        return {
            "signal": signal,
            "confidence": 70,
            "confluence": ["EMA Bullish", "MACD Bullish", "Premium Zone"]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/signal")
async def get_signal(pair: str, timeframe: str):
    result = await fetch_signal(pair, timeframe)
    return result