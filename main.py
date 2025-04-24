from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from sklearn.linear_model import LinearRegression
import numpy as np
import httpx
from datetime import datetime

app = FastAPI()

CRYPTOCOMPARE_API_KEY = "2c8ffa2fb7e46dd438e4c06007ac9e6ee726b4584c40780c7f8998abdb1d2f98"

class PredictionRequest(BaseModel):
    symbol: str
    target_date: str  # YYYY-MM-DD

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

@app.post("/predict_linear")
def predict_linear_price(request: PredictionRequest):
    try:
        symbol = request.symbol.upper()
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        today = datetime.today().date()
        days_ahead = (target_date - today).days

        if days_ahead < 1 or days_ahead > 30:
            return {"error": "Only 1–30 day predictions are supported."}

        # Fetch actual historical data
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 30
        }
        headers = {
            "authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"
        }

        response = httpx.get(url, params=params, headers=headers)
        data = response.json()

        history = data.get("Data", {}).get("Data", [])
        closes = [d["close"] for d in history if d.get("close")]

        if len(closes) < 10:
            return {"error": "Insufficient historical data for prediction."}

        X = np.arange(len(closes)).reshape(-1, 1)
        y = np.array(closes)
        model = LinearRegression()
        model.fit(X, y)

        future_index = len(closes) + days_ahead - 1
        predicted_price = model.predict([[future_index]])[0]

        return {
            "symbol": symbol,
            "target_date": request.target_date,
            "predicted_price_usd": round(predicted_price, 2),
            "note": f"Prediction based on {len(closes)} real days of historical data from CryptoCompare."
        }

    except Exception as e:
        print("🔥 Predict exception:", str(e))
        return {"error": "Server error during prediction."}


@app.get("/coin_stats/{symbol}")
def coin_stats(symbol: str):
    try:
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {
            "fsyms": symbol.upper(),
            "tsyms": "USD"
        }
        headers = {
            "authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"
        }

        response = httpx.get(url, params=params, headers=headers)
        data = response.json()
        raw = data.get("RAW", {}).get(symbol.upper(), {}).get("USD", {})

        if not raw:
            return {"error": "Invalid symbol or no data found."}

        return {
            "symbol": symbol.upper(),
            "current_price_usd": raw.get("PRICE"),
            "high_24h": raw.get("HIGH24HOUR"),
            "low_24h": raw.get("LOW24HOUR"),
            "price_change_percentage_1h": raw.get("CHANGEHOUR"),
            "price_change_percentage_24h": raw.get("CHANGEPCT24HOUR"),
            "price_change_percentage_7d": raw.get("CHANGEPCTDAY"),
            "market_cap": raw.get("MKTCAP"),
            "volume_24h": raw.get("TOTALVOLUME24H")
        }

    except Exception as e:
        print("🔥 CryptoCompare stats exception:", str(e))
        return {"error": "Failed to fetch stats from CryptoCompare."}


@app.get("/historical_prices/{symbol}/{days}")
def get_price_history(symbol: str, days: int):
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": days
        }
        headers = {
            "authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"
        }

        response = httpx.get(url, params=params, headers=headers)
        data = response.json()
        history = data.get("Data", {}).get("Data", [])

        if not history:
            return {"error": "Could not load price history."}

        formatted = [
            [d["time"] * 1000, d["close"]]
            for d in history if d.get("close") is not None
        ]

        return {
            "symbol": symbol.upper(),
            "days": days,
            "prices": formatted
        }

    except Exception as e:
        print("🔥 History exception:", str(e))
        return {"error": "Failed to load historical prices."}