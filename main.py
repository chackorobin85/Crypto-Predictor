from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from sklearn.linear_model import LinearRegression
import numpy as np
import httpx
from datetime import datetime
CMC_API_KEY = "49b3f548-348d-4c31-8faf-249b0084718a"

app = FastAPI()

class PredictionRequest(BaseModel):
    symbol: str
    target_date: str  # YYYY-MM-DD

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

@app.post("/predict_linear")
def predict_linear_price(request: PredictionRequest):
    try:
        target = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        days_ahead = (target - today).days

        if days_ahead < 1 or days_ahead > 365:
            return {"error": "Target date must be between 1 and 365 days from today."}

        url = f"https://api.coingecko.com/api/v3/coins/{request.symbol.lower()}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": 365
        }

        response = httpx.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"CoinGecko error: status code {response.status_code}"}

        data = response.json()
        prices = data.get("prices", [])

        if not prices:
            return {"error": "No price data returned from CoinGecko."}

        prices = [price[1] for price in prices][-365:]
        if len(prices) < 30:
            return {"error": "Insufficient price data for prediction."}

        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        model = LinearRegression()
        model.fit(X, y)

        future_index = len(prices) + days_ahead - 1
        predicted_price = model.predict([[future_index]])[0]

        return {
            "symbol": request.symbol.upper(),
            "target_date": request.target_date,
            "predicted_price_usd": round(predicted_price, 2),
            "note": f"Predicted using linear regression on past {len(prices)} days"
        }

    except Exception as e:
        print("🔥 Exception:", str(e))
        return {"error": "Internal server error. Please try again."}
@app.get("/coin_stats/{symbol}")
def coin_stats(symbol: str):
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
        params = {"symbol": symbol.upper(), "convert": "USD"}

        response = httpx.get(url, headers=headers, params=params)
        data = response.json()

        if "data" not in data or symbol.upper() not in data["data"]:
            return {"error": "Invalid symbol or data not found."}

        info = data["data"][symbol.upper()]
        quote = info.get("quote", {}).get("USD", {})

        return {
            "symbol": symbol.upper(),
            "current_price_usd": quote.get("price"),
            "high_24h": quote.get("high_24h"),
            "low_24h": quote.get("low_24h"),
            "price_change_percentage_1h": quote.get("percent_change_1h"),
            "price_change_percentage_24h": quote.get("percent_change_24h"),
            "price_change_percentage_7d": quote.get("percent_change_7d"),
            "market_cap": quote.get("market_cap"),
            "volume_24h": quote.get("volume_24h")
        }

    except Exception as e:
        print("🔥 CMC stats exception:", str(e))
        return {"error": "Failed to fetch from CoinMarketCap"}