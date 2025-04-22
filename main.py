from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from sklearn.linear_model import LinearRegression
import numpy as np
import httpx
import pandas as pd
import random

app = FastAPI()

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

@app.get("/ui")
def get_ui():
    return FileResponse("static/index.html")

@app.post("/predict_linear")
def predict_linear_price(request: PredictionRequest):
    symbol = request.symbol.lower()
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 7
    }

    try:
        response = httpx.get(url, params=params)
        print("📡 URL called:", url)
        print("📦 Status:", response.status_code)

        if response.status_code != 200:
            return {"error": f"CoinGecko error: status code {response.status_code}"}

        data = response.json()
        print("📈 CoinGecko response:", data)

        if "prices" not in data or not data["prices"]:
            return {"error": "CoinGecko returned no price data. Check the coin ID or try again in a minute."}

        prices = [price[1] for price in data["prices"]]
        prices = prices[-30:]

        if len(prices) < 5:
            return {"error": "Not enough price data returned to generate prediction."}

        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        model = LinearRegression()
        model.fit(X, y)

        future_day = len(prices) + request.days_ahead - 1
        predicted_price = model.predict([[future_day]])[0]

        return {
            "symbol": request.symbol.upper(),
            "days_ahead": request.days_ahead,
            "predicted_price_usd": round(predicted_price, 2),
            "note": "Prediction based on linear regression using recent prices from CoinGecko."
        }

    except Exception as e:
        print("🔥 EXCEPTION:", str(e))
        return {"error": "Server error while fetching or processing data."}