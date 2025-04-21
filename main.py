from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import pandas as pd
import random

app = FastAPI()

class MARequest(BaseModel):
    symbol: str
    days: int

from fastapi.responses import FileResponse

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

@app.post("/ma")
def moving_average(request: MARequest):
    url = f"https://api.coingecko.com/api/v3/coins/{request.symbol.lower()}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": request.days,
        "interval": "daily"
    }

    response = httpx.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        return {"error": "Could not fetch price history."}

    # Convert the price data to a list of prices only
    prices = [price[1] for price in data["prices"]]
    
    # Use pandas to calculate the simple moving average
    df = pd.DataFrame(prices, columns=["price"])
    moving_avg = df["price"].mean()

    return {
        "symbol": request.symbol.upper(),
        "days": request.days,
        "moving_average_usd": round(moving_avg, 2)
    }
class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int

@app.post("/predict_price")
def predict_future_price(request: PredictionRequest):
    url = f"https://api.coingecko.com/api/v3/coins/{request.symbol.lower()}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 7,
        "interval": "daily"
    }

    response = httpx.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        return {"error": "Could not fetch price history."}

    # Get the list of prices
    prices = [price[1] for price in data["prices"]]

    # Calculate moving average
    df = pd.DataFrame(prices, columns=["price"])
    moving_avg = df["price"].mean()

    # Simulate a prediction by adding a small random % change
    percent_change = random.uniform(-0.05, 0.05)  # ±5% variation
    predicted_price = moving_avg * (1 + percent_change)

    return {
        "symbol": request.symbol.upper(),
        "days_ahead": request.days_ahead,
        "moving_average_usd": round(moving_avg, 2),
        "predicted_price_usd": round(predicted_price, 2),
        "note": "Prediction is randomly simulated and definitely not financial advice."
    }
from fastapi.responses import FileResponse

@app.get("/ui")
def get_ui():
    return FileResponse("static/index.html")
from sklearn.linear_model import LinearRegression
import numpy as np

@app.post("/predict_linear")
def predict_linear_price(request: PredictionRequest):
    url = f"https://api.coingecko.com/api/v3/coins/{request.symbol.lower()}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 30,
        "interval": "daily"
    }

    response = httpx.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        return {"error": "Could not fetch price history."}

    prices = [price[1] for price in data["prices"]]

    # Build training data
    X = np.arange(len(prices)).reshape(-1, 1)         # Days: 0, 1, 2, ..., n
    y = np.array(prices)                              # Prices

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future price
    future_day = len(prices) + request.days_ahead - 1
    predicted_price = model.predict([[future_day]])[0]

    return {
        "symbol": request.symbol.upper(),
        "days_ahead": request.days_ahead,
        "predicted_price_usd": round(predicted_price, 2),
        "note": "Based on Linear Regression using the last 30 days of prices"
    }