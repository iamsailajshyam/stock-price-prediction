import os
import random
from datetime import timedelta

# Import our custom auth logic
import auth
import joblib  # type: ignore
import numpy as np  # type: ignore
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="NeuralStock API")

# Allow the frontend to connect without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---
class UserPayload(BaseModel):
    email: str
    password: str


# --- Authentication Routes ---
@app.post("/auth/register")
async def register(user: UserPayload):
    return auth.register_user(user.email, user.password)


@app.post("/auth/login")
async def login(user: UserPayload):
    return auth.login_user(user.email, user.password)


# --- Prediction & Data Routes ---
@app.get("/api/predict")
async def predict(ticker: str = "AAPL", days: int = 7):
    # Phase 4 Completion: We load the .h5 LSTM model matching the Colab Notebook architecture
    # and utilize it against the latest 60 sequences of live yfinance data.

    try:
        stock = yf.Ticker(ticker)
        # Pull 100 days to ensure we have at least 60 valid trading days
        hist = stock.history(period="100d").dropna()

        if hist.empty or len(hist) < 60:
            raise HTTPException(
                status_code=404, detail="Not enough historical data found for ticker"
            )

        # Get purely the recent 60 trading days for the LSTM input sequence
        recent_60_days = hist["Close"].values[-60:]

        # Safely extract floats (prevents JSON NaN crashes and IDE round type-warnings)
        import math

        def safe_float(val):
            val = float(val)
            return 0.0 if math.isnan(val) or math.isinf(val) else float(round(val, 2))

        hist_prices = [safe_float(p) for p in hist["Close"].tolist()]
        labels = [d.strftime("%b %d") for d in hist.index]
        current_price = hist_prices[-1]

        # Build predictions array (mostly nulls except the future days)
        from typing import List, Optional

        pred_data: List[Optional[float]] = [None for _ in range(len(hist_prices) - 1)]
        pred_data.append(current_price)

        # Attempt to load Trained Keras Model and Scaler
        model_path = os.path.join("models", f"{ticker.lower()}_lstm.h5")
        scaler_path = os.path.join("models", f"{ticker.lower()}_scaler.pkl")

        ml_active = False
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                from tensorflow.keras.models import load_model

                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                ml_active = True

                # Setup recursive multi-day prediction
                current_batch = np.asarray(recent_60_days).reshape(-1, 1)
                scaled_batch = scaler.transform(current_batch)

                # Predict 'days' into the future
                input_seq = scaled_batch.reshape(1, 60, 1)

                for i in range(1, days + 1):
                    future_date = hist.index[-1] + timedelta(days=i)
                    labels.append(future_date.strftime("%b %d"))

                    next_scaled_price = model.predict(input_seq, verbose=0)
                    next_price = scaler.inverse_transform(next_scaled_price)[0][0]
                    pred_data.append(safe_float(next_price))

                    # Update sequence: remove oldest, append newest
                    input_seq = np.append(
                        input_seq[:, 1:, :], [next_scaled_price], axis=1
                    )

            except Exception as ml_err:
                print(
                    f"[LSTM Load Warning] {ml_err} - Falling back to math simulation."
                )
                ml_active = False

        if not ml_active:
            # Fallback mathematical simulation if .h5 is missing or TF is not installed
            volatility = 0.02
            for i in range(1, days + 1):
                future_date = hist.index[-1] + timedelta(days=i)
                labels.append(future_date.strftime("%b %d"))
                simulated_val = current_price * (
                    1 + (random.random() - 0.45) * volatility
                )
                pred_data.append(safe_float(simulated_val))

        return {
            "ticker": ticker,
            "labels": labels,
            "historical": hist_prices,
            "predicted": pred_data,
            "current": safe_float(current_price),
            "model_engine": "TensorFlow LSTM" if ml_active else "Stochastic Simulation",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment")
async def get_sentiment(ticker: str):
    # Simulated NLP sentiment analysis
    sentiment_score = round(float(random.uniform(-4, 4)), 1)
    return {
        "ticker": ticker,
        "score": sentiment_score,
        "status": "Bullish" if sentiment_score > 0 else "Bearish",
    }


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
