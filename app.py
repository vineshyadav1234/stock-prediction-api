
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("stock_model.pkl")

app = FastAPI()

class StockInput(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    SMA20: float
    EMA20: float
    RSI14: float
    MACD: float
    Signal: float

@app.get("/")
def home():
    return {"message": "Stock Prediction API Running"}

@app.post("/predict")
def predict(data: StockInput):

    df = pd.DataFrame([{
        "Open": data.Open,
        "High": data.High,
        "Low": data.Low,
        "Close": data.Close,
        "Volume": data.Volume,
        "SMA20": data.SMA20,
        "EMA20": data.EMA20,
        "RSI14": data.RSI14,
        "MACD": data.MACD,
        "Signal": data.Signal
    }])

    prediction = model.predict(df)[0]

    return {
        "predicted_price": float(prediction)
    }
