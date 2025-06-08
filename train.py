import pandas as pd
from prophet import Prophet
import pickle
import os

CSV_FILE = "data.csv"
MODEL_FILE = "prophet_model.pkl"

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Close"]).sort_values("Timestamp")

    # Prepare data for Prophet
    df_prophet = df.rename(columns={"Timestamp": "ds", "Close": "y"})[["ds", "y"]]

    # Downsample hourly for faster training
    df_hourly = df_prophet.set_index('ds').resample('H').mean().dropna().reset_index()

    print(f"Training model on {len(df_hourly)} hourly data points...")
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_hourly)

    print(f"Saving trained model to {MODEL_FILE}...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Done!")

if __name__ == "__main__":
    main()
