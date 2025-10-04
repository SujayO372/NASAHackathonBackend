# server.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CONFIG ---
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
NASA_FILENAME = "NasaWeatherPred.csv"
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

HOURLY_VARIABLES = [
    "time",
    "temperature_2m",
    "pressure_msl",
    "cloudcover",
    "visibility",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
]

# --- GLOBALS ---
NASA_MODEL = None
NASA_LABEL_ENCODER = None

# --- HELPERS ---
def train_nasa_model():
    """
    Trains the NASA weather prediction model using the CSV columns:
    Temperature (°C), Relative Humidity (%), Wind Speed (km/h), Precipitation (mm/h)
    Target: Weather Type
    """
    global NASA_MODEL, NASA_LABEL_ENCODER

    if not os.path.exists(NASA_FILENAME):
        print(f"⚠️ {NASA_FILENAME} not found.")
        return False

    df = pd.read_csv(NASA_FILENAME)
    df.columns = df.columns.str.strip()  # remove leading/trailing spaces

    feature_cols = ["Temperature (°C)", "Relative Humidity (%)", "Wind Speed (km/h)", "Precipitation (mm/h)"]
    target_col = "Weather Type"

    # Check all columns exist
    if not all(col in df.columns for col in feature_cols + [target_col]):
        print("⚠️ CSV missing one or more required columns.")
        return False

    X = df[feature_cols]
    y = df[target_col]

    NASA_LABEL_ENCODER = LabelEncoder()
    y_enc = NASA_LABEL_ENCODER.fit_transform(y)

    NASA_MODEL = RandomForestClassifier(n_estimators=200, random_state=42)
    NASA_MODEL.fit(X, y_enc)

    print("✅ NASA model trained successfully.")
    return True

def get_lat_lon(city: str, state: str):
    query = f"{city}"
    resp = requests.get(GEOCODING_API_URL, params={"name": query, "count": 1})
    data = resp.json()
    if "results" not in data or not data["results"]:
        raise HTTPException(400, "Could not geocode location")
    res = data["results"][0]
    return res["latitude"], res["longitude"], res["name"]

def fetch_archive(lat, lon, start_date, end_date):
    resp = requests.get(ARCHIVE_API_URL, params={
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto"
    })
    data = resp.json()
    if "hourly" not in data:
        raise HTTPException(500, "Archive API did not return hourly data")
    return pd.DataFrame(data["hourly"])

def save_plots(df, city, state):
    plots = {}
    for col, label in [
        ("temperature_2m", "Temperature (°C)"),
        ("wind_speed_10m", "Wind Speed (km/h)"),
        ("precipitation", "Precipitation (mm/h)"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(df["time"], df[col], marker="o")
        plt.title(f"{label} over 10 years")
        plt.xlabel("Time")
        plt.ylabel(label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = f"{city}_{state}_{col}.png".replace(" ", "_")
        filepath = STATIC_DIR / filename
        plt.savefig(filepath)
        plt.close()
        plots[col] = f"/static/{filename}"
    return plots

# --- FASTAPI APP WITH LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    trained = train_nasa_model()
    if not trained:
        print("⚠️ NASA model not trained. Make sure NasaWeatherPred.csv exists.")
    yield

app = FastAPI(title="Weather AI Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- SCHEMAS ---
class PredictRequest(BaseModel):
    state: str
    city: str
    date: str
    hour: int

class PredictDayRequest(BaseModel):
    state: str
    city: str
    date: str

# --- ROUTES ---
@app.post("/predict")
def predict(req: PredictRequest):
    lat, lon, city_name = get_lat_lon(req.city, req.state)
    date_obj = datetime.strptime(req.date, "%Y-%m-%d")

    start_date = (date_obj - timedelta(days=3650)).strftime("%Y-%m-%d")
    end_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_archive(lat, lon, start_date, end_date)

    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour

    df_day = df[(df["time"].dt.month == date_obj.month) & (df["time"].dt.day == date_obj.day)]
    df_hour = df_day[df_day["hour"] == req.hour]

    if df_hour.empty:
        raise HTTPException(404, "No data available for this hour")

    avg = {
        "Temperature (°C)": df_hour["temperature_2m"].mean(),
        "Humidity (%)": df_hour["relative_humidity_2m"].mean(),
        "Precipitation (mm/h)": df_hour["precipitation"].mean(),
        "Wind Speed (km/h)": df_hour["wind_speed_10m"].mean()
    }
    X_input = pd.DataFrame([avg])
    y_pred = NASA_MODEL.predict(X_input)[0]
    condition = NASA_LABEL_ENCODER.inverse_transform([y_pred])[0]

    plots = save_plots(df_day, req.city, req.state)

    return {
        "city": city_name,
        "state": req.state,
        "date": req.date,
        "hour": req.hour,
        "prediction": avg,
        "predicted_weather": condition,
        "plots": {
            "temperature_plot": plots["temperature_2m"],
            "wind_plot": plots["wind_speed_10m"],
            "precipitation_plot": plots["precipitation"],
        }
    }

@app.post("/predict_day")
def predict_day(req: PredictDayRequest):
    lat, lon, city_name = get_lat_lon(req.city, req.state)
    date_obj = datetime.strptime(req.date, "%Y-%m-%d")

    start_date = (date_obj - timedelta(days=3650)).strftime("%Y-%m-%d")
    end_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_archive(lat, lon, start_date, end_date)

    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour

    df_day = df[(df["time"].dt.month == date_obj.month) & (df["time"].dt.day == date_obj.day)]
    if df_day.empty:
        raise HTTPException(404, "No data available for this day")

    hourly_preds = []
    for hour in range(24):
        df_hour = df_day[df_day["hour"] == hour]
        if df_hour.empty:
            continue
        avg = {
            "Temperature (°C)": df_hour["temperature_2m"].mean(),
            "Humidity (%)": df_hour["relative_humidity_2m"].mean(),
            "Precipitation (mm/h)": df_hour["precipitation"].mean(),
            "Wind Speed (km/h)": df_hour["wind_speed_10m"].mean()
        }
        X_input = pd.DataFrame([avg])
        y_pred = NASA_MODEL.predict(X_input)[0]
        condition = NASA_LABEL_ENCODER.inverse_transform([y_pred])[0]
        hourly_preds.append({
            "hour": hour,
            "temperature": avg["Temperature (°C)"],
            "humidity": avg["Humidity (%)"],
            "precipitation": avg["Precipitation (mm/h)"],
            "wind_speed": avg["Wind Speed (km/h)"],
            "predicted_weather": condition
        })

    return {
        "city": city_name,
        "state": req.state,
        "date": req.date,
        "predictions": hourly_preds
    }

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
