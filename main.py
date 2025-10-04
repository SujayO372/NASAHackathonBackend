# main.py - Weather AI Backend
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ------------------ Config ------------------
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "pressure_msl",
    "cloudcover",
    "visibility",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation"
]

# ------------------ FastAPI ------------------
app = FastAPI(title="Weather-AI Backend", version="1.0")

# Enhanced CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Request Schemas ------------------
class DayRequest(BaseModel):
    city: str
    state: str
    date: str  # YYYY-MM-DD

class HourRequest(DayRequest):
    hour: int  # 0-23

# ------------------ Utilities ------------------
def geolocate_place(city: str, state: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (lat, lon, timezone) or (None, None, None)."""
    place_name = f"{city}, {state}, US"
    try:
        print(f"[geolocate] querying geocoding for: {place_name}")
        resp = requests.get(
            GEOCODING_API_URL,
            params={"name": place_name, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("results"):
            print("[geolocate] no results returned")
            return None, None, None
        result = data["results"][0]
        if result.get("country_code") != "US":
            print(f"[geolocate] non-US result: {result.get('country_code')}")
            return None, None, None
        lat = result.get("latitude")
        lon = result.get("longitude")
        tz = result.get("timezone") or "UTC"
        print(f"[geolocate] found: lat={lat}, lon={lon}, tz={tz}")
        return lat, lon, tz
    except requests.exceptions.RequestException as e:
        print(f"[geolocate] request error: {e}")
        return None, None, None

def fetch_weather_data(lat: float, lon: float, start_date: str, end_date: str, timezone: str) -> Optional[dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone
    }
    try:
        print(f"[fetch] archive API: {start_date} -> {end_date} tz={timezone}")
        resp = requests.get(ARCHIVE_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[fetch] request error: {e}")
        return None

def synthesize_weather_type(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize weather type based on conditions"""
    df = df.copy()
    df['Weather Type'] = None
    
    df.loc[df['Precipitation (mm/h)'] > 1.0, 'Weather Type'] = 'Heavy Rain/Snow'
    df.loc[(df['Precipitation (mm/h)'] > 0.1) & (df['Precipitation (mm/h)'] <= 1.0), 'Weather Type'] = 'Light Rain/Drizzle'
    df.loc[(df['Cloud Cover (%)'] >= 75) & (df['Precipitation (mm/h)'] <= 0.1), 'Weather Type'] = 'Overcast'
    df.loc[(df['Cloud Cover (%)'] >= 30) & (df['Cloud Cover (%)'] < 75) & (df['Precipitation (mm/h)'] <= 0.1), 'Weather Type'] = 'Partly Cloudy'
    df.loc[df['Weather Type'].isna(), 'Weather Type'] = 'Clear/Sunny'
    
    return df

def predict_hour(df_filtered: pd.DataFrame, analysis_data: pd.Series) -> str:
    """Predict weather type using Random Forest"""
    if df_filtered.shape[0] < 1:
        return "Undefined"
    
    required_cols = ["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]
    if not all(c in df_filtered.columns for c in required_cols):
        return "Undefined"
    
    X = df_filtered[required_cols]
    y = df_filtered["Weather Type"]
    
    if y.nunique() <= 1:
        return y.iloc[0] if not y.empty else "Undefined"
    
    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_enc)
        
        input_df = pd.DataFrame([[
            analysis_data.get('Temperature (°C)', 0.0),
            analysis_data.get('Wind Speed (km/h)', 0.0),
            analysis_data.get('Precipitation (mm/h)', 0.0)
        ]], columns=required_cols)
        
        pred = model.predict(input_df)[0]
        return le.inverse_transform([pred])[0]
    except Exception as e:
        print(f"[predict_hour] error: {e}")
        return "Undefined"

def get_day_predictions(city: str, state: str, date_str: str) -> Tuple[List[dict], str]:
    """Get predictions for all 24 hours of a given day"""
    print("inside get_day_predictions")
    
    try:
        upcoming_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    historical_end = min(upcoming_date - timedelta(days=1), datetime.now().date() - timedelta(days=1))
    historical_start = historical_end - timedelta(days=365 * 10)
    
    # Remove older CSVs
    for f in glob.glob("historical_weather_data_*.csv"):
        try:
            os.remove(f)
        except Exception:
            pass

    lat, lon, tz = geolocate_place(city, state)
    print(f"[get_day_predictions] geolocated to lat={lat}, lon={lon}, tz={tz}")
    
    if lat is None:
        raise HTTPException(status_code=404, detail="Location not found or not in the USA.")

    data = fetch_weather_data(lat, lon, historical_start.strftime('%Y-%m-%d'), historical_end.strftime('%Y-%m-%d'), tz)
    print(f"[get_day_predictions] fetched historical data from {historical_start} to {historical_end}")
    
    if data is None:
        raise HTTPException(status_code=502, detail="Failed fetching historical weather data from provider.")
    if "hourly" not in data:
        raise HTTPException(status_code=502, detail="Historical API response missing hourly data.")

    df = pd.DataFrame(data["hourly"])
    rename_map = {
        "time": "Timestamp",
        "temperature_2m": "Temperature (°C)",
        "pressure_msl": "Atmospheric Pressure (hPa)",
        "cloudcover": "Cloud Cover (%)",
        "visibility": "Visibility (m)",
        "relative_humidity_2m": "Relative Humidity (%)",
        "wind_speed_10m": "Wind Speed (km/h)",
        "precipitation": "Precipitation (mm/h)"
    }
    df = df.rename(columns=rename_map)
    
    if "Timestamp" not in df.columns:
        raise HTTPException(status_code=502, detail="Historical API response missing 'time' values.")
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    target_month, target_day = upcoming_date.month, upcoming_date.day
    df_day = df[(df['Timestamp'].dt.month == target_month) & (df['Timestamp'].dt.day == target_day)].copy()
    
    if df_day.empty:
        raise HTTPException(status_code=404, detail=f"No historical hourly records found for {target_month:02d}-{target_day:02d}.")

    predictions: List[dict] = []
    for hour in range(24):
        hours_window = [(hour - 1) % 24, hour, (hour + 1) % 24]
        df_hour_window = df_day[df_day['Timestamp'].dt.hour.isin(hours_window)].copy()
        
        if df_hour_window.empty:
            predictions.append({
                "hour": hour,
                "predicted_weather": "Undefined",
                "temperature": None,
                "wind_speed": None,
                "precipitation": None,
                "pressure": None,
                "cloud_cover": None,
                "visibility": None,
                "humidity": None,
                "history_count": 0
            })
            continue

        df_hour_window = synthesize_weather_type(df_hour_window)
        analysis_data = df_hour_window.drop(columns=['Timestamp', 'Weather Type']).mean()
        predicted_weather = predict_hour(df_hour_window, analysis_data)

        predictions.append({
            "hour": hour,
            "predicted_weather": predicted_weather,
            "temperature": None if pd.isna(analysis_data.get('Temperature (°C)')) else round(float(analysis_data['Temperature (°C)']), 1),
            "wind_speed": None if pd.isna(analysis_data.get('Wind Speed (km/h)')) else round(float(analysis_data['Wind Speed (km/h)']), 1),
            "precipitation": None if pd.isna(analysis_data.get('Precipitation (mm/h)')) else round(float(analysis_data['Precipitation (mm/h)']), 2),
            "pressure": None if pd.isna(analysis_data.get('Atmospheric Pressure (hPa)')) else round(float(analysis_data['Atmospheric Pressure (hPa)']), 1),
            "cloud_cover": None if pd.isna(analysis_data.get('Cloud Cover (%)')) else round(float(analysis_data['Cloud Cover (%)']), 1),
            "visibility": None if pd.isna(analysis_data.get('Visibility (m)')) else round(float(analysis_data['Visibility (m)']), 1),
            "humidity": None if pd.isna(analysis_data.get('Relative Humidity (%)')) else round(float(analysis_data['Relative Humidity (%)']), 1),
            "history_count": int(len(df_hour_window))
        })

    # Build summary
    temps = [p["temperature"] for p in predictions if p["temperature"] is not None]
    hums = [p["humidity"] for p in predictions if p["humidity"] is not None]
    rains = [p for p in predictions if p["predicted_weather"] and "Rain" in p["predicted_weather"]]

    avg_temp = (sum(temps) / len(temps)) if temps else None
    avg_humidity = (sum(hums) / len(hums)) if hums else None

    if avg_temp is not None and avg_humidity is not None:
        summary = f"Expect an average temperature of {avg_temp:.1f}°C with humidity around {avg_humidity:.0f}%. "
    elif avg_temp is not None:
        summary = f"Expect an average temperature of {avg_temp:.1f}°C. "
    else:
        summary = "Insufficient historical data to compute averages. "

    summary += f"Rain likely in {len(rains)} hour(s) of the day." if len(rains) > 0 else "No rain is expected based on historical averages."

    return predictions, summary

# ------------------ API Endpoints ------------------
@app.get("/")
def root():
    return {
        "service": "weather-ai-backend",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Single hour prediction",
            "/predict_day": "POST - Full day (24h) prediction",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict_day")
def predict_day(req: DayRequest, request: Request):
    print("inside predict_day")
    """Predict weather for all 24 hours of a given day"""
    print(f"[predict_day] request from {request.client.host if request.client else 'unknown'} -> {req.city}, {req.state}, {req.date}")
    
    try:
        preds, summary = get_day_predictions(req.city, req.state, req.date)
        return {
            "city": req.city,
            "state": req.state,
            "date": req.date,
            "predictions": preds,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict_day] unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict")
def predict(req: HourRequest, request: Request):
    """Predict weather for a specific hour"""
    print(f"[predict] request from {request.client.host if request.client else 'unknown'} -> {req.city}, {req.state}, {req.date} @ {req.hour}")
    
    if not (0 <= req.hour <= 23):
        raise HTTPException(status_code=400, detail="Hour must be 0-23.")
    
    try:
        preds, summary = get_day_predictions(req.city, req.state, req.date)
        hour_obj = next((p for p in preds if p["hour"] == req.hour), None)
        
        if hour_obj is None:
            raise HTTPException(status_code=404, detail="Prediction for requested hour not available.")
        
        history_count = hour_obj.get("history_count", 0)
        conf = max(0.25, min(0.98, 0.25 + 0.007 * min(history_count, 100)))
        
        return {
            "city": req.city,
            "state": req.state,
            "date": req.date,
            "prediction": hour_obj,
            "confidence": round(conf, 2),
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict] unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ------------------ CRITICAL: Run Server ------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌦️  WEATHER AI BACKEND SERVER STARTING...")
    print("="*70)
    print("📡 Server will be available at:")
    print("   • http://127.0.0.1:8000")
    print("   • http://localhost:8000")
    print("\n📚 API Documentation:")
    print("   • Swagger UI: http://127.0.0.1:8000/docs")
    print("   • ReDoc: http://127.0.0.1:8000/redoc")
    print("\n💡 Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        print("❌ ERROR: uvicorn is not installed!")
        print("   Run: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")