import requests
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
# CORS(app)  # 🔥 this enables CORS for all routes
# Allow only React dev server
CORS(app, origins=["http://localhost:3000"])


VC_API_KEY = "JYJP42CES8DXR62C4RGKEPG8T"


# --------------------------
# Helpers for unit conversion
# --------------------------
def c_to_f(c):
    return None if c is None else round((c * 9/5) + 32, 1)

def ms_to_mph(ms):
    return None if ms is None else round(ms * 2.23694, 1)

def derive_condition(temp_c, precip_mm, cloud_cover=None):
    """Simple heuristic for condition description."""
    if precip_mm and precip_mm > 2:
        return "Rainy"
    elif precip_mm and precip_mm > 0:
        return "Light Rain"
    elif cloud_cover is not None:
        if cloud_cover > 70:
            return "Cloudy"
        elif cloud_cover > 30:
            return "Partly Cloudy"
        else:
            return "Clear/Sunny"
    elif temp_c is not None and temp_c > 25:
        return "Sunny"
    else:
        return "Clear/Cloudy"

# --------------------------
# API Calls
# --------------------------

def forecast_open_meteo(lat, lon, target_date):
    """Forecast up to 16 days ahead (Open-Meteo, free)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,cloudcover_mean",
        "timezone": "auto",
        "start_date": target_date,
        "end_date": target_date,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def visual_crossing_weather(location, target_date, api_key, unit_group="metric"):
    """Historical + long-term forecast/climatology from Visual Crossing."""
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{target_date}"
    params = {
        "unitGroup": unit_group,
        "key": api_key,
        "include": "days"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

# --------------------------
# Main function
# --------------------------

def get_weather_any_date(zip_code, target_date, vc_api_key):
    """
    Unified weather retriever:
      - Past dates → Visual Crossing (historical)
      - Near future (<=16 days) → Open-Meteo
      - Far future (>16 days) → Visual Crossing (forecast/climatology)
    """
    if isinstance(target_date, str):
        target = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        target = target_date

    today = datetime.date.today()
    delta = (target - today).days

    if target <= today:
        # Historical → Visual Crossing
        data = visual_crossing_weather(zip_code, target.strftime("%Y-%m-%d"), vc_api_key)
        day = data["days"][0]
        weather = {
            "T2M_MAX": day.get("tempmax"),
            "T2M_MIN": day.get("tempmin"),
            "PRECIP": day.get("precip"),
            "WIND": day.get("windspeed"),
            "CLOUD": day.get("cloudcover")
        }
        source = "VisualCrossing-Historical"

    elif delta <= 16:
        # Forecast → Open-Meteo
        # First need lat/lon → use Visual Crossing to geocode ZIP
        loc_data = visual_crossing_weather(zip_code, target.strftime("%Y-%m-%d"), vc_api_key)
        lat, lon = loc_data["latitude"], loc_data["longitude"]

        data = forecast_open_meteo(lat, lon, target.strftime("%Y-%m-%d"))
        daily = data["daily"]

        weather = {
            "T2M_MAX": daily.get("temperature_2m_max", [None])[0],
            "T2M_MIN": daily.get("temperature_2m_min", [None])[0],
            "PRECIP": daily.get("precipitation_sum", [None])[0],
            "WIND": daily.get("windspeed_10m_max", [None])[0],
            "CLOUD": daily.get("cloudcover_mean", [None])[0],
        }
        source = "OpenMeteo-Forecast"

    else:
        # Long-term future → Visual Crossing forecast/climatology
        data = visual_crossing_weather(zip_code, target.strftime("%Y-%m-%d"), vc_api_key)
        day = data["days"][0]
        weather = {
            "T2M_MAX": day.get("tempmax"),
            "T2M_MIN": day.get("tempmin"),
            "PRECIP": day.get("precip"),
            "WIND": day.get("windspeed"),
            "CLOUD": day.get("cloudcover")
        }
        source = "VisualCrossing-Forecast/Climatology"

    # --------------------------
    # Format friendly output
    # --------------------------
    friendly = {
        "High (°F)": c_to_f(weather.get("T2M_MAX")),
        "Low (°F)": c_to_f(weather.get("T2M_MIN")),
        "Wind (mph)": ms_to_mph(weather.get("WIND")),
        "Precipitation (mm)": weather.get("PRECIP"),
        "Condition": derive_condition(weather.get("T2M_MAX"), weather.get("PRECIP"), weather.get("CLOUD")),
        "Source": source
    }
    return target, friendly



@app.route("/predict", methods=["POST"])
def predict_weather():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    city = data.get("city")
    state = data.get("state")
    date_str = data.get("date")
    hour_str = data.get("hour")

    if not state or not city or not date_str:
        return jsonify({"error": "Missing required fields: city, state, date"}), 400

    try:
        target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Date must be YYYY-MM-DD"}), 400

    location = f"{city},{state}"
    best_date, weather = get_weather_any_date(location, target_date, VC_API_KEY)

    if hour_str is not None:
        weather["Requested Hour"] = hour_str

    return jsonify({
        "location": location,
        "date": str(best_date),
        "weather": weather
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
