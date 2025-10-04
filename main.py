import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- GLOBAL CONSTANTS ---
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
NASA_FILENAME = "DataNasaWeatherPredFinal.csv" #Changed CSV File

HOURLY_VARIABLES = [
    "temperature_2m",
    "pressure_msl",
    "cloudcover",
    "visibility",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation"
]

# Global variables to store the pre-trained model and encoder
NASA_MODEL = None
NASA_LABEL_ENCODER = None


# --- NEW FUNCTION: Train the model on the NASA CSV ---
def train_nasa_model():
    """Loads NASA data, trains a RandomForestClassifier, and stores it globally."""
    global NASA_MODEL, NASA_LABEL_ENCODER

    print("\n" + "=" * 50)
    print(f"🚀 Training Model on External Data: {NASA_FILENAME}")
    print("=" * 50)

    if not os.path.exists(NASA_FILENAME):
        print(f"FATAL ERROR: The file '{NASA_FILENAME}' was not found.")
        print("Please ensure the CSV is in the same directory as the script.")
        return False

    try:
        # Load the NASA dataset
        nasa_df = pd.read_csv(NASA_FILENAME)

        # Prepare features (X) and target (y)
        X = nasa_df[["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]]
        y = nasa_df["Weather Type"]

        # Check for missing values in training columns
        if X.isnull().any().any() or y.isnull().any():
            print("Warning: Missing values found in NASA data. Dropping rows with NaN...")
            combined_df = pd.concat([X, y], axis=1).dropna()
            X = combined_df[["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]]
            y = combined_df["Weather Type"]

        if y.nunique() <= 1:
            print("Error: The NASA CSV does not contain enough unique 'Weather Type' labels for training.")
            return False

        # Encode the target labels
        NASA_LABEL_ENCODER = LabelEncoder()
        y_encoded = NASA_LABEL_ENCODER.fit_transform(y)

        # Train the Random Forest model
        NASA_MODEL = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        NASA_MODEL.fit(X, y_encoded)

        print("✅ NASA Model Training Complete!")
        print(f"   Trained on {len(nasa_df)} records with {y.nunique()} unique weather types.")
        return True

    except Exception as e:
        print(f"ERROR during NASA model training: {e}")
        return False


# --- EXISTING FUNCTION: Geolocation (No Change) ---
def geolocate_place(place_name):
    print(f"-> Searching for location: {place_name}...")
    try:
        response = requests.get(GEOCODING_API_URL,
                                params={"name": place_name, "count": 1, "language": "en", "format": "json"})
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            print("Error: Location not found. Please check spelling or use specific city/country names.")
            return None, None, None

        result = data["results"][0]

        if result.get("country_code") != "US":
            print(
                f"Error: Location must be in the USA. Found location in {result.get('country_code', 'an unknown country')}.")
            return None, None, None

        latitude = result["latitude"]
        longitude = result["longitude"]
        timezone = result["timezone"]
        print(f"-> Found location: {result['name']} ({latitude:.2f}, {longitude:.2f})")
        return latitude, longitude, timezone

    except requests.exceptions.RequestException as e:
        print(f"Error during geolocation API call: {e}")
        return None, None, None


# --- EXISTING FUNCTION: Fetch Weather Data (No Change) ---
def fetch_weather_data(lat, lon, start_date, end_date, timezone):
    print(f"-> Fetching historical data from {start_date} to {end_date}...")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone
    }

    try:
        response = requests.get(ARCHIVE_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error during weather API call: {e}")
        print("Tip: Check if your date range is too long or if the API URL is correct.")
        return None


# --- DEPRECATED FUNCTION: No longer needed as we use NASA labels ---
def synthesize_weather_type(df):
    """Placeholder to indicate this function is skipped in the new logic."""
    print("Skipping synthetic labeling: Using pre-trained NASA model labels.")
    return df


# --- MODIFIED FUNCTION: Uses the globally trained NASA model ---
def process_and_save(weather_data, location_name, target_hour, upcoming_date_str):
    global NASA_MODEL, NASA_LABEL_ENCODER

    if not weather_data or "hourly" not in weather_data:
        print("Error: No valid hourly data received.")
        return

    if NASA_MODEL is None or NASA_LABEL_ENCODER is None:
        print("FATAL ERROR: NASA model is not trained. Cannot proceed with prediction.")
        return

    hourly_data = weather_data["hourly"]
    df = pd.DataFrame(hourly_data)

    column_mapping = {
        "time": "Timestamp",
        "temperature_2m": "Temperature (°C)",
        "pressure_msl": "Atmospheric Pressure (hPa)",
        "cloudcover": "Cloud Cover (%)",
        "visibility": "Visibility (m)",
        "relative_humidity_2m": "Relative Humidity (%)",
        "wind_speed_10m": "Wind Speed (km/h)",
        "precipitation": "Precipitation (mm/h)"
    }
    df = df.rename(columns=column_mapping)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    target_hour = int(target_hour)

    upcoming_date = datetime.strptime(upcoming_date_str, '%Y-%m-%d').date()
    target_month = upcoming_date.month
    target_day = upcoming_date.day

    # Filter by month and day over the 10 years
    df_day_month_filtered = df[
        (df['Timestamp'].dt.month == target_month) &
        (df['Timestamp'].dt.day == target_day)
        ]

    # Filter by the target hour and +/- 1 hour
    hours_to_keep = [(target_hour + i) % 24 for i in range(-1, 2)]

    df_filtered = df_day_month_filtered[df_day_month_filtered['Timestamp'].dt.hour.isin(hours_to_keep)].copy()

    if df_filtered.empty:
        print(
            f"\nWarning: No data found for the specified date ({target_month:02d}-{target_day:02d}) and hour window over 10 years.")
        return

    # Calculate the 10-year mean (input for the prediction)
    # This mean is calculated from the Open-Meteo historical data
    analysis_data = df_filtered.drop(columns=['Timestamp']).mean()

    # --- Prediction Logic using NASA Model ---

    # Prepare the mean data for prediction, matching NASA training features
    input_data = pd.DataFrame(
        [[analysis_data['Temperature (°C)'], analysis_data['Wind Speed (km/h)'],
          analysis_data['Precipitation (mm/h)']]],
        columns=["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]
    )

    # Predict the weather type based on the 10-year average features
    prediction = NASA_MODEL.predict(input_data)[0]
    predicted_weather = NASA_LABEL_ENCODER.inverse_transform([prediction])[0]

    # --- Output and Save ---

    print("\n" + "=" * 50)
    print(f"📊 PREDICTIVE ANALYSIS FOR {location_name} on {upcoming_date_str} at {target_hour:02d}:00:")
    print("--------------------------------------------------")
    print(f"🔮 PREDICTED WEATHER (using NASA Model): {predicted_weather}")
    print("--------------------------------------------------")

    prediction_line = (
        f"Open-Meteo Expected Features (10-Year Average +/- 1h on {target_month:02d}-{target_day:02d}):\n"
        f"Temp: {analysis_data['Temperature (°C)']:.1f}°C, "
        f"Wind Speed: {analysis_data['Wind Speed (km/h)']:.1f}km/h, "
        f"Precipitation: {analysis_data['Precipitation (mm/h)']:.1f}mm/h, "
        f"Pressure: {analysis_data['Atmospheric Pressure (hPa)']:.1f}hPa, "
        f"Cloud Cover: {analysis_data['Cloud Cover (%)']:.0f}%, "
        f"Visibility: {analysis_data['Visibility (m)']:.0f}m, "
        f"Humidity: {analysis_data['Relative Humidity (%)']:.0f}%"
    )
    print(prediction_line)
    print("=" * 50)

    # Save the Open-Meteo historical data (without synthetic label, as it's not used)
    df_final = df_filtered.copy()  # Use the full filtered data for saving

    # Save a column with the 10-year average for easy reference in the CSV
    for col in analysis_data.index:
        df_final[f'10_Year_Avg_{col}'] = analysis_data[col]

    safe_location = "".join(c for c in location_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    filename = f"open_meteo_historical_data_{safe_location}.csv"

    # Limit to max rows for saving
    max_rows = 1000
    if len(df_final) > max_rows:
        df_final = df_final.sample(n=max_rows, random_state=42)
        print(f"Data was sampled down to {max_rows} rows from {len(df_filtered)} historical hourly entries for saving.")
    else:
        print(f"Data contains {len(df_final)} entries (no sampling required for saving).")

    df_final.to_csv(filename, index=False)
    print(f"\nSUCCESS! Sampled Open-Meteo historical data (with 10-year averages) saved to {filename}")
    print(f"Sample data head:\n{df_final.head().to_string(index=False)}")
    print("=" * 50)


# --- MODIFIED MAIN FUNCTION: Includes NASA Model Training ---
def main():
    # 1. Train the NASA model first
    if not train_nasa_model():
        print("\nExiting program because the NASA model could not be trained.")
        return

    # 2. Get user input for location and date
    location_name = input("\nEnter the place name (MUST be in the USA, e.g., 'New York City', 'Chicago'): ").strip()
    upcoming_date_str = input("Enter the UPCOMING date (YYYY-MM-DD): ").strip()
    target_hour = input("Enter the TARGET hour for prediction (0-23, e.g., 14 for 2 PM): ").strip()

    try:
        upcoming_date = datetime.strptime(upcoming_date_str, '%Y-%m-%d').date()
        target_hour = int(target_hour)
        if not (0 <= target_hour <= 23):
            raise ValueError("Hour must be between 0 and 23.")
    except ValueError as e:
        print(f"\nError: Invalid input format. Check your date (YYYY-MM-DD) or hour (0-23). Details: {e}")
        return

    # 3. Define historical date range (10 years up to the day before the prediction)
    requested_historical_end_date = upcoming_date - timedelta(days=1)
    latest_available_date = datetime.now().date() - timedelta(days=1)

    if requested_historical_end_date > latest_available_date:
        print(
            f"\nAPI Data Warning: Your requested date is too far in the future. Data will be fetched up to {latest_available_date} instead.")
        historical_end_date = latest_available_date
    else:
        historical_end_date = requested_historical_end_date

    historical_start_date = historical_end_date - timedelta(days=365 * 10)

    start_date_str = historical_start_date.strftime('%Y-%m-%d')
    end_date_str = historical_end_date.strftime('%Y-%m-%d')

    # 4. Clean up old Open-Meteo files
    files_to_delete = glob.glob("open_meteo_historical_data_*.csv")
    for file_to_delete in files_to_delete:
        try:
            os.remove(file_to_delete)
            print(f"\nSuccessfully deleted old file: {file_to_delete}")
        except OSError as e:
            print(f"Error deleting file {file_to_delete}: {e}")

    # 5. Geolocation and Data Fetch
    latitude, longitude, timezone = geolocate_place(location_name)
    if latitude is None:
        return

    data = fetch_weather_data(latitude, longitude, start_date_str, end_date_str, timezone)
    if data is None:
        return

    # 6. Process and Predict using the NASA model
    process_and_save(data, location_name, target_hour, upcoming_date_str)


if __name__ == "__main__":
    main()
