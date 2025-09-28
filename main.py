import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import random
import glob  # Added for finding files to delete
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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


def synthesize_weather_type(df):
    # Synthetically create a 'Weather Type' label for training the model
    # Rule 1: Heavy Rain/Snow (precipitation > 1.0 mm/h)
    df.loc[df['Precipitation (mm/h)'] > 1.0, 'Weather Type'] = 'Heavy Rain/Snow'
    # Rule 2: Light Rain/Drizzle (0.1 < precipitation <= 1.0 mm/h)
    df.loc[
        (df['Precipitation (mm/h)'] > 0.1) & (df['Precipitation (mm/h)'] <= 1.0), 'Weather Type'] = 'Light Rain/Drizzle'
    # Rule 3: Overcast (cloudcover >= 75%) and no rain
    df.loc[(df['Cloud Cover (%)'] >= 75) & (df['Precipitation (mm/h)'] <= 0.1), 'Weather Type'] = 'Overcast'
    # Rule 4: Partly Cloudy (30% <= cloudcover < 75%) and no rain
    df.loc[(df['Cloud Cover (%)'] >= 30) & (df['Cloud Cover (%)'] < 75) & (
                df['Precipitation (mm/h)'] <= 0.1), 'Weather Type'] = 'Partly Cloudy'
    # Rule 5: Clear/Sunny (cloudcover < 30%) and no rain (fill remaining)
    df.loc[df['Weather Type'].isna(), 'Weather Type'] = 'Clear/Sunny'

    return df


def process_and_save(weather_data, location_name, target_hour, upcoming_date_str):
    if not weather_data or "hourly" not in weather_data:
        print("Error: No valid hourly data received.")
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

    df_day_month_filtered = df[
        (df['Timestamp'].dt.month == target_month) &
        (df['Timestamp'].dt.day == target_day)
        ]

    hours_to_keep = [(target_hour + i) % 24 for i in range(-1, 2)]

    df_filtered = df_day_month_filtered[df_day_month_filtered['Timestamp'].dt.hour.isin(hours_to_keep)].copy()

    if df_filtered.empty:
        print(
            f"\nWarning: No data found for the specified date ({target_month:02d}-{target_day:02d}) and hour window over 10 years.")
        return

    df_filtered = synthesize_weather_type(df_filtered)

    # Calculate the 10-year mean (input for the prediction)
    analysis_data = df_filtered.drop(columns=['Timestamp', 'Weather Type']).mean()

    # --- Machine Learning Prediction Logic ---
    X = df_filtered[["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]]
    y = df_filtered["Weather Type"]

    if y.nunique() <= 1:
        predicted_weather = y.iloc[0] if not y.empty else "Undefined"
        print(
            "Warning: Only one 'Weather Type' found in historical data. Skipping ML training and using the only available type.")
    else:
        # Encode the target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train the model using all filtered historical data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        # Prepare the mean data for prediction
        input_data = pd.DataFrame(
            [[analysis_data['Temperature (°C)'], analysis_data['Wind Speed (km/h)'],
              analysis_data['Precipitation (mm/h)']]],
            columns=["Temperature (°C)", "Wind Speed (km/h)", "Precipitation (mm/h)"]
        )

        # Predict the weather type based on the 10-year average features
        prediction = model.predict(input_data)[0]
        predicted_weather = label_encoder.inverse_transform([prediction])[0]

    # --- Output and Save ---

    print("\n" + "=" * 50)
    print(f"PREDICTIVE ANALYSIS FOR {location_name} on {upcoming_date_str} at {target_hour:02d}:00:")
    print("--------------------------------------------------")

    prediction_line = (
        f"Expected Features (10-Year Average +/- 1h on {target_month:02d}-{target_day:02d}):\n"
        f"PREDICTED WEATHER: {predicted_weather}\n"
        f"Temp: {analysis_data['Temperature (°C)']:.1f}°C, "
        f"Wind Speed: {analysis_data['Wind Speed (km/h)']:.1f}km/h, "
        f"Precipitation: {analysis_data['Precipitation (mm/h)']:.1f}mm/h, "
        f"Pressure: {analysis_data['Atmospheric Pressure (hPa)']:.1f}hPa, "
        f"Cloud Cover: {analysis_data['Cloud Cover (%)']:.0f}%, "
        f"Visibility: {analysis_data['Visibility (m)']:.0f}m, "
        f"Humidity: {analysis_data['Relative Humidity (%)']:.0f}%"
    )
    print(prediction_line)
    print("--------------------------------------------------")

    max_rows = 1000
    if len(df_filtered) > max_rows:
        df_final = df_filtered.sample(n=max_rows, random_state=42)
        print(f"Data was sampled down to {max_rows} rows from {len(df_filtered)} historical hourly entries.")
    else:
        df_final = df_filtered
        print(f"Data contains {len(df_final)} entries (no sampling required).")

    safe_location = "".join(c for c in location_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    filename = f"historical_weather_data_{safe_location}.csv"

    df_final.to_csv(filename, index=False)
    print(f"\nSUCCESS! Sampled historical data saved to {filename}")
    print(f"Sample data head:\n{df_final.head().to_string(index=False)}")
    print("=" * 50)


def main():

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

    # --- NEW FILE DELETION LOGIC ---
    # Delete any existing historical weather CSV file regardless of its location name
    files_to_delete = glob.glob("historical_weather_data_*.csv")
    for file_to_delete in files_to_delete:
        try:
            os.remove(file_to_delete)
            print(f"\nSuccessfully deleted old file: {file_to_delete}")
        except OSError as e:
            print(f"Error deleting file {file_to_delete}: {e}")
    # --- END NEW FILE DELETION LOGIC ---

    safe_location = "".join(c for c in location_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    filename = f"historical_weather_data_{safe_location}.csv"  # This filename variable is now only used to name the *new* file

    latitude, longitude, timezone = geolocate_place(location_name)
    if latitude is None:
        return

    data = fetch_weather_data(latitude, longitude, start_date_str, end_date_str, timezone)
    if data is None:
        return

    process_and_save(data, location_name, target_hour, upcoming_date_str)


if __name__ == "__main__":
    main()