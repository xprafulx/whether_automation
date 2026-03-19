from __future__ import annotations

import sqlite3
from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# -------------------------------
# Config
# -------------------------------
LOCATIONS = [
    {"name": "Kathmandu", "lat": 27.7017, "lon": 85.3206},
    {"name": "Copenhagen", "lat": 55.6759, "lon": 12.5655},
    {"name": "Aalborg", "lat": 57.048, "lon": 9.9187},
]

BASE_URL = "https://api.open-meteo.com/v1/forecast"


# -------------------------------
# API Client
# -------------------------------
def create_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# -------------------------------
# Fetch Weather Data
# -------------------------------
def fetch_weather() -> pd.DataFrame:
    client = create_client()

    params = {
        "latitude": [loc["lat"] for loc in LOCATIONS],
        "longitude": [loc["lon"] for loc in LOCATIONS],
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "relative_humidity_2m",
            "precipitation",
            "cloud_cover",
        ],
        "timezone": "Europe/Copenhagen",
        "forecast_days": 2,
    }

    responses = client.weather_api(BASE_URL, params=params)

    all_data = []

    for idx, response in enumerate(responses):
        location_name = LOCATIONS[idx]["name"]
        hourly = response.Hourly()

        df = pd.DataFrame({
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "wind": hourly.Variables(1).ValuesAsNumpy(),
            "humidity": hourly.Variables(2).ValuesAsNumpy(),
            "precipitation": hourly.Variables(3).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
        })

        df["location"] = location_name
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour

        all_data.append(df)

    return pd.concat(all_data)


# -------------------------------
# Transform Data (Tomorrow + Periods)
# -------------------------------
def transform_weather(df: pd.DataFrame) -> pd.DataFrame:
    tomorrow = pd.Timestamp.now().date() + pd.Timedelta(days=1)
    df = df[df["date"] == tomorrow].copy()

    def get_period(hour: int) -> str:
        if 6 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 17:
            return "afternoon"
        elif 18 <= hour <= 21:
            return "evening"
        else:
            return "night"

    df["period"] = df["hour"].apply(get_period)

    summary = df.groupby(["location", "date", "period"]).agg({
        "temperature": "mean",
        "wind": "mean",
        "precipitation": "sum",
        "humidity": "mean",
        "cloud_cover": "mean"
    }).reset_index()

    return summary.round(2)


# -------------------------------
# Store in SQLite
# -------------------------------
def store_weather(data: pd.DataFrame) -> None:
    conn = sqlite3.connect("data/weather.db")
    data.to_sql("weather", conn, if_exists="replace", index=False)
    conn.close()


# -------------------------------
# Main
# -------------------------------
def main():
    raw_data = fetch_weather()
    processed_data = transform_weather(raw_data)
    store_weather(processed_data)

    print("✅ Weather data pipeline completed")
    print(processed_data.head())


if __name__ == "__main__":
    main()