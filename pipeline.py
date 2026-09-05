#!/usr/bin/env python3
"""
Daily pipeline for the Skjern Å (Vardevej, station 251519) water-level forecast.

Runs once a day via GitHub Actions (see .github/workflows/daily.yml), with
full unrestricted internet access on the runner. It:

  1. Pulls the last ~20 days of actual rainfall + the next 7 days of forecast
     rainfall and general weather from Open-Meteo (free, no API key).
  2. Pulls the latest observed water level for station 251519 from
     Vandportalen's public plot-data endpoint.
  3. Applies the pre-trained rainfall -> water-level lag/regression model
     (model_coefficients.json) recursively to produce a 7-day water-level
     forecast.
  4. Writes forecast.json (consumed by the mobile web app) and appends the
     newly-observed day to data_log.csv, which future retraining can use to
     keep improving the model as more history accumulates.

No secrets or API keys are required for any of this - both data sources are
public, unauthenticated endpoints.
"""
import json
import csv
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

STATION_LAT = 55.9529
STATION_LON = 8.5469
STATION_NAME = "251519 Skjern Å, Vardevej"
VANDPORTALEN_TSID = 103787
VANDPORTALEN_PW = 524

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_coefficients.json")
FORECAST_OUT = os.path.join(os.path.dirname(__file__), "forecast.json")
LOG_OUT = os.path.join(os.path.dirname(__file__), "data_log.csv")

WEATHER_CODE_DESC = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def http_get_json(url, timeout=30, retries=2):
    """GET a URL and parse JSON, with basic retry + diagnostics on failure.

    Raises RuntimeError with the actual HTTP status/body snippet on failure,
    instead of letting a bare JSONDecodeError obscure what really happened.
    """
    last_err = None
    for attempt in range(1, retries + 2):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "skjern-aa-forecast/1.0 (+https://github.com/RasmusTerpstraLange/skjern-aa-forecast)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                raw = resp.read()
                if not raw:
                    raise RuntimeError(f"HTTP {status} but empty response body for {url}")
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as je:
                    snippet = raw[:300].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"HTTP {status} but non-JSON body for {url}: {je}. Body starts with: {snippet!r}"
                    )
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {e.code} {e.reason} for {url}. Body: {body!r}")
        except urllib.error.URLError as e:
            last_err = RuntimeError(f"Network error for {url}: {e.reason}")
        except RuntimeError as e:
            last_err = e

        print(f"  attempt {attempt} failed: {last_err}", file=sys.stderr)
        if attempt <= retries:
            time.sleep(3 * attempt)

    raise last_err


def fetch_weather():
    """Past ~20 days actual + next 7 days forecast, one call."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={STATION_LAT}&longitude={STATION_LON}"
        "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max,weather_code"
        "&past_days=20&forecast_days=7&timezone=Europe%2FCopenhagen"
    )
    data = http_get_json(url)
    daily = data["daily"]
    dates = daily["time"]
    out = {}
    for i, d in enumerate(dates):
        out[d] = {
            "precip_mm": daily.get("precipitation_sum", [None] * len(dates))[i],
            "temp_max_c": daily.get("temperature_2m_max", [None] * len(dates))[i],
            "temp_min_c": daily.get("temperature_2m_min", [None] * len(dates))[i],
            "precip_prob_max": daily.get("precipitation_probability_max", [None] * len(dates))[i],
            "weather_code": daily.get("weather_code", daily.get("weathercode", [None] * len(dates)))[i],
        }
    return out


def fetch_latest_water_level():
    """Returns (date_str YYYY-MM-DD, level_m) for the most recent COMPLETE calendar day."""
    now = datetime.now(timezone.utc)
    enddate = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    url = (
        "https://vandportalen.dk/api/hyd/getplotdata"
        f"?tsid={VANDPORTALEN_TSID}&enddate={enddate}&days=6&pw={VANDPORTALEN_PW}"
    )
    data = http_get_json(url)
    recs = data.get("PlotRecs", [])
    if not recs:
        raise RuntimeError("Vandportalen returned no PlotRecs")

    # Aggregate readings into daily means (matches how the training CSV was built:
    # "Dagdata... autogenereret på baggrund af Minut serie").
    daily_sums, daily_counts = {}, {}
    for rec in recs:
        day = rec["dt"][:10]
        vals = rec.get("V", [])
        if not vals:
            continue
        v = sum(vals) / len(vals)
        daily_sums[day] = daily_sums.get(day, 0.0) + v
        daily_counts[day] = daily_counts.get(day, 0) + 1

    daily_means = {d: daily_sums[d] / daily_counts[d] for d in daily_sums}
    complete_days = sorted(daily_means.keys())
    if not complete_days:
        raise RuntimeError("Could not compute any daily mean from PlotRecs")

    # Drop today (likely incomplete) unless it's the only day available.
    today_str = now.strftime("%Y-%m-%d")
    usable = [d for d in complete_days if d != today_str] or complete_days
    last_day = usable[-1]
    return last_day, round(daily_means[last_day], 3)


def predict(model, weather, anchor_date_str, anchor_level, horizon=7):
    a = model["level_lag1_coef"]
    b = model["precip_lag_coefs"]  # index k = precip_lag_k, k=0..len(b)-1
    c = model["intercept"]
    maxlag = len(b)

    anchor_date = datetime.strptime(anchor_date_str, "%Y-%m-%d").date()

    def rain_on(date_obj):
        key = date_obj.strftime("%Y-%m-%d")
        entry = weather.get(key)
        if entry is None or entry.get("precip_mm") is None:
            return 0.0  # missing data treated conservatively as no rain
        return entry["precip_mm"]

    level = anchor_level
    results = []
    for step in range(1, horizon + 1):
        target_date = anchor_date + timedelta(days=step)
        rain_terms = 0.0
        for k in range(maxlag):
            d = target_date - timedelta(days=k)
            rain_terms += b[k] * rain_on(d)
        level = a * level + rain_terms + c

        w = weather.get(target_date.strftime("%Y-%m-%d"), {})
        wc = w.get("weather_code")
        results.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_level_m": round(level, 3),
            "temp_max_c": w.get("temp_max_c"),
            "temp_min_c": w.get("temp_min_c"),
            "precip_mm": w.get("precip_mm"),
            "precip_probability_max": w.get("precip_prob_max"),
            "weather_code": wc,
            "weather_desc": WEATHER_CODE_DESC.get(wc, None),
        })
    return results


def append_data_log(date_str, level_m, precip_mm):
    file_exists = os.path.exists(LOG_OUT)
    existing_dates = set()
    if file_exists:
        with open(LOG_OUT, newline="") as f:
            for row in csv.DictReader(f):
                existing_dates.add(row["date"])
    if date_str in existing_dates:
        return  # already logged
    with open(LOG_OUT, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "level_m", "precip_mm"])
        writer.writerow([date_str, level_m, precip_mm])


def main():
    with open(MODEL_PATH) as f:
        model = json.load(f)

    try:
        weather = fetch_weather()
    except Exception as e:
        print(f"ERROR fetching weather: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        anchor_date, anchor_level = fetch_latest_water_level()
    except Exception as e:
        print(f"ERROR fetching water level: {e}", file=sys.stderr)
        sys.exit(1)

    forecast_days = predict(model, weather, anchor_date, anchor_level, horizon=7)

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "station": STATION_NAME,
        "station_lat": STATION_LAT,
        "station_lon": STATION_LON,
        "last_observed": {"date": anchor_date, "level_m": anchor_level},
        "days": forecast_days,
        "model_holdout_mae_cm": model.get("recursive_backtest_mae_cm"),
    }

    with open(FORECAST_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {FORECAST_OUT}")

    anchor_precip = weather.get(anchor_date, {}).get("precip_mm")
    append_data_log(anchor_date, anchor_level, anchor_precip)
    print(f"Updated {LOG_OUT}")


if __name__ == "__main__":
    main()
