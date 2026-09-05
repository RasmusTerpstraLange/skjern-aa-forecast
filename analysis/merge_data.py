import json
import pandas as pd
import numpy as np

# --- Load rainfall history (Open-Meteo) ---
with open("rainfall_history.json") as f:
    rain_raw = json.load(f)

rain = pd.DataFrame({
    "date": pd.to_datetime(rain_raw["daily"]["time"]),
    "precip_mm": rain_raw["daily"]["precipitation_sum"],
    "temp_mean_c": rain_raw["daily"]["temperature_2m_mean"],
})

# --- Load water level history (Vandportalen / DMP), semicolon-delimited, header junk at top ---
with open("waterlevel_251519.csv", encoding="utf-8") as f:
    lines = f.readlines()

# find the header row (starts with "Dato")
header_idx = next(i for i, l in enumerate(lines) if l.startswith("Dato"))
data_lines = lines[header_idx:]

from io import StringIO
wl_raw = pd.read_csv(StringIO("".join(data_lines)), sep=";", decimal=".")
wl_raw.columns = ["date_str", "level_m", "qc"]
wl_raw = wl_raw.dropna(subset=["date_str"])
wl_raw["date"] = pd.to_datetime(wl_raw["date_str"], format="%d-%m-%Y")
wl_raw["level_m"] = pd.to_numeric(wl_raw["level_m"], errors="coerce")
wl = wl_raw[["date", "level_m"]].copy()

print("Rain range:", rain["date"].min(), "to", rain["date"].max(), "n=", len(rain))
print("Water level range:", wl["date"].min(), "to", wl["date"].max(), "n=", len(wl))
print("Water level missing days:", wl["level_m"].isna().sum())

# --- Merge on overlapping date range ---
merged = pd.merge(wl, rain, on="date", how="inner").sort_values("date").reset_index(drop=True)
print("\nMerged range:", merged["date"].min(), "to", merged["date"].max(), "n=", len(merged))
print("Merged missing water level:", merged["level_m"].isna().sum())

# Check for missing calendar days (gaps in the date sequence itself)
full_range = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
missing_dates = full_range.difference(merged["date"])
print("Missing calendar dates in merged range:", len(missing_dates))

merged.to_csv("merged_daily.csv", index=False)
print("\nSaved merged_daily.csv")
print(merged.head())
print(merged.tail())
