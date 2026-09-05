# Skjern Å water level forecast

Predicts the water level at **station 251519, Skjern Å, Vardevej** (near Albæk bro,
West Jutland, Denmark) 7 days ahead, based on a rainfall-to-water-level model
trained on historic data.

## How it works

- **`model_coefficients.json`** — a linear ARX (autoregressive + distributed-lag
  rainfall) model: `level(t) = a * level(t-1) + Σ b_k * rain(t-k) + c` for
  `k = 0..14` days. Fitted on ~20 months of daily water level readings and
  matching Open-Meteo rainfall history for the same coordinates.
  - Peak rainfall response occurs ~3 days after a rain event.
  - Recession half-life ≈ 50 days (slow, groundwater-fed lowland catchment —
    confirmed against a 23-year record from an upstream station, Gjaldbæk bro).
  - Backtested accuracy (assuming a perfect rain forecast): ~1.3cm MAE at
    1 day ahead, ~4.9cm MAE at 7 days ahead.
- **`pipeline.py`** — runs daily via GitHub Actions
  (`.github/workflows/daily.yml`). It:
  1. Fetches the last ~20 days of actual rainfall + next 7 days of forecast
     rainfall/weather from [Open-Meteo](https://open-meteo.com/) (free, no key).
  2. Fetches the latest observed water level from Vandportalen's public plot
     data endpoint for this station.
  3. Runs the model recursively 7 days forward.
  4. Writes **`forecast.json`** (consumed by the companion web app) and
     appends the newly observed day to **`data_log.csv`** — building up a
     growing, clean training set for future model refits.
- No API keys or secrets are needed — both data sources are public and
  unauthenticated.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Daily fetch + predict script |
| `model_coefficients.json` | Trained model coefficients |
| `forecast.json` | Latest output (overwritten daily) |
| `data_log.csv` | Growing log of observed (date, level, rain) — for future retraining |
| `.github/workflows/daily.yml` | Schedules `pipeline.py` once a day |

## Re-training the model

As `data_log.csv` grows, the model can be refit with more data (especially
useful once it's seen a full seasonal cycle or two, or an extreme event).
The original training/analysis scripts used `sklearn.linear_model.RidgeCV`
on lagged rainfall features — see the project history for the exact
methodology if you want to reproduce or extend it.

## Manually running the pipeline

From the **Actions** tab on GitHub, select "Daily water level forecast" →
"Run workflow" to trigger it on demand instead of waiting for the schedule.
