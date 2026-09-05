import json
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- rainfall (full range) ---
with open("rainfall_history_full.json") as f:
    rain_raw = json.load(f)
rain = pd.DataFrame({
    "date": pd.to_datetime(rain_raw["daily"]["time"]),
    "precip_mm": rain_raw["daily"]["precipitation_sum"],
})

# --- upstream water level (Gjaldbæk bro, 2000-2022) ---
up = pd.read_csv("waterlevel_upstream_clean.csv", parse_dates=["date"])
up = up[["date", "level_m"]]

merged = pd.merge(up, rain, on="date", how="inner").sort_values("date").reset_index(drop=True)
print("Merged upstream range:", merged["date"].min(), "to", merged["date"].max(), "n=", len(merged))

MAXLAG = 15
d2 = merged.copy()
d2["level_lag1"] = d2["level_m"].shift(1)
d2["date_prev"] = d2["date"].shift(1)
d2["consecutive"] = (d2["date"] - d2["date_prev"]) == pd.Timedelta(days=1)
for k in range(MAXLAG):
    d2[f"precip_lag{k}"] = d2["precip_mm"].shift(k)

feature_cols = ["level_lag1"] + [f"precip_lag{k}" for k in range(MAXLAG)]
model_df = d2[d2["consecutive"]].dropna(subset=["level_m"] + feature_cols).copy().reset_index(drop=True)
print(f"Training rows: {len(model_df)}  ({model_df['date'].min().date()} to {model_df['date'].max().date()})")

X = model_df[feature_cols].values
y = model_df["level_m"].values
n = len(model_df)
split = int(n * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

ridge = RidgeCV(alphas=np.logspace(-3, 3, 50))
ridge.fit(X_train, y_train)
pred_test = ridge.predict(X_test)
print(f"\nHoldout (last {n-split} days, {model_df['date'].iloc[split].date()} to {model_df['date'].iloc[-1].date()}):")
print(f"  R^2  = {r2_score(y_test, pred_test):.4f}")
print(f"  MAE  = {mean_absolute_error(y_test, pred_test)*100:.2f} cm")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, pred_test))*100:.2f} cm")

ridge_final = RidgeCV(alphas=np.logspace(-3, 3, 50))
ridge_final.fit(X, y)
a = ridge_final.coef_[0]
b = ridge_final.coef_[1:]
print(f"\na (level_lag1 coef) = {a:.5f}  -> recession half-life = {np.log(0.5)/np.log(a):.1f} days")
print("(compare to Vardevej: a=0.98616 -> half-life 49.7 days)")

def simulate_impulse(pulse_mm, horizon=45):
    rain_arr = np.zeros(horizon + MAXLAG)
    rain_arr[MAXLAG - 1] = pulse_mm
    level_dev = np.zeros(horizon)
    prev = 0.0
    for t in range(horizon):
        idx = t + MAXLAG - 1
        rain_terms = sum(b[k] * rain_arr[idx - k] for k in range(MAXLAG))
        cur = a * prev + rain_terms
        level_dev[t] = cur
        prev = cur
    return level_dev

for pulse in [10, 20]:
    resp = simulate_impulse(pulse)
    peak_day = int(np.argmax(resp))
    print(f"\n{pulse}mm pulse -> peak {resp[peak_day]*100:.2f} cm at day {peak_day+1}; "
          f"+{resp[13]*100:.2f} cm @14d; +{resp[29]*100:.2f} cm @30d "
          f"(Vardevej equivalent: peak ~{'9.17' if pulse==10 else '18.33'} cm @ day 3)")

# Seasonal check: does winter vs summer behave differently? Quick decomposition by month
merged["month"] = merged["date"].dt.month
monthly_mean = merged.groupby("month")["level_m"].mean()
print("\nMonthly mean level (m) at Gjaldbæk bro, 2000-2022 (seasonality check):")
print(monthly_mean.round(3))

# How many extreme events (>90th percentile daily rain) are in this long record vs short Vardevej record?
p90 = merged["precip_mm"].quantile(0.90)
extreme_days = (merged["precip_mm"] > p90).sum()
max_rain_day = merged["precip_mm"].max()
print(f"\n90th percentile daily rain: {p90:.1f}mm; max single-day rain in 23yr record: {max_rain_day:.1f}mm")
print(f"Level range across full record: {merged['level_m'].min():.2f}m to {merged['level_m'].max():.2f}m "
      f"(span {merged['level_m'].max()-merged['level_m'].min():.2f}m)")
