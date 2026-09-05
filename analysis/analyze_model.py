import pandas as pd
import numpy as np
import json
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("merged_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

MAXLAG = 15
d2 = df.copy()
d2["level_lag1"] = d2["level_m"].shift(1)
d2["date_prev"] = d2["date"].shift(1)
d2["consecutive"] = (d2["date"] - d2["date_prev"]) == pd.Timedelta(days=1)
for k in range(MAXLAG):
    d2[f"precip_lag{k}"] = d2["precip_mm"].shift(k)

feature_cols = ["level_lag1"] + [f"precip_lag{k}" for k in range(MAXLAG)]
model_df = d2[d2["consecutive"]].dropna(subset=["level_m"] + feature_cols).copy().reset_index(drop=True)

X = model_df[feature_cols].values
y = model_df["level_m"].values
n = len(model_df)
split = int(n * 0.8)

ridge_final = RidgeCV(alphas=np.logspace(-3, 3, 50))
ridge_final.fit(X, y)
a = ridge_final.coef_[0]
b = ridge_final.coef_[1:]
c = ridge_final.intercept_
print(f"a (level_lag1 coef) = {a:.5f}   ->  implied recession half-life = {np.log(0.5)/np.log(a):.1f} days")

# ---------- CORRECT impulse response via simulation ----------
# Simulate: baseline = long-run mean level (so we look at *deviation* dynamics),
# inject a single rain pulse of `pulse_mm` on day 0, zero rain otherwise, run recursion.
def simulate_impulse(pulse_mm, horizon=60):
    rain = np.zeros(horizon + MAXLAG)
    rain[MAXLAG - 1] = pulse_mm  # so at sim-day 0 (index MAXLAG-1), precip_lag0 = pulse
    level_dev = np.zeros(horizon)
    prev = 0.0
    for t in range(horizon):
        idx = t + MAXLAG - 1
        rain_terms = sum(b[k] * rain[idx - k] for k in range(MAXLAG))
        cur = a * prev + rain_terms  # deviation dynamics: no intercept (steady baseline = c/(1-a))
        level_dev[t] = cur
        prev = cur
    return level_dev

for pulse in [1, 10, 20, 30]:
    resp = simulate_impulse(pulse, horizon=45)
    peak_day = int(np.argmax(resp))
    print(f"\nSingle {pulse:>2d} mm rain-day pulse -> peak level rise = {resp[peak_day]*100:.2f} cm at day {peak_day+1}; "
          f"still +{resp[13]*100:.2f} cm after 14 days; +{resp[29]*100:.2f} cm after 30 days")

# ---------- Proper multi-day-ahead backtest (recursive forecast, using TRUE future rain) ----------
# This isolates model skill from weather-forecast skill: "if we knew the rain perfectly, how good
# is the water-level model itself at 1/3/7-day-ahead prediction?"
test_start = split
horizons = [1, 3, 5, 7]
errors = {h: [] for h in horizons}

precip_arr = model_df["precip_mm"].values
level_arr = model_df["level_m"].values

for i in range(test_start, n - max(horizons)):
    # need precip lags going back MAXLAG-1 before day i for the recursive steps; use model_df directly
    sim_level = level_arr[i]  # true starting level (this is "today", known)
    for step in range(1, max(horizons) + 1):
        t = i + step
        # rain terms use TRUE historical rain (stand-in for a perfect forecast)
        rain_terms = 0.0
        for k in range(MAXLAG):
            src_idx = t - k
            if src_idx < 0:
                continue
            rain_terms += b[k] * precip_arr[src_idx]
        sim_level = a * sim_level + rain_terms + c
        if step in horizons:
            errors[step].append(abs(sim_level - level_arr[t]))

print("\n=== Recursive multi-day-ahead backtest (assumes perfect rain forecast) ===")
for h in horizons:
    err = np.array(errors[h])
    print(f"{h}-day-ahead:  MAE = {err.mean()*100:.2f} cm   (n={len(err)})")

# Save enriched model file
model = {
    "station": "251519 Skjern Å, Vardevej",
    "station_lat": 55.9529,
    "station_lon": 8.5469,
    "trained_on_rows": int(n),
    "date_range": [str(model_df['date'].min().date()), str(model_df['date'].max().date())],
    "max_lag_days": MAXLAG,
    "ridge_alpha": float(ridge_final.alpha_),
    "intercept": float(c),
    "level_lag1_coef": float(a),
    "precip_lag_coefs": [float(x) for x in b],
    "recursive_backtest_mae_cm": {str(h): float(np.array(errors[h]).mean() * 100) for h in horizons},
}
with open("model_coefficients.json", "w") as f:
    json.dump(model, f, indent=2)
print("\nSaved model_coefficients.json (with recursive backtest results)")
