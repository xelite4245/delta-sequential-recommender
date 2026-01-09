import pandas as pd
import numpy as np
from pathlib import Path

# Paths
repo = Path(__file__).resolve().parents[1]
src_csv = repo / "data" / "baseline" / "User2_legs_squat_data.csv"
out_dir = repo / "data" / "user_inputs"
out_dir.mkdir(parents=True, exist_ok=True)

# Read and map columns
df = pd.read_csv(src_csv)
use = df[["Weight_lbs", "Reps"]].rename(columns={"Weight_lbs": "weight", "Reps": "reps"}).copy()
use = use.dropna(subset=["weight", "reps"]).reset_index(drop=True)

# Add synthetic but realistic columns
use["date"] = pd.date_range(start="2024-01-01", periods=len(use), freq="3D")
use["exercise_name"] = "Barbell Squat"
use["exercise_normalized"] = "squat"
use["workout_name"] = "Leg Day"
use["workout_name_clean"] = "legs"
use["rpe"] = 7.0 + (use.index % 3) * 0.5
use["set_order"] = 1  # All are top sets
use["set_volume"] = use["weight"] * use["reps"]
use["effective_load"] = use["weight"] * (use["rpe"] / 10.0)

# Compute max weight seen so far (per-user metric)
use["max_weight_so_far"] = use["weight"].expanding().max()

# Percent of max (normalized strength)
use["percent_of_max"] = (use["weight"] / use["max_weight_so_far"] * 100).round(1)

# Periodization features
use["cycle_number"] = (use.index // 12) + 1  # ~12 sessions per cycle
use["weeks_in_cycle"] = ((use.index % 12) // 3) + 1  # ~3 weeks per block
use["cycle_weight_trend"] = use.groupby("cycle_number")["weight"].transform(
    lambda x: x - x.iloc[0]
)  # Weight change within cycle

# Deload detection (arbitrary: reps < 3 or weight drop > 10)
use["is_deload"] = (
    ((use["reps"] < 3) | (use["weight"].diff().fillna(0) < -10)).astype(int)
)

# Compute load_delta (weight change to next session)
use["load_delta"] = use["weight"].diff().shift(-1)

if len(use) < 2:
    raise SystemExit("Not enough rows to split.")

# Split: all but last -> history, last -> future
history = use.iloc[:-1].copy()
future = use.iloc[[-1]].copy()
future = future.drop(columns=["load_delta"])  # future doesn't have load_delta

hist_path = out_dir / "user2_squat_history.csv"
fut_path = out_dir / "user2_squat_future.csv"

history.to_csv(hist_path, index=False)
future.to_csv(fut_path, index=False)

print(f"✓ Wrote history: {hist_path}")
print(f"  Rows: {len(history)}, Cols: {len(history.columns)}")
print(f"  Non-null load_delta: {history['load_delta'].notna().sum()}")
print(f"✓ Wrote future:  {fut_path}")
print(f"  Rows: {len(future)}, Cols: {len(future.columns)}")
print(f"\nFeatures in CSVs:")
for col in history.columns:
    print(f"  - {col}")
