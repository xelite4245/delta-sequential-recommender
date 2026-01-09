import pandas as pd
from pathlib import Path

# Paths
repo = Path(__file__).resolve().parents[1]
src_csv = repo / "data" / "baseline" / "User2_legs_squat_data.csv"
out_dir = repo / "data" / "user_inputs"
out_dir.mkdir(parents=True, exist_ok=True)

# Read and map columns
# Expected in source: Exercise, Category, Weight_lbs, Reps, Volume
# CLI requires at minimum: weight, reps

df = pd.read_csv(src_csv)
if not {"Weight_lbs", "Reps"}.issubset(df.columns):
    raise SystemExit("Source CSV missing required columns 'Weight_lbs' and 'Reps'.")

use = df[["Weight_lbs", "Reps"]].rename(columns={"Weight_lbs": "weight", "Reps": "reps"}).copy()
use = use.dropna(subset=["weight", "reps"]).reset_index(drop=True)

if len(use) < 2:
    raise SystemExit("Not enough rows to split into history and future (need >=2).")

# Split: all but last -> history, last -> future
history = use.iloc[:-1].copy()
future = use.iloc[[-1]].copy()

hist_path = out_dir / "user2_squat_history.csv"
fut_path = out_dir / "user2_squat_future.csv"

history.to_csv(hist_path, index=False)
future.to_csv(fut_path, index=False)

print(f"Wrote history: {hist_path} ({len(history)} rows)")
print(f"Wrote future:  {fut_path} (1 row)")
