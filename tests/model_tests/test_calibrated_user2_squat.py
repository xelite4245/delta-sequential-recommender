"""
Evaluate per-user affine calibration on User2 squat data.

Steps:
1) Train SquatProgressionModel on full leg_workouts (squat, top sets).
2) Predict on a small calibration window of User2 data; fit affine (a, b).
3) Evaluate on the remaining User2 rows before vs after calibration.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.compound_models import SquatProgressionModel, add_periodization_features
from utils.user_personalization import UserPersonalization


BASE_DIR = Path(__file__).parent.parent.parent
LEG_CSV = BASE_DIR / "data/processed/PPL_data/leg_workouts.csv"
USER2_CSV = BASE_DIR / "data/baseline/User2_legs_squat_data.csv"


def load_and_prepare(df: pd.DataFrame, target_col: str = "load_delta"):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    if target_col not in df.columns:
        df[target_col] = df["weight"].diff().shift(-1)
    df = df.dropna(subset=[target_col])
    df = add_periodization_features(df)
    X = df.drop(columns=[target_col, "date", "exercise_normalized", "workout_name", "exercise_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    return X, y, df


def main():
    # Train global squat model
    leg = pd.read_csv(LEG_CSV)
    leg["date"] = pd.to_datetime(leg["date"])
    if "set_order" in leg.columns:
        leg = leg[leg["set_order"] == 1]
    if "exercise_normalized" in leg.columns:
        leg = leg[leg["exercise_normalized"] == "squat"]
    X_train, y_train, df_train = load_and_prepare(leg)
    model = SquatProgressionModel()
    model.fit(X_train, y_train)
    print(f"Global squat training rows: {len(df_train)}")

    # Load User2 data
    user2 = pd.read_csv(USER2_CSV)
    rename_map = {
        "Exercise": "exercise_name",
        "Category": "category",
        "Weight_lbs": "weight",
        "Reps": "reps",
        "Volume": "set_volume",
    }
    user2 = user2.rename(columns=rename_map)
    user2["exercise_normalized"] = "squat"
    user2["date"] = pd.date_range(start="2024-01-01", periods=len(user2), freq="2D")
    user2["set_order"] = 1
    user2["rpe"] = 8.0
    user2["seconds"] = 0
    user2["distance"] = 0.0
    user2["workout_name"] = "USER2_LEG"
    user2["workout_notes"] = ""
    user2["cleaned_workout_name"] = "leg"
    user2["effective_load"] = user2["weight"] * (user2["rpe"] / 10.0)

    X_user2, y_user2, df_user2 = load_and_prepare(user2)
    if len(df_user2) < 12:
        print("oh no: not enough rows for calibration+eval")
        return

    # Split calibration and evaluation
    calib_n = 12
    X_calib, y_calib = X_user2.iloc[:calib_n], y_user2.iloc[:calib_n]
    X_eval, y_eval = X_user2.iloc[calib_n:], y_user2.iloc[calib_n:]

    # Predict on calibration window
    y_pred_calib = model.predict(X_calib)

    # Fit affine per-user calibration
    up = UserPersonalization(user_id="user2")
    ab = up.calibrate_affine(
        compound="squat",
        y_true=y_calib.values,
        y_pred=y_pred_calib,
        min_samples=8,
        gain_bounds=(0.6, 1.4),
    )
    if ab is None:
        print("oh no: calibration skipped (too few samples)")
        return
    a, b = ab
    print(f"Calibration fitted: a={a:.3f}, b={b:.3f}")

    # Evaluate before/after calibration on remaining rows
    y_pred_eval = model.predict(X_eval)
    y_pred_eval_adj = (y_pred_eval * up.scaling_factors["squat"]) + up.baseline_offsets["squat"]

    def metrics(y_true, y_hat):
        mae = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2 = r2_score(y_true, y_hat)
        return mae, rmse, r2

    mae_raw, rmse_raw, r2_raw = metrics(y_eval, y_pred_eval)
    mae_adj, rmse_adj, r2_adj = metrics(y_eval, y_pred_eval_adj)

    print("\nUser2 eval (real-only):")
    print(f"  BEFORE: MAE={mae_raw:7.2f}, RMSE={rmse_raw:7.2f}, R2={r2_raw:7.4f}")
    print(f"  AFTER : MAE={mae_adj:7.2f}, RMSE={rmse_adj:7.2f}, R2={r2_adj:7.4f}")

    if r2_adj > r2_raw:
        print("hooray (improved)")
    else:
        print("oh no (no improvement)")


if __name__ == "__main__":
    main()
