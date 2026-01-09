"""
Train squat model on full leg dataset, save to users/Ayfs/trained_models/squat_model.pkl,
then evaluate on baseline User2 squat data.
"""

import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.compound_models import SquatProgressionModel, add_periodization_features


BASE_DIR = Path(__file__).parent.parent
LEG_CSV = BASE_DIR / "data/processed/PPL_data/leg_workouts.csv"
USER2_CSV = BASE_DIR / "data/baseline/User2_legs_squat_data.csv"
MODEL_OUT = BASE_DIR / "users/Ayfs/trained_models/squat_model.pkl"


def load_and_prepare(df: pd.DataFrame, target_col: str = "load_delta"):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    if "load_delta" not in df.columns:
        df[target_col] = df["weight"].diff().shift(-1)
    df = df.dropna(subset=[target_col])
    df = add_periodization_features(df)
    X = df.drop(columns=[target_col, "date", "exercise_normalized", "workout_name", "exercise_name"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    return X, y, df


def main():
    # Train on full leg dataset
    leg = pd.read_csv(LEG_CSV)
    leg["date"] = pd.to_datetime(leg["date"])
    # filter to top sets and squat
    if "set_order" in leg.columns:
        leg = leg[leg["set_order"] == 1]
    if "exercise_normalized" in leg.columns:
        leg = leg[leg["exercise_normalized"] == "squat"]
    X_train, y_train, df_train = load_and_prepare(leg)

    print(f"Training rows: {len(df_train)}; Features: {X_train.shape[1]}")
    model = SquatProgressionModel()
    model.fit(X_train, y_train)

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained squat model to {MODEL_OUT}")

    # Load User2 data
    user2 = pd.read_csv(USER2_CSV)
    # Normalize column names if needed
    rename_map = {
        "Exercise": "exercise_name",
        "Category": "category",
        "Weight_lbs": "weight",
        "Reps": "reps",
        "Volume": "set_volume",
    }
    user2 = user2.rename(columns=rename_map)
    # Add minimal required columns
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

    if len(user2) < 5:
        print("oh no: too few rows to compute load_delta reliably")
        return

    X_test, y_test, df_test = load_and_prepare(user2)

    if len(df_test) < 5:
        print("oh no: too few rows after load_delta")
        return

    # Predict
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nEVAL ON USER2 SQUAT DATA:")
    print(f"  MAE:  {mae:7.2f}")
    print(f"  RMSE: {rmse:7.2f}")
    print(f"  R²:   {r2:7.4f}")

    if r2 > 0:
        print("hooray")
    else:
        print("oh no")


if __name__ == "__main__":
    main()
