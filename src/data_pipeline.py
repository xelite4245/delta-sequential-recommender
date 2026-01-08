"""
Handles all data preprocessing and feature engineering:
- Clean and merge datasets
- Compute rolling averages, volume, effort
- Prepare features for ML models
- Optional: synthetic data generation



Baseline datasesets:
- strong_4krows_baseline_data.csv
- strong_721rows_baseline_data.csv

Columns for 4k baseline dataset:
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,Notes,Workout Notes,RPE

Columns for 721 rows baseline dataset:
Same, except missing RPE and Set Order

To compensate for missing RPE and Set Order in the 721 rows dataset, we will:
- Impute RPE with the average RPE from the 4k dataset for the same Exercise Name
- Assign Set Order based on the order of appearance within each Workout Name
    To make sure Set Order is consistent, we will sort by Date within each Workout Name before assigning Set Order

"""

from pathlib import Path
import pandas as pd
import numpy as np

from .utils import (
    standardize_cols,
    compute_rpe_lookup,
    impute_rpe,
    assign_set_order,
    normalize_exercise_name,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_DIR = DATA_DIR / "baseline"
USER_DATA_DIR = DATA_DIR / "user_data"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Loading data function

def load_all_baseline_data():
    """Return Paths to baseline datasets (4k and 721)."""
    df_4k = BASELINE_DIR / "strong_4krows_baseline_data.csv"
    df_721 = BASELINE_DIR / "strong_721rows_baseline_data.csv"
    return df_4k, df_721


def read_baseline_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read baseline CSVs into DataFrames (raw, unprocessed)."""
    p4k, p721 = load_all_baseline_data()
    df4k = pd.read_csv(p4k)
    df721 = pd.read_csv(p721)
    return df4k, df721


# Cleaning and preprocessing functions

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a raw export dataset.

    Steps:
    - Rename columns to standard names
    - Parse dates
    - Coerce numeric columns
    - Trim string columns
    - Add normalized exercise name helper column
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = standardize_cols(df)

    # Ensure mandatory columns are present after rename
    required = [
        "date",
        "workout_name",
        "duration",
        "exercise_name",
        "weight",
        "reps",
        "distance",
        "seconds",
        "notes",
        "workout_notes",
    ]
    missing = [c for c in required if c not in df.columns]
    # It's okay if 'rpe' or 'set_order' are missing; they will be imputed/assigned
    if missing:
        # Keep only the columns that exist; caller can decide how to proceed
        pass

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Coerce numeric columns
    for col in ["duration", "weight", "reps", "distance", "seconds", "rpe", "set_order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Trim string columns
    for col in ["workout_name", "exercise_name", "notes", "workout_notes"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Normalized exercise name to support joins/lookups
    if "exercise_name" in df.columns:
        df["exercise_normalized"] = df["exercise_name"].apply(normalize_exercise_name)

    # Basic per-set features
    if {"weight", "reps"}.issubset(df.columns):
        df["set_volume"] = (df["weight"].fillna(0) * df["reps"].fillna(0)).astype(float)
    if {"weight", "rpe"}.issubset(df.columns):
        df["effective_load"] = (df["weight"].fillna(0) * (df["rpe"].fillna(0) / 10.0)).astype(float)

    return df


def preprocess_baseline_data(write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end preprocessing for baseline datasets.

    - Load 4k and 721 datasets
    - Clean/standardize both
    - Compute RPE lookup from 4k and impute into 721
    - Assign set order for 721 if missing
    - Recompute features and return processed frames
    - Optionally write processed CSVs into data/processed
    """
    df4k_raw, df721_raw = read_baseline_dataframes()

    df4k = clean_and_standardize_data(df4k_raw)
    df721 = clean_and_standardize_data(df721_raw)

    # Build RPE lookup from 4k and apply to 721 if rpe missing
    if "rpe" in df4k.columns and not df4k["rpe"].dropna().empty:
        rpe_lookup = compute_rpe_lookup(df4k)
        df721 = impute_rpe(df721, rpe_lookup)

    # Assign set order if missing or all NaN
    if ("set_order" not in df721.columns) or df721["set_order"].isna().all():
        df721 = assign_set_order(df721)

    # Recompute helper features post-imputation/assignment
    df721 = clean_and_standardize_data(df721)

    # Ensure 4k has set_order/effective features as well (in case missing)
    if ("set_order" not in df4k.columns) or df4k["set_order"].isna().all():
        df4k = assign_set_order(df4k)
        df4k = clean_and_standardize_data(df4k)

    # Combine for convenience
    common_cols = sorted(set(df4k.columns).intersection(set(df721.columns)))
    df_all = pd.concat([df4k[common_cols], df721[common_cols]], ignore_index=True)

    if write_outputs:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_DIR / "baseline_4k_processed.csv").write_text("", encoding="utf-8")
        (PROCESSED_DATA_DIR / "baseline_721_processed.csv").write_text("", encoding="utf-8")
        # Use pandas to_csv to actually write the frames
        df4k.to_csv(PROCESSED_DATA_DIR / "baseline_4k_processed.csv", index=False)
        df721.to_csv(PROCESSED_DATA_DIR / "baseline_721_processed.csv", index=False)
        df_all.to_csv(PROCESSED_DATA_DIR / "baseline_all_processed.csv", index=False)

    return df4k, df721, df_all


def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute simple dataset stats for sanity checks.

    Returns keys: rows, workouts, exercises, dates, avg_sets_per_session
    """
    if df is None or len(df) == 0:
        return {"rows": 0, "workouts": 0, "exercises": 0, "dates": 0, "avg_sets_per_session": 0.0}

    rows = len(df)
    workouts = df["workout_name"].nunique() if "workout_name" in df.columns else None
    exercises = df["exercise_name"].nunique() if "exercise_name" in df.columns else None
    dates = df["date"].dt.date.nunique() if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]) else None

    if {"workout_name", "date"}.issubset(df.columns) and pd.api.types.is_datetime64_any_dtype(df["date"]):
        sets_per_session = df.groupby(["workout_name", "date"]).size()
        avg_sets_per_session = float(sets_per_session.mean()) if not sets_per_session.empty else 0.0
    else:
        avg_sets_per_session = None

    return {
        "rows": rows,
        "workouts": workouts,
        "exercises": exercises,
        "dates": dates,
        "avg_sets_per_session": avg_sets_per_session,
    }