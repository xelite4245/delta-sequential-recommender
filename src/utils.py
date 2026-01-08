"""
Utility functions used across the project:
- Plotting helpers
- Date/time calculations
- Data validation
- Miscellaneous helper functions
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


from datetime import datetime, timedelta


COLUMN_RENAME_MAP = {
    "Date": "date",
    "Workout Name": "workout_name",
    "Duration": "duration",
    "Exercise Name": "exercise_name",
    "Set Order": "set_order",
    "Weight": "weight",
    "Reps": "reps",
    "Distance": "distance",
    "Seconds": "seconds",
    "Notes": "notes",
    "Workout Notes": "workout_notes",
    "RPE": "rpe",
}


def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df = df.copy()
    return df.rename(columns=COLUMN_RENAME_MAP)


def normalize_exercise_name(name: str) -> str:
    """Normalize exercise names by removing special characters and converting to lowercase."""
    name = name.lower().strip()
    name = re.sub(r"\s*\(.*?\)\s*", "", name) # Remove text within parentheses
    name = re.sub(r"\s+", " ", name) # Replace multiple spaces with single space
    return name


def compute_rpe_lookup(df_4k: pd.DataFrame) -> dict:
    """
    Compute average RPE per exercise from the 4k dataset.
    Uses normalized exercise names for consistent matching.
    """
    df_temp = df_4k.dropna(subset=["rpe"]).copy()
    df_temp["exercise_normalized"] = df_temp["exercise_name"].apply(normalize_exercise_name)
    
    lookup = (
        df_temp
        .groupby("exercise_normalized")["rpe"]
        .mean()
        .to_dict()
    )

    return lookup


def impute_rpe(df: pd.DataFrame, rpe_lookup: dict) -> pd.DataFrame:
    """
    Impute missing RPE values using exercise-level averages.
    Uses normalized exercise names to match with the lookup dict.
    """
    df = df.copy()

    def _impute(row):
        if not pd.isna(row.get("rpe")):
            return row["rpe"]
        exercise_normalized = normalize_exercise_name(row["exercise_name"])
        return rpe_lookup.get(exercise_normalized, np.nan)

    df["rpe"] = df.apply(_impute, axis=1)
    return df


def assign_set_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Set Order for datasets missing it.

    Logic:
    - Group by workout name
    - Sort by date
    - Assign order of appearance
    """
    df = df.copy()

    df = df.sort_values(["workout_name", "date"])

    df["set_order"] = (
        df
        .groupby(["workout_name", "date"])
        .cumcount()
        .add(1)
    )

    return df