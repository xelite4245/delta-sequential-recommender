"""
Test script for compound progression models and exercise mapping.

Tests:
1. Load and train all 4 compound models on PPL data
2. Validate predictions on held-out sets
3. Test exercise mapping (child exercises -> parent compound)
4. Compute metrics per compound and per category
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.compound_models import (
    SquatProgressionModel,
    BenchPressProgressionModel,
    LatPulldownProgressionModel,
    SeatedRowProgressionModel,
    add_periodization_features,
)
from utils.exercise_mapping import (
    EXERCISE_MAPPING,
    SCALING_FACTORS,
    get_parent_compound,
    get_scaling_factor,
    predict_exercise_delta,
    print_exercise_mapping,
)


def train_and_eval_compound(
    csv_path: Path,
    model_class,
    compound_name: str,
    train_fraction: float = 0.8,
) -> dict:
    """
    Train a single compound model and evaluate on held-out set.
    
    Returns:
        Dict with metrics: mae_train, rmse_train, r2_train, mae_val, rmse_val, r2_val
    """
    print(f"\n{'='*70}")
    print(f"{compound_name.upper()}")
    print(f"{'='*70}")
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return {}
    
    # Load and prepare
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    
    # Filter to top sets if needed
    if 'set_order' in df.columns:
        df = df[df['set_order'] == 1].copy()
        print(f"Filtered to {len(df)} top sets")
    
    if len(df) < 50:
        print(f"Too few rows ({len(df)}). Skipping.")
        return {}
    
    # Sort by date and compute load_delta
    df = df.sort_values('date').reset_index(drop=True)
    df['load_delta'] = df['weight'].diff().shift(-1)
    df = df.dropna(subset=['load_delta'])
    
    print(f"After load_delta computation: {len(df)} rows")
    
    # Split into train/val
    split_idx = int(len(df) * train_fraction)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    
    print(f"Train: {len(df_train)}, Validation: {len(df_val)}")
    
    # Prepare features
    X_train, y_train = model_class.prepare_compound_data(
        df_train,
        target_col='load_delta',
        drop_cols=['date', 'exercise_normalized', 'workout_name']
    )
    
    X_val, y_val = model_class.prepare_compound_data(
        df_val,
        target_col='load_delta',
        drop_cols=['date', 'exercise_normalized', 'workout_name']
    )
    
    # Align columns (val might be missing some columns from train)
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for col in missing_cols:
        X_val[col] = 0
    X_val = X_val[X_train.columns]
    
    print(f"Feature space: {X_train.shape[1]} features")
    
    # Train
    model = model_class()
    model.fit(X_train, y_train)
    
    # Evaluate on train set
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_val = r2_score(y_val, y_pred_val)
    
    # Print results
    print(f"\nTrain   - MAE: {mae_train:7.2f}, RMSE: {rmse_train:7.2f}, R²: {r2_train:7.4f}")
    print(f"Val     - MAE: {mae_val:7.2f}, RMSE: {rmse_val:7.2f}, R²: {r2_val:7.4f}")
    
    return {
        "compound": compound_name,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "mae_val": mae_val,
        "rmse_val": rmse_val,
        "r2_val": r2_val,
    }


def test_exercise_mapping():
    """Validate exercise mapping and scaling factors."""
    print(f"\n{'='*70}")
    print("EXERCISE MAPPING AND SCALING FACTORS")
    print(f"{'='*70}\n")
    
    print_exercise_mapping()
    
    # Test a few lookups
    print(f"\n{'='*70}")
    print("EXAMPLE LOOKUPS")
    print(f"{'='*70}\n")
    
    test_exercises = [
        "squat",
        "leg_press",
        "bench_press",
        "incline_db_press",
        "lat_pulldown",
        "pull_ups",
        "seated_row",
        "dumbbell_row",
    ]
    
    for exercise in test_exercises:
        parent = get_parent_compound(exercise)
        scaling = get_scaling_factor(exercise)
        print(f"{exercise:25} -> {parent:20} (scaling: {scaling:.2f})")


def test_prediction_scaling():
    """Test how predictions are scaled for different exercises."""
    print(f"\n{'='*70}")
    print("PREDICTION SCALING EXAMPLE")
    print(f"{'='*70}\n")
    
    # Example: squat model predicts +5 lbs
    parent_pred = 5.0
    
    exercises = ["squat", "leg_press", "hack_squat", "lunges", "leg_extensions"]
    
    print(f"If Squat model predicts: {parent_pred:.2f} lbs load delta")
    print(f"\nPredictions for related exercises:")
    for exercise in exercises:
        scaled = predict_exercise_delta(parent_pred, exercise)
        print(f"  {exercise:20} -> {scaled:7.2f} lbs")


if __name__ == "__main__":
    ppl_dir = Path(__file__).parent.parent / "data" / "processed" / "PPL_data"
    
    print(f"\n{'#'*70}")
    print(f"# COMPOUND PROGRESSION MODELS - TEST SUITE")
    print(f"{'#'*70}")
    
    # Test exercise mapping
    test_exercise_mapping()
    test_prediction_scaling()
    
    # Train and evaluate compounds
    print(f"\n{'#'*70}")
    print(f"# TRAINING COMPOUND MODELS")
    print(f"{'#'*70}")
    
    compounds = [
        (ppl_dir / "leg_workouts.csv", SquatProgressionModel, "Squat"),
        (ppl_dir / "push_workouts.csv", BenchPressProgressionModel, "Bench Press"),
        (ppl_dir / "pull_workouts.csv", LatPulldownProgressionModel, "Lat Pulldown"),
        (ppl_dir / "pull_workouts.csv", SeatedRowProgressionModel, "Seated Row"),
    ]
    
    results = []
    for csv_path, model_class, compound_name in compounds:
        result = train_and_eval_compound(csv_path, model_class, compound_name)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}\n")
    
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"AVERAGE METRICS (all compounds)")
        print(f"{'='*70}")
        print(f"Train - MAE: {summary_df['mae_train'].mean():.2f}, RMSE: {summary_df['rmse_train'].mean():.2f}, R²: {summary_df['r2_train'].mean():.4f}")
        print(f"Val   - MAE: {summary_df['mae_val'].mean():.2f}, RMSE: {summary_df['rmse_val'].mean():.2f}, R²: {summary_df['r2_val'].mean():.4f}")
    else:
        print("No results to summarize. Check file paths.")
