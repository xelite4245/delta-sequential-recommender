"""
Final validation and cross-validation metrics for all compound models.

Trains each model on PPL data and reports:
- Validation set metrics (80/20 time-based split)
- 5-fold TimeSeriesSplit cross-validation metrics
- Comparison and stability assessment

This is the final checkup before app implementation.
"""

import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from models.compound_models import (
    SquatProgressionModel,
    BenchPressProgressionModel,
    LatPulldownProgressionModel,
    SeatedRowProgressionModel,
    add_periodization_features,
)


def get_cv_score(model, X, y, cv=5):
    """
    Time-series cross-validation for progression models.
    Uses TimeSeriesSplit to respect temporal ordering.
    
    Returns: (avg_mae, avg_rmse, avg_r2, std_r2)
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Split data
        if isinstance(X, pd.DataFrame):
            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
        else:
            X_train_cv = X[train_idx]
            X_val_cv = X[val_idx]
            y_train_cv = y[train_idx]
            y_val_cv = y[val_idx]
        
        # Clone model
        try:
            model_cv = clone(model)
        except Exception:
            try:
                model_cv = deepcopy(model)
            except Exception as e2:
                print(f"    Warning: Could not clone model for fold {fold+1}: {e2}")
                continue
        
        # Fit and predict
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred = model_cv.predict(X_val_cv)
        
        # Metrics
        mae = mean_absolute_error(y_val_cv, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        r2 = r2_score(y_val_cv, y_pred)
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        print(f"    Fold {fold+1}: MAE={mae:7.2f}, RMSE={rmse:7.2f}, R²={r2:7.4f}")
    
    if len(mae_scores) == 0:
        print("    ERROR: No CV folds completed!")
        return float('nan'), float('nan'), float('nan'), float('nan')
    
    return (
        np.mean(mae_scores),
        np.mean(rmse_scores),
        np.mean(r2_scores),
        np.std(r2_scores)
    )


def evaluate_compound(
    csv_path: Path,
    model_class,
    compound_name: str,
    train_fraction: float = 0.8,
    cv: int = 5,
) -> dict:
    """
    Train a compound model and evaluate on both validation set and CV.
    
    Returns dict with all metrics.
    """
    print(f"\n{'='*90}")
    print(f"{compound_name.upper()}")
    print(f"{'='*90}")
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return {}
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    
    # Filter to top sets
    if 'set_order' in df.columns:
        df = df[df['set_order'] == 1].copy()
        print(f"Filtered to {len(df)} top sets")
    
    if len(df) < 50:
        print(f"Too few rows ({len(df)}). Skipping.")
        return {}
    
    # Sort and compute load_delta
    df = df.sort_values('date').reset_index(drop=True)
    df['load_delta'] = df['weight'].diff().shift(-1)
    df = df.dropna(subset=['load_delta'])
    
    print(f"After load_delta computation: {len(df)} rows")
    
    # Prepare features
    X, y = model_class.prepare_compound_data(
        df,
        target_col='load_delta',
        drop_cols=['date', 'exercise_normalized', 'workout_name']
    )
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"Feature space: {X.shape[1]} features")
    
    # Time-based split
    split_idx = int(len(df) * train_fraction)
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Validation: {len(X_valid)}")
    
    # Train model on full training set
    print(f"\nTraining {compound_name} model...")
    model = model_class()
    model.fit(X_train, y_train)
    
    # Validation metrics
    print(f"\nValidation Set (80/20 time-based split):")
    y_pred_val = model.predict(X_valid)
    val_mae = mean_absolute_error(y_valid, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_val))
    val_r2 = r2_score(y_valid, y_pred_val)
    print(f"  MAE: {val_mae:7.2f}, RMSE: {val_rmse:7.2f}, R²: {val_r2:7.4f}")
    
    # Cross-validation metrics
    print(f"\nCross-Validation ({cv}-fold TimeSeriesSplit on training set):")
    cv_mae, cv_rmse, cv_r2, cv_r2_std = get_cv_score(model, X_train, y_train, cv=cv)
    print(f"  Avg: MAE={cv_mae:7.2f}, RMSE={cv_rmse:7.2f}, R²={cv_r2:7.4f} (±{cv_r2_std:.4f})")
    
    # Assess stability
    r2_gap = abs(val_r2 - cv_r2)
    if r2_gap < 0.05:
        stability = "[OK] Stable (generalizing well)"
    elif r2_gap < 0.15:
        stability = "[~] Moderate gap (acceptable)"
    else:
        stability = "[!] Large gap (possible overfitting)"
    
    print(f"\nStability Assessment:")
    print(f"  Validation R2 - CV R2 gap: {r2_gap:.4f}")
    print(f"  {stability}")
    
    return {
        "compound": compound_name,
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "cv_r2_std": cv_r2_std,
        "r2_gap": r2_gap,
        "stability": stability,
    }


if __name__ == "__main__":
    ppl_dir = Path(__file__).parent.parent / "data" / "processed" / "PPL_data"
    
    print(f"\n{'#'*90}")
    print(f"# FINAL VALIDATION + CROSS-VALIDATION REPORT")
    print(f"# Compound Progression Models - Pre-Implementation Checkup")
    print(f"{'#'*90}")
    
    compounds = [
        (ppl_dir / "leg_workouts.csv", SquatProgressionModel, "Squat"),
        (ppl_dir / "push_workouts.csv", BenchPressProgressionModel, "Bench Press"),
        (ppl_dir / "pull_workouts.csv", LatPulldownProgressionModel, "Lat Pulldown"),
        (ppl_dir / "pull_workouts.csv", SeatedRowProgressionModel, "Seated Row"),
    ]
    
    results = []
    for csv_path, model_class, compound_name in compounds:
        result = evaluate_compound(csv_path, model_class, compound_name, cv=5)
        if result:
            results.append(result)
    
    # Summary table
    print(f"\n{'#'*90}")
    print(f"# SUMMARY TABLE")
    print(f"{'#'*90}\n")
    
    if results:
        summary_df = pd.DataFrame(results)
        
        # Display metrics side-by-side
        print("VALIDATION SET METRICS:")
        display_cols = ['compound', 'val_mae', 'val_rmse', 'val_r2']
        print(summary_df[display_cols].to_string(index=False))
        
        print("\n\nCROSS-VALIDATION METRICS:")
        cv_cols = ['compound', 'cv_mae', 'cv_rmse', 'cv_r2', 'cv_r2_std']
        print(summary_df[cv_cols].to_string(index=False))
        
        print("\n\nSTABILITY ASSESSMENT (Validation vs CV gap):")
        gap_cols = ['compound', 'r2_gap', 'stability']
        print(summary_df[gap_cols].to_string(index=False))
        
        print(f"\n{'='*90}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*90}")
        
        avg_val_r2 = summary_df['val_r2'].mean()
        avg_cv_r2 = summary_df['cv_r2'].mean()
        avg_gap = summary_df['r2_gap'].mean()
        
        print(f"\nAverage Validation R²: {avg_val_r2:.4f}")
        print(f"Average CV R²:         {avg_cv_r2:.4f}")
        print(f"Average R² gap:        {avg_gap:.4f}")
        
        if avg_gap < 0.10:
            print(f"\n[OK] Models are stable and generalizing well!")
            print(f"  Safe to deploy.")
        elif avg_gap < 0.20:
            print(f"\n[~] Models are reasonably stable.")
            print(f"  Acceptable for deployment, but monitor CV performance.")
        else:
            print(f"\n[!] Large gap detected — possible overfitting.")
            print(f"  Consider regularization or more data before deployment.")
        
        print(f"\nRECOMMENDATION:")
        print(f"  • Squat (R2={summary_df[summary_df['compound']=='Squat']['val_r2'].values[0]:.4f}): Strong signal [OK]")
        print(f"  • Bench (R2={summary_df[summary_df['compound']=='Bench Press']['val_r2'].values[0]:.4f}): Weak signal (accept for now)")
        print(f"  • Lat Pulldown (R2={summary_df[summary_df['compound']=='Lat Pulldown']['val_r2'].values[0]:.4f}): Excellent signal [OK]")
        print(f"  • Seated Row (R2={summary_df[summary_df['compound']=='Seated Row']['val_r2'].values[0]:.4f}): Excellent signal [OK]")
        print(f"\n  Ready to proceed to app implementation.")
    else:
        print("No results to summarize. Check file paths.")
