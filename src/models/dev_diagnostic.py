"""
dev_diagnostic.py

Simple test program for all progression models.
Uses lazy-loading registries to store model instances and metrics.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from .base_model import BaseModel

from .progression_model import (
    RandomForestProgressionModel,
    LinearRegressionProgressionModel,
    XGBoostProgressionModel,
)


# ===========================
# Registries (Lazy Loading)
# ===========================

model_registry = {}
model_full_name_registry = {
    "RFR": "Random Forest Regressor",
    "LR": "Linear Regression",
    "XG": "XGBoost",
}
metrics_registry = {}  # Stores (MAE, RMSE, R2) for each model
cv_registry = {}  # Stores (MAE_cv, RMSE_cv, R2_cv) for each model


# ===========================
# Helper Functions
# ===========================

def get_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate regression metrics.
    
    Returns:
        (MAE, RMSE, R2)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def get_score(
    model: object, X_valid: pd.DataFrame, y_valid: pd.Series
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation set.
    
    Returns:
        (MAE, RMSE, R2)
    """
    y_pred = model.predict(X_valid)
    mae, rmse, r2 = get_regression_metrics(y_valid.values, y_pred)
    return mae, rmse, r2


def get_cv_score(
    model: object, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
) -> Tuple[float, float, float]:
    """
    Calculate cross-validation metrics.
    
    Returns:
        (MAE_cv, RMSE_cv, R2_cv) - averaged across CV folds
    """
    # Clone the pipeline to get an unfitted version for cross-validation
    if hasattr(model, 'model_pipeline') and model.model_pipeline is not None:
        pipeline_to_cv = clone(model.model_pipeline)
    else:
        # Fallback: use just the estimator
        pipeline_to_cv = clone(model.estimator)
    
    # For MAE
    mae_cv_scores = cross_val_score(
        pipeline_to_cv,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    mae_cv = -mae_cv_scores.mean()  # Negate because sklearn uses neg_mae
    
    # For MSE -> RMSE
    mse_cv_scores = cross_val_score(
        clone(pipeline_to_cv),  # Clone again for fresh unfitted version
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    rmse_cv = np.sqrt(-mse_cv_scores.mean())  # Convert MSE to RMSE
    
    # For R2
    r2_cv_scores = cross_val_score(
        clone(pipeline_to_cv),  # Clone again for fresh unfitted version
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
    )
    r2_cv = r2_cv_scores.mean()
    
    return mae_cv, rmse_cv, r2_cv


# ===========================
# Data Loading
# ===========================

def load_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load processed baseline data.
    
    Returns:
        (DataFrame, target_column_name)
    """
    if data_path is None:
        # Try to find data relative to this file
        repo_root = Path(__file__).resolve().parents[2]
        data_path = repo_root / "data" / "processed" / "baseline_all_processed.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"\n[OK] Loaded data: {data_path.name} ({len(df)} rows, {len(df.columns)} columns)")
    
    return df, "effective_load"  # Default target column


# ===========================
# Model Testing
# ===========================

def test_model(
    model_key: str,
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> None:
    """
    Train and evaluate a model, store results in registries.
    """
    if model_key not in model_full_name_registry:
        print(f"[ERROR] Unknown model key: {model_key}")
        return
    
    model_name = model_full_name_registry[model_key]
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({model_key})")
    print(f"{'='*60}")
    
    # Skip if already trained
    if model_key in model_registry:
        print(f"[OK] Model already trained. Showing cached results...")
        mae, rmse, r2 = metrics_registry[model_key]
        print(f"  Validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        if model_key in cv_registry:
            mae_cv, rmse_cv, r2_cv = cv_registry[model_key]
            print(f"  CV        - MAE: {mae_cv:.4f}, RMSE: {rmse_cv:.4f}, R²: {r2_cv:.4f}")
        return
    
    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"\n[DATA] Train/Valid split: {len(X_train)}/{len(X_valid)} rows")
    
    # Instantiate model
    try:
        if model_key == "RFR":
            model = RandomForestProgressionModel(
                feature_columns=None,
                target_column=target_column,
            )
        elif model_key == "LR":
            model = LinearRegressionProgressionModel(
                feature_columns=None,
                target_column=target_column,
            )
        elif model_key == "XG":
            model = XGBoostProgressionModel(
                feature_columns=None,
                target_column=target_column,
            )
        else:
            print(f"[ERROR] Unknown model: {model_key}")
            return
    except ImportError as e:
        print(f"[ERROR] Import error (missing dependency?): {e}")
        return
    
    # Train
    print(f"[TRAIN] Training {model_name}...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return
    
    # Validate
    print(f"[EVAL] Evaluating on validation set...")
    try:
        mae, rmse, r2 = get_score(model, X_valid, y_valid)
        metrics_registry[model_key] = (mae, rmse, r2)
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return
    
    # Cross-validation
    print(f"[CV] Computing 5-fold cross-validation...")
    try:
        mae_cv, rmse_cv, r2_cv = get_cv_score(model, X_train, y_train, cv=5)
        cv_registry[model_key] = (mae_cv, rmse_cv, r2_cv)
        print(f"  MAE:  {mae_cv:.4f}")
        print(f"  RMSE: {rmse_cv:.4f}")
        print(f"  R²:   {r2_cv:.4f}")
    except Exception as e:
        print(f"[WARN] CV evaluation failed: {e}")
    
    # Store model
    model_registry[model_key] = model
    print(f"\n[SUCCESS] {model_name} ready for use!")


def show_results() -> None:
    """Display all trained models and their metrics."""
    if not model_registry:
        print("\n[WARN] No models trained yet.")
        return
    
    print(f"\n{'='*80}")
    print("Model Comparison Summary")
    print(f"{'='*80}")
    
    for model_key in ["RFR", "LR", "XG"]:
        if model_key not in model_registry:
            print(f"{model_key:3s} - {model_full_name_registry[model_key]:25s} | Not trained")
            continue
        
        mae, rmse, r2 = metrics_registry.get(model_key, (None, None, None))
        mae_cv, rmse_cv, r2_cv = cv_registry.get(model_key, (None, None, None))
        
        val_str = f"Val: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}" if mae else "Validation N/A"
        cv_str = f"CV:  MAE={mae_cv:.4f}, RMSE={rmse_cv:.4f}, R²={r2_cv:.4f}" if mae_cv else "CV N/A"
        
        print(f"\n{model_key} - {model_full_name_registry[model_key]}")
        print(f"  {val_str}")
        print(f"  {cv_str}")
    
    print(f"\n{'='*80}")


# ===========================
# Main Menu
# ===========================

def main() -> None:
    """Main interactive menu."""
    print("\n" + "="*60)
    print("Progression Model Diagnostic Tool")
    print("="*60)
    
    # Load data once
    try:
        df, target_col = load_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Main loop
    while True:
        print("\n" + "="*60)
        print("Menu Options:")
        print("1. Test RFR (Random Forest Regressor)")
        print("2. Test LR  (Linear Regression)")
        print("3. Test XG  (XGBoost)")
        print("4. Show Results Summary")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            test_model("RFR", df, target_col)
        elif choice == "2":
            test_model("LR", df, target_col)
        elif choice == "3":
            test_model("XG", df, target_col)
        elif choice == "4":
            show_results()
        elif choice == "5":
            print("\n[EXIT] Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
