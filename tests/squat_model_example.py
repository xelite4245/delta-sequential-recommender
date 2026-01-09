import pandas as pd
import numpy as np
from utils import *
import re 

#skklearn imports
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Optional, Sequence, Tuple


#Problem: ImportError: attempted relative import with no known parent package
# FIX: Put models in their own package and import from there
# Now we call it like this:

from models.base_model import BaseModel, FeatureEngineeringTransformer

from models.progression_model import (
    RandomForestProgressionModel,
    LinearRegressionProgressionModel,
    XGBoostProgressionModel
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

def add_periodization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect training cycles and add phase features.
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Detect deloads (big drops in weight)
    df['is_deload'] = (df['weight'].shift(1) - df['weight']) > 30  # 30+ lb drop
    
    # Detect start of new cycle (weight increases after deload)
    df['post_deload'] = df['is_deload'].shift(1).fillna(False)
    
    # Calculate cycle number (increments at each deload)
    df['cycle_number'] = df['is_deload'].cumsum()
    
    # Position within cycle (sessions since last deload)
    df['weeks_in_cycle'] = df.groupby('cycle_number').cumcount() + 1
    
    # Distance from previous max (are you in PR territory?)
    df['max_weight_so_far'] = df['weight'].expanding().max()
    df['percent_of_max'] = df['weight'] / df['max_weight_so_far']
    
    # Trend within current cycle
    df['cycle_weight_trend'] = (
        df.groupby('cycle_number')['weight']
        .transform(lambda x: x.diff().fillna(0))
    )
    
    return df


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

def get_cv_score(model, X, y, cv=5):
    """
    Time-series cross-validation for progression models.
    Uses TimeSeriesSplit to respect temporal ordering.
    """
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from copy import deepcopy
    
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
        
        # Clone model - try sklearn clone, fall back to deepcopy
        try:
            model_cv = clone(model)
        except Exception as e:
            # Use deepcopy as fallback
            try:
                model_cv = deepcopy(model)
            except Exception as e2:
                print(f"Warning: Could not clone model for fold {fold+1}: {e2}")
                continue
        
        # Fit on train fold
        model_cv.fit(X_train_cv, y_train_cv)
        
        # Predict on validation fold
        y_pred = model_cv.predict(X_val_cv)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val_cv, y_pred)
        mse = mean_squared_error(y_val_cv, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_cv, y_pred)
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        print(f"  Fold {fold+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    if len(mae_scores) == 0:
        print("ERROR: No CV folds completed!")
        return float('nan'), float('nan'), float('nan')
    
    # Return averages
    return np.mean(mae_scores), np.mean(rmse_scores), np.mean(r2_scores)



def test_model_regression(
    model_key: str,
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str] = None,
    train_fraction: float = 0.8,
    cv: int = 5 
) -> None:
    """
    Test regression model and store results in registries.
    """

    # Lazy model initialization
    if model_key not in model_registry:
        if model_key == "RFR":
            model_registry[model_key] = RandomForestProgressionModel(max_depth = 4, max_features='sqrt', min_samples_leaf=3, min_samples_split=8, n_estimators=100)
        elif model_key == "LR":
            model_registry[model_key] = LinearRegressionProgressionModel()
        elif model_key == "XG":
            model_registry[model_key] = XGBoostProgressionModel()
        else:
            raise ValueError(f"Unknown model key: {model_key}")
        
    model = model_registry[model_key]
    model_name = model_full_name_registry[model_key]

    print("="*40)
    print(f"\n\nTesting model: {model_name}")
    print("="*40)

    # if model already cached
    if model_key in metrics_registry:
        print("Model already tested. Retrieving cached results.")
        mae, rmse, r2 = metrics_registry[model_key]
        mae_cv, rmse_cv, r2_cv = cv_registry[model_key]
        print(f"  Validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"  CV        - MAE: {mae_cv:.4f}, RMSE: {rmse_cv:.4f}, R²: {r2_cv:.4f}")
        return
    

    # Split data (not randomly since it's progression over time)
    if "date" not in df.columns:
        raise ValueError("Time-based split requires a 'date' column.")

    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date")

    split_idx = int(len(df_sorted) * train_fraction)

    train_df = df_sorted.iloc[:split_idx]
    valid_df = df_sorted.iloc[split_idx:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col]

    # ===========================
    # Train model (pipeline handles FE)
    # ===========================
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        #Having inconsistent number of sample
        print(f"Error during model fitting: {e}")
        return
        

    # ===========================
    # Validation metrics
    # ===========================
    mae, rmse, r2 = get_score(model, X_valid, y_valid)
    metrics_registry[model_key] = (mae, rmse, r2)

    # ===========================
    # CV (still random; acceptable baseline)
    # ===========================
    mae_cv, rmse_cv, r2_cv = get_cv_score(model, X_train, y_train, cv=cv)
    cv_registry[model_key] = (mae_cv, rmse_cv, r2_cv)

    # ===========================
    # Print results
    # ===========================
    print(f"  Validation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"  CV        - MAE: {mae_cv:.4f}, RMSE: {rmse_cv:.4f}, R²: {r2_cv:.4f}")
        


def RFR_find_best_hyperparameters(
    df: pd.DataFrame,
    target_col: str,
    param_grid: dict,
    time_col: Optional[str] = "date",
    train_fraction: float = 0.8,
    cv: int = 3
) -> dict:
    """
    Grid search for best hyperparameters for Random Forest Regressor.
    Uses CV for more robust evaluation.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    # Split data
    df_sorted = df.copy()
    if "date" in df_sorted.columns and not pd.api.types.is_numeric_dtype(df_sorted["date"]):
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date")

    split_idx = int(len(df_sorted) * train_fraction)

    train_df = df_sorted.iloc[:split_idx]
    valid_df = df_sorted.iloc[split_idx:]

    X_train = train_df.drop(columns=[target_col, "date"])
    y_train = train_df[target_col]

    X_valid = valid_df.drop(columns=[target_col, "date"])
    y_valid = valid_df[target_col]

    best_params = None
    best_score = float("inf")
    best_val_mae = float("inf")
    
    tscv = TimeSeriesSplit(n_splits=cv)
    
    results = []

    for params in ParameterGrid(param_grid):
        print(f"Testing: {params}")
        
        # CV score
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            cv_maes.append(mean_absolute_error(y_val, y_pred))
        
        avg_cv_mae = np.mean(cv_maes)
        
        # Validation score
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        val_mae = mean_absolute_error(y_valid, y_pred)
        
        print(f"  CV MAE: {avg_cv_mae:.2f}, Val MAE: {val_mae:.2f}")
        
        results.append({
            'params': params,
            'cv_mae': avg_cv_mae,
            'val_mae': val_mae
        })
        
        # Use CV score to pick best (more robust)
        if avg_cv_mae < best_score:
            best_score = avg_cv_mae
            best_params = params
            best_val_mae = val_mae

    print(f"\n{'='*60}")
    print(f"Best params: {best_params}")
    print(f"  CV MAE: {best_score:.2f}")
    print(f"  Val MAE: {best_val_mae:.2f}")
    print(f"{'='*60}")
    
    return best_params, results







if __name__ == "__main__":
    
    # ========== Load and process REAL data ==========
    df = pd.read_csv("data/leg_day_exercises.csv")
    df = df[df["exercise_normalized"] == "squat"]
    print(f"Number of rows after filtering for squats only: {len(df)}")

    # Apply feature engineering
    feature_transformer = FeatureEngineeringTransformer()
    df = feature_transformer.fit_transform(df)
    df = df[df["is_top_set"] == True]
    print(f"Number of rows after filtering for top sets only: {len(df)}")

    # ========== Add periodization features ==========
    df = add_periodization_features(df)
    print("Added periodization features")

    # ========== Calculate load_delta ==========
    df = df.sort_values(by=["exercise_normalized", "date"])
    df["load_delta"] = df.groupby("exercise_normalized")["weight"].transform(
        lambda x: x.shift(-1) - x
    )
    df["load_delta"] = df["load_delta"].clip(lower=-100, upper=100)
    df = df.dropna(subset=["load_delta"])
    
    print(f"Rows after load_delta calculation: {len(df)}")

    # ========== Check new correlations ==========
    print("\n========== CORRELATIONS WITH PERIODIZATION FEATURES ==========")
    corr_features = [
        "weight", "reps", "session_number",
        "rolling_avg_load_last_3_sessions", "rolling_trend_load",
        "weeks_in_cycle", "percent_of_max", "cycle_weight_trend"
    ]
    for col in corr_features:
        if col in df.columns:
            corr = df[[col, "load_delta"]].corr().iloc[0, 1]
            print(f"  {col}: {corr:7.3f}")

    # ========== Clip features (not target) ==========
    feature_cols = ["weight", "effective_load", "set_volume", 
                    "rolling_avg_load_last_3_sessions", "rolling_trend_load", 
                    "percent_of_max", "cycle_weight_trend", "max_weight_so_far"
                    ]
    
    for col in feature_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"{col} - clipped to [{lower:.2f}, {upper:.2f}]")

    # Drop RPE if no variance
    if 'rpe' in df.columns and df['rpe'].nunique() <= 1:
        print("Dropping RPE (no variance)")
        df = df.drop(columns=['rpe'])














    # ========== Model Testing ==========
    
    # ========== Feature Selection ==========
    print("\n========== USING BEST FEATURES ONLY ==========")

    best_features = [
        "weight",
        "reps", 
        "date",                 # Time-based split
        "percent_of_max",       # Strongest correlation: -0.523
        "cycle_weight_trend",   # Strong correlation: -0.384
        "weeks_in_cycle",       # Moderate: -0.254
        "rolling_trend_load",   # Weak but might help: -0.136
        "load_delta"            # Target
    ]

    # Create filtered dataset
    df_best = df[best_features].copy()

    print(f"Using {len(best_features)-1} features: {[f for f in best_features if f != 'load_delta']}")

    # ========== Test with best features ==========
    print("\n" + "="*40)
    print("Random Forest - BEST FEATURES ONLY")
    print("="*40)
    print(f"\nColumns in df_best: {df_best.columns.tolist()}")
    print(f"'date' in df_best: {'date' in df.columns}")
    print(f"df_best shape: {df_best.shape}")

    # SMALLER, SMARTER GRID for 141 samples
    # param_grid = {
    #     'n_estimators': [100, 150],              # More trees is almost always better
    #     'max_depth': [4, 6, 8],                  # Key parameter for small data
    #     'min_samples_split': [8, 12, 16],        # Prevent overfitting
    #     'min_samples_leaf': [3, 4, 5],           # Larger leaves for small data
    #     'max_features': ['sqrt']                 # Fixed - sqrt is usually best
    # }

    # print(f"Grid size: {len(list(ParameterGrid(param_grid)))} combinations")

    # best_params, results = RFR_find_best_hyperparameters(
    #     df=df_best,
    #     target_col="load_delta",
    #     param_grid=param_grid,
    #     time_col="date",
    #     train_fraction=0.85,
    #     cv=3
    # )
    # Best params: {'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 100}
        

    test_model_regression(
        model_key="RFR",
        df=df_best,
        target_col="load_delta",
        time_col="date",
        train_fraction=0.85,
        cv=3
    )

    # ========== Statistics Summary ==========
    print("\n========== PROGRESSION STATISTICS ==========")
    print(f"Total sessions: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Weight range: {df['weight'].min():.1f} - {df['weight'].max():.1f} lbs")
    print(f"\nLoad Delta Summary:")
    print(f"  Mean: {df['load_delta'].mean():.2f} lbs")
    print(f"  Median: {df['load_delta'].median():.2f} lbs")
    print(f"  Std: {df['load_delta'].std():.2f} lbs")
    
    print(f"\nLoad delta distribution:")
    print(f"  Increased (>0): {(df['load_delta'] > 0).sum()} ({(df['load_delta'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"  Same (=0): {(df['load_delta'] == 0).sum()} ({(df['load_delta'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  Decreased (<0): {(df['load_delta'] < 0).sum()} ({(df['load_delta'] < 0).sum()/len(df)*100:.1f}%)")
    
    if 'cycle_number' in df.columns:
        print(f"\nTraining Cycles:")
        print(f"  Total cycles detected: {df['cycle_number'].max()}")
        print(f"  Avg sessions per cycle: {df.groupby('cycle_number').size().mean():.1f}")

    
    