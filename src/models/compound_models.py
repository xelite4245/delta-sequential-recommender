"""
Compound Progression Models for core lifts.

Each model inherits from BaseModel and is specialized for:
- Specific compound exercise (Squat, Bench Press, etc.)
- load_delta target (next session's weight change)
- Periodization features (deload detection, cycle position)
- Random Forest with tuned hyperparameters

These are "signal generators" trained on clean, filtered data.
Other exercises map to these via exercise_mapping.py.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from typing import Optional, Tuple

from .base_model import BaseModel


def add_periodization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect training cycles and add phase features.
    
    Features:
    - is_deload: weight drop >= 15% (recovery week marker, scale-invariant)
    - cycle_number: cumulative deload count
    - weeks_in_cycle: sessions since last deload
    - max_weight_so_far: expanding max weight
    - percent_of_max: current weight / max_weight_so_far
    - cycle_weight_trend: within-cycle weight trend
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Detect deloads (percentage-based drop, scale-invariant)
    # 15% drop = deload marker (works for 50 lb beginners and 300 lb lifters)
    prev_weight = df['weight'].shift(1)
    pct_change = (prev_weight - df['weight']) / prev_weight
    df['is_deload'] = pct_change >= 0.15
    
    # Calculate cycle number (increments at each deload)
    df['cycle_number'] = df['is_deload'].cumsum()
    
    # Position within cycle (sessions since last deload)
    df['weeks_in_cycle'] = df.groupby('cycle_number').cumcount() + 1
    
    # Distance from previous max (PR territory?)
    df['max_weight_so_far'] = df['weight'].expanding().max()
    df['percent_of_max'] = df['weight'] / df['max_weight_so_far']
    
    # Trend within current cycle
    df['cycle_weight_trend'] = (
        df.groupby('cycle_number')['weight']
        .transform(lambda x: x.diff().fillna(0))
    )
    
    return df


class CompoundProgressionModel(BaseModel):
    """
    Base class for compound movement progression models.
    
    Uses:
    - load_delta as target (weight change to next session)
    - Periodization features (deload, cycle position, PR distance)
    - Random Forest with tuned hyperparameters
    - Top-set data only
    """
    
    def __init__(
        self,
        name: str = "Compound",
        estimator_type: str = "rf",
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize compound model.
        
        Args:
            name: Human-readable name (e.g., "Squat", "Bench Press")
            estimator_type: "rf" for Random Forest (only option for now)
            random_state: For reproducibility
            **kwargs: Passed to RandomForestRegressor (n_estimators, max_depth, etc.)
        """
        self.name = name
        self.estimator_type = estimator_type
        self.random_state = random_state
        self.model_kwargs = kwargs
        
        # Default RFR hyperparameters (best from grid search on squat)
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'max_features': 'sqrt',
            'min_samples_leaf': 3,
            'min_samples_split': 8,
            'random_state': random_state,
            'n_jobs': -1,
        }
        
        # Override with provided kwargs
        default_params.update(kwargs)
        
        if estimator_type == "rf":
            estimator = RandomForestRegressor(**default_params)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        super().__init__(estimator=estimator, model_name=name)
    
    @staticmethod
    def prepare_compound_data(
        df: pd.DataFrame,
        target_col: str = "load_delta",
        drop_cols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data: periodization features, clip outliers, compute target.
        
        Args:
            df: Raw dataframe (assumed filtered to one compound + top sets)
            target_col: Column to predict (default: load_delta)
            drop_cols: Columns to drop before training
            
        Returns:
            (X, y) ready for fit()
        """
        df = df.copy()
        
        # Add periodization features
        df = add_periodization_features(df)
        
        # Clip outliers on features (not target)
        feature_cols = [
            "weight", "effective_load", "set_volume",
            "rolling_avg_load_last_3_sessions", "rolling_trend_load",
            "percent_of_max", "cycle_weight_trend", "max_weight_so_far"
        ]
        
        for col in feature_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        # Drop low-variance columns
        if 'rpe' in df.columns and df['rpe'].nunique() <= 1:
            df = df.drop(columns=['rpe'])
        
        # Default drop list
        if drop_cols is None:
            drop_cols = ['date']  # Keep date for split, let BaseModel handle
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        y = df[target_col]
        X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
        
        return X, y


class SquatProgressionModel(CompoundProgressionModel):
    """Random Forest model for squat progression."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Squat",
            estimator_type="rf",
            **kwargs
        )


class BenchPressProgressionModel(CompoundProgressionModel):
    """Random Forest model for bench press progression."""
    
    def __init__(self, **kwargs):
        # Baseline hyperparameters (max_depth=4, min_samples_leaf=3)
        defaults = {
            'n_estimators': 100,
            'max_depth': 4,
            'max_features': 'sqrt',
            'min_samples_leaf': 3,
            'min_samples_split': 8,
        }
        defaults.update(kwargs)
        super().__init__(
            name="Bench Press",
            estimator_type="rf",
            **defaults
        )


class LatPulldownProgressionModel(CompoundProgressionModel):
    """Random Forest model for lat pulldown (pull vertical) progression."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Lat Pulldown",
            estimator_type="rf",
            **kwargs
        )


class SeatedRowProgressionModel(CompoundProgressionModel):
    """Random Forest model for seated row (pull horizontal) progression."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Seated Row",
            estimator_type="rf",
            **kwargs
        )
