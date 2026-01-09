"""
Utility to run predictions with per-user affine calibration.

Workflow:
- Optionally refit an affine correction on recent history when enough samples exist.
- Apply stored scaling/offset (or newly refit) to upcoming predictions.
- Persist personalization to disk via PersonalizationRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Support both package and src-path execution
try:  # pragma: no cover - exercise import flexibility
    from .models.base_model import BaseModel
    from .utils.user_personalization import PersonalizationRegistry
except ImportError:  # pragma: no cover
    from models.base_model import BaseModel
    from utils.user_personalization import PersonalizationRegistry


@dataclass
class CalibrationConfig:
    """Configuration for per-user affine calibration."""

    min_samples: int = 8
    refit_every: int = 10
    calibration_window: int = 32
    gain_bounds: Tuple[float, float] = (0.6, 1.4)
    target_column: str = "load_delta"
    save_on_update: bool = True


def _prep_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop target column while keeping all other columns intact."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.drop(columns=[target_column], errors="ignore")


def _should_refit(meta: dict, available: int, refit_every: int) -> bool:
    last_size = meta.get("last_calibrated_size", 0)
    if last_size == 0:
        return True
    return (available - last_size) >= refit_every


def maybe_calibrate_affine(
    *,
    model: BaseModel,
    registry: PersonalizationRegistry,
    user_id: str,
    compound: str,
    history: Optional[pd.DataFrame],
    config: CalibrationConfig,
) -> Optional[Tuple[float, float]]:
    """Refit affine correction if history is sufficient and cadence is met."""
    if history is None or history.empty:
        return None
    if config.target_column not in history.columns:
        return None

    usable = history.dropna(subset=[config.target_column]).tail(config.calibration_window)
    if len(usable) < config.min_samples:
        return None

    up = registry.get_or_create(user_id)
    meta = up.calibration_meta.get(compound, {"last_calibrated_size": 0, "runs": 0})
    if not _should_refit(meta, len(usable), config.refit_every):
        return None

    X_hist = _prep_features(usable, config.target_column)
    y_true = usable[config.target_column].values
    y_pred = model.predict(X_hist)

    ab = up.calibrate_affine(
        compound=compound,
        y_true=y_true,
        y_pred=y_pred,
        min_samples=config.min_samples,
        gain_bounds=config.gain_bounds,
    )
    if ab and config.save_on_update:
        registry.save(user_id)
    return ab


def predict_with_user_calibration(
    *,
    model: BaseModel,
    user_id: str,
    compound: str,
    future_df: pd.DataFrame,
    registry: Optional[PersonalizationRegistry] = None,
    history: Optional[pd.DataFrame] = None,
    config: Optional[CalibrationConfig] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """
    Predict load deltas and apply per-user affine calibration if available.

    Returns (raw_pred, adjusted_pred, fitted_calibration) where the last item
    is the (a, b) tuple if a calibration was refit during this call.
    """
    cfg = config or CalibrationConfig()
    reg = registry or PersonalizationRegistry()

    fitted = maybe_calibrate_affine(
        model=model,
        registry=reg,
        user_id=user_id,
        compound=compound,
        history=history,
        config=cfg,
    )

    up = reg.get_or_create(user_id)
    X_future = _prep_features(future_df, cfg.target_column)
    raw_pred = model.predict(X_future)
    adjusted = up.adjust_prediction(compound, raw_pred)
    return raw_pred, adjusted, fitted
