"""
progression_model.py

Subclass of BaseModel for predicting the next workout progression for a user.
Includes:
- Training and prediction of weights, reps, and volume per exercise
- Feature handling specific to workout progression
- Integration with baseline and per-user models for progressive personalization

Will be used by the main application to recommend next workout parameters.

Models in this file:
- RandomForestProgressionModel: Uses Random Forests to predict next workout parameters.
- LinearRegressionProgressionModel: Uses Linear Regression for predictions.
- XGBoostProgressionModel: Uses XGBoost for enhanced prediction accuracy.

At the end, after testing each model, we will select the best performing one for deployment.
"""

from __future__ import annotations

from typing import Optional, Sequence, Any

import numpy as np

try:
	from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	XGBRegressor = None  # type: ignore

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base_model import BaseModel


DEFAULT_FEATURES: tuple[str, ...] = (
	"set_volume",
	"effective_load",
	"rpe",
	"set_order",
	"reps",
	"weight",
)


class RandomForestProgressionModel(BaseModel):
	"""Random Forest regression model for workout progression."""

	def __init__(
		self,
		*,
		feature_columns: Optional[Sequence[str]] = DEFAULT_FEATURES,
		target_column: Optional[str] = None,
		n_estimators: int = 300,
		max_depth: Optional[int] = None,
		random_state: Optional[int] = 42,
		**kwargs: Any,
	) -> None:
		estimator = RandomForestRegressor(
			n_estimators=n_estimators,
			max_depth=max_depth,
			random_state=random_state,
			n_jobs=-1,
		)
		super().__init__(
			estimator=estimator,
			model_name="progression_rf",
			feature_columns=feature_columns,
			target_column=target_column,
			random_state=random_state,
			extra={"model_type": "random_forest", **kwargs},
		)


class LinearRegressionProgressionModel(BaseModel):
	"""Linear Regression baseline for workout progression."""

	def __init__(
		self,
		*,
		feature_columns: Optional[Sequence[str]] = DEFAULT_FEATURES,
		target_column: Optional[str] = None,
		fit_intercept: bool = True,
		random_state: Optional[int] = None,
		**kwargs: Any,
	) -> None:
		estimator = LinearRegression(fit_intercept=fit_intercept)
		super().__init__(
			estimator=estimator,
			model_name="progression_linear",
			feature_columns=feature_columns,
			target_column=target_column,
			random_state=random_state,
			extra={"model_type": "linear_regression", **kwargs},
		)


class XGBoostProgressionModel(BaseModel):
	"""XGBoost regression model for workout progression (optional dependency)."""

	def __init__(
		self,
		*,
		feature_columns: Optional[Sequence[str]] = DEFAULT_FEATURES,
		target_column: Optional[str] = None,
		n_estimators: int = 400,
		learning_rate: float = 0.05,
		max_depth: int = 6,
		subsample: float = 0.9,
		colsample_bytree: float = 0.9,
		random_state: Optional[int] = 42,
		**kwargs: Any,
	) -> None:
		if XGBRegressor is None:
			raise ImportError("xgboost is not installed; install it to use XGBoostProgressionModel")

		estimator = XGBRegressor(
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			max_depth=max_depth,
			subsample=subsample,
			colsample_bytree=colsample_bytree,
			objective="reg:squarederror",
			random_state=random_state,
			n_jobs=-1,
		)
		super().__init__(
			estimator=estimator,
			model_name="progression_xgb",
			feature_columns=feature_columns,
			target_column=target_column,
			random_state=random_state,
			extra={"model_type": "xgboost", **kwargs},
		)
