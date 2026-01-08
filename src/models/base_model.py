"""
Base class for ML models (train, predict, save, load)
Provides a common interface for progression, fatigue, or future models.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector


# ===========================
# Custom Feature Transformers
# ===========================

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
	"""Apply all feature engineering steps in sequence in-place within the pipeline."""

	def __init__(self, n_rolling: int = 3, n_trend: int = 5):
		self.n_rolling = n_rolling
		self.n_trend = n_trend

	def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineeringTransformer:
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		df = X.copy()

		# Time features
		if "date" in df.columns:
			df["date"] = pd.to_datetime(df["date"])
			first_workout_date = df["date"].min()
			df["days_since_first_workout"] = (df["date"] - first_workout_date).dt.days
			df = df.sort_values(by="date")
			df["days_since_last_workout"] = df["date"].diff().dt.days.fillna(0).astype(int)

		# Session number per exercise
		if {"exercise_normalized", "date"}.issubset(df.columns):
			df = df.sort_values(by=["exercise_normalized", "date"])
			df["session_number"] = (
				df.groupby("exercise_normalized")["date"].rank(method="dense").astype(int)
			)

		# Rolling load features
		if {"exercise_normalized", "effective_load"}.issubset(df.columns):
			df = df.sort_values(by=["exercise_normalized", "date"])
			df[f"rolling_avg_load_last_{self.n_rolling}_sessions"] = (
				df.groupby("exercise_normalized")["effective_load"]
				.rolling(window=self.n_rolling, min_periods=1)
				.mean()
				.reset_index(level=0, drop=True)
			)
			if "session_number" in df.columns:
				def compute_slope(x: np.ndarray) -> float:
					if len(x) < 2:
						return 0.0
					y = x.values
					x_vals = np.arange(len(y))
					A = np.vstack([x_vals, np.ones(len(x_vals))]).T
					slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
					return slope

				df["rolling_trend_load"] = (
					df.groupby("exercise_normalized")["effective_load"]
					.rolling(window=self.n_trend, min_periods=2)
					.apply(compute_slope, raw=False)
					.reset_index(level=0, drop=True)
				)
				df["rolling_trend_load"] = df["rolling_trend_load"].fillna(df["rolling_trend_load"].mean())

		# RPE handling
		if "rpe" in df.columns:
			df["rpe_missing"] = df["rpe"].isna().astype(int)
			median_rpe = df["rpe"].median()
			df["rpe"] = df["rpe"].fillna(median_rpe)
			bins = [0, 5, 7, 10]
			labels = ["Low", "Medium", "High"]
			df["rpe_binned"] = pd.cut(df["rpe"], bins=bins, labels=labels, include_lowest=True)
			rpe_mapping = {"Low": 1, "Medium": 2, "High": 3}
			df["rpe_ordinal"] = df["rpe_binned"].map(rpe_mapping).astype(int)

		# Top set flag
		if {"days_since_first_workout", "workout_name", "exercise_normalized", "set_volume"}.issubset(df.columns):
			df["is_top_set"] = False
			idx = (
				df.groupby(["days_since_first_workout", "workout_name", "exercise_normalized"])["set_volume"]
				.idxmax()
			)
			df.loc[idx, "is_top_set"] = True

		# Reps binned
		if "reps" in df.columns:
			df["reps_binned"] = pd.cut(df["reps"], bins=[0, 5, 15, np.inf], labels=["Strength", "Hypertrophy", "Endurance"], include_lowest=True)

		# Convert any remaining datetime to numeric seconds
		for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
			df[col] = df[col].astype("int64") // 1_000_000_000

		return df



def _repo_root() -> Path:
	# src/models/base_model.py -> src -> repo root
	return Path(__file__).resolve().parents[2]


def _user_models_dir(username: str) -> Path:
	return _repo_root() / "users" / username / "models"


@dataclass
class ModelMetadata:
	model_name: str
	version: str = "0.1.0"
	created_at: str = datetime.utcnow().isoformat()
	feature_columns: Optional[Sequence[str]] = None
	target_column: Optional[str] = None
	random_state: Optional[int] = None
	extra: Optional[dict] = None


class BaseModel:
	"""
	Generic base wrapper around an estimator.

	- Provides fit/predict and regression evaluation helpers
	- Saves/loads per-user models under users/<username>/models/
	- Stores metadata (features, target, version, created_at)
	"""

	def __init__(
		self,
		estimator: Any,
		model_name: str,
		*,
		feature_columns: Optional[Sequence[str]] = None,
		target_column: Optional[str] = None,
		version: str = "0.1.0",
		random_state: Optional[int] = None,
		extra: Optional[dict] = None,
	) -> None:
		self.estimator = estimator
		self.metadata = ModelMetadata(
			model_name=model_name,
			version=version,
			feature_columns=tuple(feature_columns) if feature_columns else None,
			target_column=target_column,
			random_state=random_state,
			extra=extra or {},
		)
		self._fitted: bool = False

		# Apply random_state if estimator supports it
		if random_state is not None and hasattr(self.estimator, "random_state"):
			try:
				setattr(self.estimator, "random_state", random_state)
			except Exception:
				pass

	# -------------------------------
	# Data handling helpers
	# -------------------------------
	def _select_X(self, X: Any) -> Any:
		cols = self.metadata.feature_columns
		if isinstance(X, pd.DataFrame) and cols:
			missing = [c for c in cols if c not in X.columns]
			if missing:
				raise ValueError(f"Missing required feature columns: {missing}")
			return X.loc[:, list(cols)]
		return X

	# -------------------------------
	# Feature engineering & preprocessing
	# -------------------------------
	def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Basic preprocessing placeholder.

		- Drops unneeded columns (workout_notes, distance, notes, seconds)
		- Converts datetime columns to numeric timestamps
		- Returns a copy for safety
		"""
		if df is None:
			return pd.DataFrame()

		out = df.copy()
		drop_cols = {"workout_notes", "distance", "notes", "seconds"}
		existing_drop = [c for c in drop_cols if c in out.columns]
		if existing_drop:
			out = out.drop(columns=existing_drop)

		for col in out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
			out[col] = out[col].astype("int64") // 1_000_000_000  # seconds since epoch

		return out

	def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
		target = self.metadata.target_column
		if self.metadata.feature_columns:
			return [c for c in self.metadata.feature_columns if c in df.columns and c != target]
		# Default: use all columns except target
		return [c for c in df.columns if c != target]

	def _build_col_transformer(self) -> ColumnTransformer:
		"""ColumnTransformer that auto-selects numeric/categorical columns after feature engineering."""
		num_pipeline = Pipeline([
			("imputer", SimpleImputer(strategy="mean")),
			("scaler", StandardScaler()),
		])

		cat_pipeline = Pipeline([
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		])

		return ColumnTransformer(
			transformers=[
				("num", num_pipeline, make_column_selector(dtype_include=np.number)),
				("cat", cat_pipeline, make_column_selector(dtype_exclude=np.number)),
			],
			remainder="drop",
		)

	def build_preprocessor(self, df: pd.DataFrame) -> Pipeline:
		"""Build a preprocessing pipeline (unused in new unified pipeline)."""
		raise NotImplementedError("build_preprocessor is unused in the unified pipeline approach")

	# -------------------------------
	# Core API
	# -------------------------------
	def fit(self, X: Any, y: Optional[Iterable] = None, **fit_kwargs: Any) -> "BaseModel":
		# If a DataFrame is provided, build a single end-to-end pipeline
		if isinstance(X, pd.DataFrame):
			self.model_pipeline = Pipeline([
				("feature_engineering", FeatureEngineeringTransformer()),
				("preprocessing", self._build_col_transformer()),
				("estimator", self.estimator),
			])
			try:
				if y is None:
					self.model_pipeline.fit(X, **fit_kwargs)
				else:
					self.model_pipeline.fit(X, y, **fit_kwargs)
			except Exception as e:
				print(f"[DEBUG] Pipeline fit failed: {e}")
				import traceback
				traceback.print_exc()
				raise
			self._fitted = True
			return self

		# Fallback for array-like
		X_sel = self._select_X(X)
		if y is None:
			self.estimator.fit(X_sel, **fit_kwargs)
		else:
			self.estimator.fit(X_sel, y, **fit_kwargs)
		self._fitted = True
		return self

	def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
		if not self._fitted:
			raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

		if hasattr(self, "model_pipeline") and self.model_pipeline is not None:
			# Pipeline includes feature engineering, preprocessing, and estimator
			return self.model_pipeline.predict(X, **predict_kwargs)

		X_sel = self._select_X(X)
		return self.estimator.predict(X_sel, **predict_kwargs)

	# -------------------------------
	# Evaluation (regression)
	# -------------------------------
	def evaluate_regression(self, X: Any, y_true: Iterable, **predict_kwargs: Any) -> dict:
		y_pred = self.predict(X, **predict_kwargs)
		y_true = np.asarray(list(y_true), dtype=float)
		y_pred = np.asarray(y_pred, dtype=float)
		mae = float(np.mean(np.abs(y_true - y_pred)))
		mse = float(np.mean((y_true - y_pred) ** 2))
		rmse = float(np.sqrt(mse))
		# r2 guard against zero variance
		var = np.var(y_true)
		r2 = float(1.0 - mse / var) if var > 0 else float("nan")
		return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

	# -------------------------------
	# Persistence (per-user)
	# -------------------------------
	def save(self, username: str, *, overwrite: bool = True) -> Path:
		models_dir = _user_models_dir(username)
		models_dir.mkdir(parents=True, exist_ok=True)
		path = models_dir / f"{self.metadata.model_name}.pkl"
		if path.exists() and not overwrite:
			raise FileExistsError(f"Model already exists: {path}")
		payload = {
			"pipeline": getattr(self, "model_pipeline", None),
			"metadata": asdict(self.metadata),
			"fitted": self._fitted,
		}
		joblib.dump(payload, path)
		return path

	@classmethod
	def load(cls, username: str, model_name: str) -> "BaseModel":
		path = _user_models_dir(username) / f"{model_name}.pkl"
		if not path.exists():
			raise FileNotFoundError(f"No saved model for user '{username}': {path}")
		payload = joblib.load(path)
		md = payload.get("metadata", {})
		pipeline = payload.get("pipeline")
		# Extract estimator from pipeline if present
		estimator = None
		if pipeline is not None and hasattr(pipeline, "named_steps"):
			estimator = pipeline.named_steps.get("estimator")
		obj = cls(
			estimator=estimator,
			model_name=md.get("model_name", model_name),
			feature_columns=md.get("feature_columns"),
			target_column=md.get("target_column"),
			version=md.get("version", "0.1.0"),
			random_state=md.get("random_state"),
			extra=md.get("extra"),
		)
		obj._fitted = bool(payload.get("fitted", False))
		obj.model_pipeline = pipeline
		# Preserve original created_at if present
		if "created_at" in md:
			obj.metadata.created_at = md["created_at"]
		return obj

	# -------------------------------
	# Utilities
	# -------------------------------
	def set_feature_columns(self, columns: Sequence[str]) -> None:
		self.metadata.feature_columns = tuple(columns)

	def set_target_column(self, column: str) -> None:
		self.metadata.target_column = column

	def info(self) -> dict:
		return asdict(self.metadata)

	def __repr__(self) -> str:
		cls = self.__class__.__name__
		return f"{cls}(name={self.metadata.model_name!r}, version={self.metadata.version!r}, fitted={self._fitted})"