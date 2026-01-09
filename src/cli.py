"""
Command-line interface for the application.
Allows users to log workouts, get predictions, and view progression stats.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

"""
Note: Avoid importing data_pipeline at module import time to prevent
the utils module/package name clash from affecting commands that don't
need it (e.g., predict). We'll import data_pipeline lazily inside the
relevant command handlers.
"""
from .models.compound_models import (
    SquatProgressionModel,
    BenchPressProgressionModel,
    LatPulldownProgressionModel,
    SeatedRowProgressionModel,
)
from .models.base_model import FeatureEngineeringTransformer
from .personalized_prediction import (
	CalibrationConfig,
	predict_with_user_calibration,
)
from .rule_based import rule_based_progression
from .utils.user_personalization import PersonalizationRegistry
from .data_store import DataStore


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="pwp", description="Personalized Workout Progression System")
	sub = parser.add_subparsers(dest="command", required=True)

	sub.add_parser("preprocess", help="Preprocess baseline datasets and write processed CSVs")
	mi_parser = sub.add_parser("mi", help="Compute and print mutual information scores on preprocessed baseline data")
	mi_parser.add_argument("--target", default="set_volume", help="Target column for MI (default: set_volume)")
	
	train_compounds_parser = sub.add_parser("train-compounds", help="Train compound progression models on PPL data")
	train_compounds_parser.add_argument(
		"--ppl-dir",
		default="data/processed/PPL_data",
		help="Directory containing leg_workouts.csv, push_workouts.csv, pull_workouts.csv"
	)
	train_compounds_parser.add_argument(
		"--output-dir",
		default="models/compounds",
		help="Directory to save trained compound models"
	)

	predict_parser = sub.add_parser("predict", help="Predict with optional per-user calibration and rule-based fallback")
	predict_parser.add_argument("--model-path", required=True, help="Path to trained model pickle")
	predict_parser.add_argument("--user-id", required=True, help="User identifier")
	predict_parser.add_argument("--compound", required=True, help="Compound name (e.g., squat, bench_press)")
	predict_parser.add_argument("--history-csv", required=True, help="CSV with past workouts including load_delta")
	predict_parser.add_argument("--future-csv", required=True, help="CSV with upcoming session rows to predict")
	predict_parser.add_argument("--min-samples", type=int, default=8, help="Min samples to fit calibration")
	predict_parser.add_argument("--refit-every", type=int, default=10, help="Refit cadence for calibration")
	predict_parser.add_argument("--calib-window", type=int, default=32, help="Calibration history window")
	predict_parser.add_argument("--bench-fallback", action="store_true", help="Always use rule-based fallback for bench")
	predict_parser.add_argument("--debug", action="store_true", help="Print diagnostics (input validation, shapes)")
	predict_parser.add_argument("--db-path", default="data/user_data.db", help="Path to SQLite DB for logging")

	refresh_cal_parser = sub.add_parser("refresh-calibration", help="Recompute per-user affine calibration on new history")
	refresh_cal_parser.add_argument("--model-path", required=True, help="Path to trained model pickle")
	refresh_cal_parser.add_argument("--user-id", required=True, help="User identifier")
	refresh_cal_parser.add_argument("--compound", required=True, help="Compound name (e.g., squat, bench_press)")
	refresh_cal_parser.add_argument("--history-csv", required=True, help="CSV with past workouts including load_delta")
	refresh_cal_parser.add_argument("--min-samples", type=int, default=8)
	refresh_cal_parser.add_argument("--calib-window", type=int, default=64)
	refresh_cal_parser.add_argument("--gain-low", type=float, default=0.6)
	refresh_cal_parser.add_argument("--gain-high", type=float, default=1.4)
	refresh_cal_parser.add_argument("--db-path", default="data/user_data.db", help="Path to SQLite DB for logging")
	refresh_cal_parser.add_argument("--debug", action="store_true")

	args = parser.parse_args(argv)

	if args.command == "preprocess":
		from .data_pipeline import preprocess_baseline_data
		df4k, df721, df_all = preprocess_baseline_data(write_outputs=True)
		print(f"Processed 4k rows: {len(df4k)}")
		print(f"Processed 721 rows: {len(df721)}")
		print(f"Combined processed rows: {len(df_all)}")
		return 0
	elif args.command == "stats":
		from .data_pipeline import preprocess_baseline_data, dataset_stats
		df4k, df721, df_all = preprocess_baseline_data(write_outputs=False)
		s4k = dataset_stats(df4k)
		s721 = dataset_stats(df721)
		sall = dataset_stats(df_all)
		print("4k:", s4k)
		print("721:", s721)
		print("ALL:", sall)
		return 0
	elif args.command == "mi":
		target = args.target
		from .data_pipeline import preprocess_baseline_data
		_, _, df_all = preprocess_baseline_data(write_outputs=False)
		if target not in df_all.columns:
			print(f"Target column '{target}' not in data columns")
			return 1

		df = df_all.copy()
		# Convert datetime columns to numeric seconds since epoch
		datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
		for col in datetime_cols:
			df[col] = df[col].view("int64") // 1_000_000_000

		# After conversion, keep only numeric columns
		num_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
		if target not in num_cols_all:
			print(f"Target '{target}' is not numeric after conversion; cannot compute MI regression.")
			return 1
		# Features exclude target
		num_feature_cols = [c for c in num_cols_all if c != target]
		if not num_feature_cols:
			print("No numeric feature columns available for MI computation.")
			return 1

		# Drop NA across selected columns
		df_clean = df.dropna(subset=num_feature_cols + [target])
		X_clean = df_clean[num_feature_cols]
		y_clean = df_clean[target]

		mi_scores = mutual_info_regression(
			X_clean,
			y_clean,
			discrete_features=False,
			random_state=0,
		)
		mi_series = pd.Series(mi_scores, index=X_clean.columns, name="MI Scores").sort_values(ascending=False)
		print(mi_series)
		return 0
	
	elif args.command == "train-compounds":
		ppl_dir = Path(args.ppl_dir)
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		
		train_compound_models(ppl_dir, output_dir)
		return 0
	elif args.command == "predict":
		return predict_entrypoint(args)
	elif args.command == "refresh-calibration":
		return refresh_calibration_entrypoint(args)

	return 0


def _load_model(model_path: Path):
	import joblib
	obj = joblib.load(model_path)
	# If saved via BaseModel.save(), payload is a dict with 'pipeline'
	if isinstance(obj, dict):
		pipe = obj.get("pipeline")
		if pipe is not None and hasattr(pipe, "predict"):
			return pipe
	return obj


def _prep_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
	return df.drop(columns=[target], errors="ignore")


def _validate_inputs(history: pd.DataFrame, future: pd.DataFrame) -> list[str]:
	missing = []
	required = ["weight", "reps"]
	for col in required:
		if col not in history.columns:
			missing.append(f"history missing '{col}'")
		if col not in future.columns:
			missing.append(f"future missing '{col}'")
	return missing


def predict_entrypoint(args) -> int:
	model_path = Path(args.model_path)
	if not model_path.exists():
		print(f"Model not found: {model_path}")
		return 1

	try:
		model = _load_model(model_path)
	except Exception as e:
		print(f"Failed to load model: {e}")
		return 1

	try:
		history = pd.read_csv(args.history_csv)
		future = pd.read_csv(args.future_csv)
	except Exception as e:
		print(f"Failed to read CSVs: {e}")
		return 1

	warnings = _validate_inputs(history, future)
	if warnings:
		for w in warnings:
			print(f"[WARN] {w}")
		if args.debug:
			print("[WARN] proceeding despite missing columns; downstream may fail")
	if args.debug:
		print(f"History rows: {len(history)}, Future rows: {len(future)}")
		print(f"History columns: {list(history.columns)}")
		print(f"Future columns: {list(future.columns)}")

	cfg = CalibrationConfig(
		min_samples=args.min_samples,
		refit_every=args.refit_every,
		calibration_window=args.calib_window,
	)
	registry = PersonalizationRegistry()
	store = DataStore(Path(args.db_path)) if args.db_path else None
	compound = args.compound.lower().strip()

	fitted = None
	raw_pred = None
	adj_pred = None

	# Apply ML prediction with calibration if allowed
	use_ml = not args.bench_fallback or compound != "bench_press"
	if use_ml:
		try:
			raw_pred, adj_pred, fitted = predict_with_user_calibration(
				model=model,
				user_id=args.user_id,
				compound=compound,
				future_df=future,
				registry=registry,
				history=history,
				config=cfg,
			)
		except Exception as e:
			print(f"ML prediction failed, falling back to rules: {e}")
			use_ml = False

	# Rule-based fallback (bench or ML failure)
	fallback_reason = None
	fallback_value = None
	if (compound == "bench_press") or (not use_ml):
		last = history.tail(1)
		lw = float(last["weight"].iloc[0]) if "weight" in last.columns and not last.empty else None
		lr = float(last["reps"].iloc[0]) if "reps" in last.columns and not last.empty else None
		rpe = float(last["rpe"].iloc[0]) if "rpe" in last.columns and not last.empty else None
		sugg = rule_based_progression(last_weight=lw, last_reps=lr, last_rpe=rpe)
		fallback_value = sugg.suggested_weight
		fallback_reason = sugg.reason

	# Reporting
	if raw_pred is not None and adj_pred is not None and use_ml:
		print("ML prediction (raw -> adjusted):")
		for r, a in zip(raw_pred, adj_pred):
			print(f"  {r:.2f} -> {a:.2f}")
		if fitted:
			print(f"Calibration fitted (a,b): {fitted}")
	else:
		print("Rule-based fallback used.")

	if fallback_value is not None:
		print(f"Fallback suggested top-set weight: {fallback_value:.2f} ({fallback_reason})")

	# User-facing summary with simple confidence heuristic
	if use_ml and adj_pred is not None:
		confidence = "high" if fitted else "medium"
		top = float(adj_pred[0]) if len(adj_pred) else None
		print(f"Suggested top-set (ML, {confidence} confidence): {top:.2f}" if top is not None else "No prediction available")
	elif fallback_value is not None:
		print(f"Suggested top-set (rule, low confidence): {fallback_value:.2f}")

	# Log predictions and calibration
	if store:
		try:
			if use_ml and raw_pred is not None and adj_pred is not None:
				store.log_predictions(
					user_id=args.user_id,
					compound=compound,
					raw=raw_pred,
					adjusted=adj_pred,
					source="ml",
				)
			elif fallback_value is not None:
				store.log_predictions(
					user_id=args.user_id,
					compound=compound,
					raw=None,
					adjusted=[fallback_value],
					source="rule",
				)
			if fitted:
				meta = registry.get_or_create(args.user_id).calibration_meta.get(compound, {})
				store.upsert_calibration(
					user_id=args.user_id,
					compound=compound,
					a=fitted[0],
					b=fitted[1],
					last_calibrated_size=int(meta.get("last_calibrated_size", 0)),
					runs=int(meta.get("runs", 0)),
				)
		except Exception as e:
			if args.debug:
				print(f"[WARN] failed to log to datastore: {e}")

	# Persist personalization updates if any
	if fitted:
		registry.save(args.user_id)
		if args.debug:
			print(f"Saved personalization for {args.user_id}")

	return 0


def refresh_calibration_entrypoint(args) -> int:
	model_path = Path(args.model_path)
	if not model_path.exists():
		print(f"Model not found: {model_path}")
		return 1
	try:
		model = _load_model(model_path)
	except Exception as e:
		print(f"Failed to load model: {e}")
		return 1

	try:
		history = pd.read_csv(args.history_csv)
	except Exception as e:
		print(f"Failed to read history CSV: {e}")
		return 1

	target_col = "load_delta"
	if target_col not in history.columns:
		print(f"History is missing required target column '{target_col}'")
		return 1

	hist_use = history.dropna(subset=[target_col]).tail(args.calib_window)
	if len(hist_use) < args.min_samples:
		print(f"Not enough rows for calibration (need {args.min_samples}, have {len(hist_use)})")
		return 1

	X_hist = _prep_features(hist_use, target_col)
	y_true = hist_use[target_col].values
	try:
		y_pred = model.predict(X_hist)
	except Exception as e:
		print(f"Prediction failed on history: {e}")
		return 1

	registry = PersonalizationRegistry()
	up = registry.get_or_create(args.user_id)
	ab = up.calibrate_affine(
		compound=args.compound,
		y_true=y_true,
		y_pred=y_pred,
		min_samples=args.min_samples,
		gain_bounds=(args.gain_low, args.gain_high),
	)
	if ab is None:
		print("Calibration not updated (insufficient samples)")
		return 1

	registry.save(args.user_id)
	print(f"Updated calibration for {args.user_id}/{args.compound}: a={ab[0]:.3f}, b={ab[1]:.3f}")

	if args.db_path:
		try:
			store = DataStore(Path(args.db_path))
			meta = up.calibration_meta.get(args.compound, {})
			store.upsert_calibration(
				user_id=args.user_id,
				compound=args.compound,
				a=ab[0],
				b=ab[1],
				last_calibrated_size=int(meta.get("last_calibrated_size", len(hist_use))),
				runs=int(meta.get("runs", 1)),
			)
			if args.debug:
				print("Calibration persisted to DB")
		except Exception as e:
			if args.debug:
				print(f"[WARN] failed to persist calibration: {e}")

	return 0


def train_compound_models(ppl_dir: Path, output_dir: Path) -> None:
	"""
	Train all 4 compound progression models on PPL data.
	
	Expects:
	- ppl_dir/leg_workouts.csv
	- ppl_dir/push_workouts.csv
	- ppl_dir/pull_workouts.csv
	
	Each CSV should contain top-set data (or we filter to top sets).
	Saves trained models to output_dir.
	"""
	from .models.compound_models import add_periodization_features
	
	model_configs = [
		("leg_workouts.csv", SquatProgressionModel, "Squat"),
		("push_workouts.csv", BenchPressProgressionModel, "Bench Press"),
		("pull_workouts.csv", LatPulldownProgressionModel, "Lat Pulldown"),
		("pull_workouts.csv", SeatedRowProgressionModel, "Seated Row"),
	]
	
	for csv_file, model_class, compound_name in model_configs:
		csv_path = ppl_dir / csv_file
		if not csv_path.exists():
			print(f"Warning: {csv_path} not found. Skipping {compound_name}.")
			continue
		
		print(f"\n{'='*60}")
		print(f"Training {compound_name} model...")
		print(f"{'='*60}")
		
		# Load data
		df = pd.read_csv(csv_path)
		print(f"Loaded {len(df)} rows from {csv_file}")
		
		# Filter to top sets if not already filtered
		if 'set_order' in df.columns:
			df_top = df[df['set_order'] == 1].copy()
			print(f"Filtered to {len(df_top)} top sets")
		else:
			df_top = df.copy()
			print(f"Using all {len(df_top)} rows (assuming pre-filtered to top sets)")
		
		if len(df_top) < 30:
			print(f"Warning: Only {len(df_top)} rows. Too little data. Skipping.")
			continue
		
		# Prepare data: compute load_delta
		df_top = df_top.sort_values(by='date').reset_index(drop=True)
		df_top['load_delta'] = df_top['weight'].diff().shift(-1)
		df_top = df_top.dropna(subset=['load_delta'])
		
		print(f"After load_delta computation: {len(df_top)} rows")
		
		# Prepare features using the compound model's method
		X, y = model_class.prepare_compound_data(
			df_top,
			target_col='load_delta',
			drop_cols=['date', 'exercise_normalized', 'workout_name']
		)
		
		print(f"Features shape: {X.shape}")
		print(f"Target shape: {y.shape}")
		
		# Train model
		model = model_class()
		model.fit(X, y)
		
		# Save model (per-user default location), then copy to output_dir
		import shutil
		model_name = compound_name.lower().replace(" ", "_")
		saved_path = model.save("Rzu")
		out_path = output_dir / f"{model_name}_model.pkl"
		out_path.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(saved_path, out_path)
		print(f"Saved model to {out_path}")
		
		# Quick validation
		y_pred = model.predict(X)
		from sklearn.metrics import mean_absolute_error, r2_score
		mae = mean_absolute_error(y, y_pred)
		r2 = r2_score(y, y_pred)
		print(f"Training MAE: {mae:.2f}, R²: {r2:.4f}")


if __name__ == "__main__":
	raise SystemExit(main())