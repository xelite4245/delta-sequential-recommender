"""
Command-line interface for the application.
Allows users to log workouts, get predictions, and view progression stats.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from .data_pipeline import preprocess_baseline_data, dataset_stats
from .models.compound_models import (
    SquatProgressionModel,
    BenchPressProgressionModel,
    LatPulldownProgressionModel,
    SeatedRowProgressionModel,
)
from .models.base_model import FeatureEngineeringTransformer


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

	args = parser.parse_args(argv)

	if args.command == "preprocess":
		df4k, df721, df_all = preprocess_baseline_data(write_outputs=True)
		print(f"Processed 4k rows: {len(df4k)}")
		print(f"Processed 721 rows: {len(df721)}")
		print(f"Combined processed rows: {len(df_all)}")
		return 0
	elif args.command == "stats":
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
		
		# Save model
		model_name = compound_name.lower().replace(" ", "_")
		model.save(f"Rzu", output_dir / f"{model_name}_model.pkl")
		print(f"Saved model to {output_dir / f'{model_name}_model.pkl'}")
		
		# Quick validation
		y_pred = model.predict(X)
		from sklearn.metrics import mean_absolute_error, r2_score
		mae = mean_absolute_error(y, y_pred)
		r2 = r2_score(y, y_pred)
		print(f"Training MAE: {mae:.2f}, R²: {r2:.4f}")


if __name__ == "__main__":
	raise SystemExit(main())