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


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="pwp", description="Personalized Workout Progression System")
	sub = parser.add_subparsers(dest="command", required=True)

	sub.add_parser("preprocess", help="Preprocess baseline datasets and write processed CSVs")
	mi_parser = sub.add_parser("mi", help="Compute and print mutual information scores on preprocessed baseline data")
	mi_parser.add_argument("--target", default="set_volume", help="Target column for MI (default: set_volume)")

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

	return 0


if __name__ == "__main__":
	raise SystemExit(main())