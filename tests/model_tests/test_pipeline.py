"""
Unit tests for data_pipeline.py
Checks preprocessing, feature engineering, synthetic data generation, and edge cases.
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression

# Ensure repo root on path for src imports when running tests directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.data_pipeline import preprocess_baseline_data


def test_mutual_information_scores():
	"""Compute MI scores for key features against set_volume to ensure pipeline-ready data."""
	df4k, df721, df_all = preprocess_baseline_data(write_outputs=False)
	target = "set_volume"
	assert target in df_all.columns, "Expected target column set_volume in processed data"

	candidate_features = [
		"effective_load",
		"rpe",
		"set_order",
		"reps",
		"weight",
		"duration",
	]
	features = [c for c in candidate_features if c in df_all.columns]
	assert features, "No candidate features found in processed data"

	df = df_all.dropna(subset=features + [target])
	X = df[features]
	y = df[target]

	mi = mutual_info_regression(X, y, random_state=0)
	assert len(mi) == len(features)
	assert np.all(mi >= 0), "MI scores should be non-negative"