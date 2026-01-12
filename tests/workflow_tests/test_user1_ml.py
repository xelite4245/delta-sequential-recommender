#!/usr/bin/env python3
"""
Quick test to verify ML predictions are working for user1.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parents[2]
sys.path.insert(0, str(repo_root))

import pandas as pd
from src.recommendation_engine import get_recommendation
from src.model_quality import is_model_enabled
from src.auth import get_user_id


def test_user1_ml():
    """Test that user1 gets ML recommendations"""
    user_id = get_user_id("user1")
    assert user_id is not None, "user1 not found in auth DB"
    user_data_path = repo_root / "users" / "user1"

    compounds = ["squat", "bench_press", "lat_pulldown"]

    for compound in compounds:
        ml_enabled = is_model_enabled(user_id, compound)
        assert ml_enabled, f"ML not enabled for {compound}"

        csv_path = user_data_path / f"user1_{compound}_history.csv"
        history = pd.read_csv(csv_path)
        assert len(history) >= 1, f"No history for {compound}"

        session_count = len(history)
        last_row = history.iloc[-1]

        rec, source, reason = get_recommendation(
            user_id=user_id,
            user_data_path=str(user_data_path),
            compound=compound,
            last_weight=last_row['weight'],
            last_reps=int(last_row['reps']),
            last_rpe=last_row.get('rpe', 8.0),
            session_count=session_count
        )

        assert rec is not None, f"No recommendation for {compound}: {reason}"
        assert source in {"model", "rule_based"}, f"Unexpected source {source}"
