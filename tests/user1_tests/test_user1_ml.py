#!/usr/bin/env python3
"""
Quick test to verify ML predictions are working for user1.
"""
import sys
from pathlib import Path

# Go up 3 levels from tests/user1_tests/ to repo root
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from src.recommendation_engine import get_recommendation
from src.model_quality import is_model_enabled
from src.auth import get_user_id


def test_user1_ml():
    """Test that user1 gets ML recommendations"""
    
    print("=" * 70)
    print("TESTING ML PREDICTIONS FOR USER1")
    print("=" * 70)
    
    user_id = get_user_id("user1")
    if user_id is None:
        print("✗ user1 not found in auth DB")
        return
    user_data_path = repo_root / "users" / "user1"
    
    compounds = ["squat", "bench_press", "lat_pulldown"]
    
    for compound in compounds:
        print(f"\n{compound.upper()}:")
        print("-" * 70)
        
        # Check if ML is enabled
        ml_enabled = is_model_enabled(user_id, compound)
        print(f"  ML Enabled: {ml_enabled}")
        
        # Load history
        csv_path = user_data_path / f"user1_{compound}_history.csv"
        history = pd.read_csv(csv_path)
        
        session_count = len(history)
        last_row = history.iloc[-1]
        
        print(f"  Sessions: {session_count}")
        print(f"  Last workout: {last_row['weight']} lbs × {last_row['reps']} reps (RPE {last_row['rpe']})")
        
        # Get recommendation
        try:
            recommended_weight, source, reason = get_recommendation(
                user_id=user_id,
                user_data_path=str(user_data_path),
                compound=compound,
                last_weight=last_row['weight'],
                last_reps=int(last_row['reps']),
                last_rpe=last_row['rpe'],
                session_count=session_count
            )
            
            if recommended_weight is not None:
                print(f"  Recommendation: {recommended_weight:.1f} lbs")
                print(f"  Source: {source}")
                print(f"  Reason: {reason}")
                
                # Check if it's using ML
                if source == "model":
                    print(f"  ✓ SUCCESS: Using ML prediction!")
                else:
                    print(f"  ⚠ WARNING: Using {source} instead of ML")
            else:
                print(f"  ✗ No recommendation (insufficient data)")
                
        except Exception as e:
            print(f"  ✗ Error getting recommendation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_user1_ml()
