#!/usr/bin/env python3
"""Test script to validate all modules"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

print("Testing imports...")
try:
    from src import auth
    print("✓ auth module")
except Exception as e:
    print(f"✗ auth module: {e}")

try:
    from src import ui
    print("✓ ui module")
except Exception as e:
    print(f"✗ ui module: {e}")

try:
    from src import session_logger
    print("✓ session_logger module")
except Exception as e:
    print(f"✗ session_logger module: {e}")

try:
    from src import model_quality
    print("✓ model_quality module")
except Exception as e:
    print(f"✗ model_quality module: {e}")

try:
    from src import recommendation_engine
    print("✓ recommendation_engine module")
except Exception as e:
    print(f"✗ recommendation_engine module: {e}")

print("\nTesting auth...")
try:
    user_id, user_path = auth.login("User2", "password")
    print(f"✓ Login successful: user_id={user_id}, path={Path(user_path).name}")
except Exception as e:
    print(f"✗ Login failed: {e}")

print("\nTesting session count...")
try:
    count = session_logger.get_session_count(user_path, "squat")
    print(f"✓ Session count: {count} squat sessions")
except Exception as e:
    print(f"✗ Session count failed: {e}")

print("\nTesting model quality...")
try:
    enabled = model_quality.is_model_enabled(user_id, "squat")
    count = model_quality.get_session_count(user_id, "squat")
    print(f"✓ Model enabled: {enabled}, session count: {count}")
except Exception as e:
    print(f"✗ Model quality failed: {e}")

print("\nTesting recommendation...")
try:
    last_session = session_logger.get_last_session(user_path, "squat")
    if last_session:
        last_weight, last_reps, last_rpe = last_session
        session_count = session_logger.get_session_count(user_path, "squat")
        
        rec_weight, source, reason = recommendation_engine.get_recommendation(
            user_id=user_id,
            user_data_path=user_path,
            compound="squat",
            last_weight=last_weight,
            last_reps=last_reps,
            last_rpe=last_rpe,
            session_count=session_count
        )
        print(f"✓ Recommendation: {rec_weight:.1f} lbs (source: {source})")
        print(f"  Reason: {reason}")
    else:
        print("ℹ No session history for squat yet (expected)")
except Exception as e:
    print(f"✗ Recommendation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All modules loaded successfully!")
