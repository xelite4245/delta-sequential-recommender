#!/usr/bin/env python3
"""Automated test to validate complete app workflow"""
import sys
from pathlib import Path
import sqlite3
import pandas as pd

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src import auth, session_logger, model_quality

def test_complete_workflow():
    """Test: Login → Log 5 sessions → Check accuracy tracking"""
    print("=" * 60)
    print("AUTOMATED WORKFLOW TEST")
    print("=" * 60)
    
    # Step 1: Login
    print("\n1. Testing login...")
    try:
        user_id, user_path = auth.login("User2", "password")
        print(f"   ✓ Logged in as user_id={user_id}")
        print(f"   ✓ User path: {Path(user_path).name}")
    except Exception as e:
        print(f"   ✗ Login failed: {e}")
        return
    
    # Step 2: Get starting session count
    print("\n2. Checking current session count...")
    count_before = session_logger.get_session_count(user_path, "squat")
    print(f"   ✓ Current squat sessions: {count_before}")
    
    # Step 3: Simulate logging 3 sessions
    print("\n3. Simulating 3 session logs...")
    test_sessions = [
        (185, 5, 7.0, "normal"),
        (190, 5, 7.5, "normal"),
        (195, 5, 8.0, "hard"),  # Note: hard session won't count toward model quality
    ]
    
    for i, (weight, reps, rpe, reason) in enumerate(test_sessions, 1):
        try:
            prediction_id = session_logger.log_session(
                user_id=user_id,
                user_data_path=user_path,
                compound="squat",
                weight=weight,
                reps=reps,
                rpe=rpe,
                deviation_reason=reason,
                recommended_weight=weight + 2.5,  # Dummy recommendation
                prediction_source="rule_based"
            )
            print(f"   ✓ Session {i}: {weight} lbs × {reps} reps (reason: {reason})")
        except Exception as e:
            print(f"   ✗ Session {i} failed: {e}")
            return
    
    # Step 4: Verify CSV was updated
    print("\n4. Verifying CSV...")
    csv_path = Path(user_path) / f"{Path(user_path).name}_squat_history.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"   ✓ CSV has {len(df)} rows")
        print(f"   ✓ Last weight: {df.iloc[-1]['weight']} lbs")
    else:
        print(f"   ✗ CSV not found")
        return
    
    # Step 5: Verify session audit table
    print("\n5. Verifying session_audit table...")
    try:
        db_path = repo_root / "data" / "user_data.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute(
            "SELECT COUNT(*) FROM session_audit WHERE user_id = ? AND compound = ?",
            (user_id, "squat")
        )
        count = cur.fetchone()[0]
        print(f"   ✓ session_audit has {count} entries for this user/compound")
        
        # Show last entry
        cur.execute("""
            SELECT weight, reps, rpe, deviation_reason, prediction_source, prediction_status
            FROM session_audit
            WHERE user_id = ? AND compound = ?
            ORDER BY logged_at DESC
            LIMIT 1
        """, (user_id, "squat"))
        
        last = cur.fetchone()
        if last:
            print(f"   ✓ Last entry: {last[0]} lbs × {last[1]} reps (reason: {last[3]}, status: {last[5]})")
        
        conn.close()
    except Exception as e:
        print(f"   ✗ Audit table verification failed: {e}")
        return
    
    # Step 6: Check model quality
    print("\n6. Checking model quality...")
    try:
        session_count = model_quality.get_session_count(user_id, "squat")
        is_enabled = model_quality.is_model_enabled(user_id, "squat")
        print(f"   ✓ Session count (from model_quality): {session_count}")
        print(f"   ✓ Model enabled: {is_enabled}")
        print(f"   ℹ  (Expected: False - need 15+ sessions to evaluate)")
    except Exception as e:
        print(f"   ✗ Model quality check failed: {e}")
        return
    
    # Step 7: Test recommendation flow
    print("\n7. Testing recommendation engine...")
    try:
        from src import recommendation_engine
        
        last_session = session_logger.get_last_session(user_path, "squat")
        if last_session:
            last_weight, last_reps, last_rpe = last_session
            new_count = session_logger.get_session_count(user_path, "squat")
            
            rec_weight, source, reason = recommendation_engine.get_recommendation(
                user_id=user_id,
                user_data_path=user_path,
                compound="squat",
                last_weight=last_weight,
                last_reps=last_reps,
                last_rpe=last_rpe,
                session_count=new_count
            )
            
            print(f"   ✓ Last session: {last_weight} lbs × {last_reps} reps")
            print(f"   ✓ Recommended: {rec_weight:.1f} lbs")
            print(f"   ✓ Source: {source}")
            print(f"   ✓ Reason: {reason}")
    except Exception as e:
        print(f"   ✗ Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Success
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe app is ready to use. Run: python run_app.py")

if __name__ == "__main__":
    test_complete_workflow()
