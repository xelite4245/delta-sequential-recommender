#!/usr/bin/env python3
"""
Enable ML model for user1 by populating model_quality table.
"""
import sys
import sqlite3
from pathlib import Path

# Go up 3 levels from tests/user1_tests/ to repo root
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))


def enable_ml_for_user1():
    """Enable ML predictions for user1 on all compounds"""
    
    print("=" * 70)
    print("ENABLING ML FOR USER1")
    print("=" * 70)
    
    auth_db_path = repo_root / "data" / "auth" / "app_users.db"
    
    try:
        # Get user1's ID
        conn = sqlite3.connect(auth_db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT user_id FROM users WHERE username = ?", ("user1",))
        row = cur.fetchone()
        
        if row is None:
            print("✗ User user1 not found!")
            conn.close()
            return False
        
        user_id = row[0]
        print(f"✓ Found user1 with ID: {user_id}")
        
        # Create model_quality table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_quality (
                user_id INTEGER NOT NULL,
                compound TEXT NOT NULL,
                session_count INTEGER DEFAULT 0,
                model_mape REAL,
                rule_mape REAL,
                model_enabled INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, compound)
            )
        """)
        conn.commit()
        print("✓ model_quality table ready")
        
        # Enable ML for each compound
        compounds = ['squat', 'bench_press', 'lat_pulldown', 'seated_row']
        
        for compound in compounds:
            print(f"\n  Setting {compound} to ML-enabled...")
            
            # Upsert into model_quality
            cur.execute("""
                INSERT OR REPLACE INTO model_quality 
                (user_id, compound, session_count, model_mape, rule_mape, model_enabled)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, compound, 14, 0.08, 0.15, 1))
            
            conn.commit()
            print(f"    ✓ {compound} enabled with session_count=14, model_mape=0.08, model_enabled=1")
        
        conn.close()
        
        print("\n" + "=" * 70)
        print("✓ ML ENABLED FOR USER1")
        print("=" * 70)
        print("\nuser1 will now use ML predictions instead of rule-based!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = enable_ml_for_user1()
    sys.exit(0 if success else 1)
