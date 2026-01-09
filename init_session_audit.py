import sqlite3
from pathlib import Path

db_path = Path("data/user_data.db")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get existing tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cur.fetchall()]
print("Existing tables:", tables)

# Create session_audit table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS session_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        compound TEXT NOT NULL,
        weight REAL,
        reps INTEGER,
        rpe REAL,
        deviation_reason TEXT,
        prediction_source TEXT,
        recommended_weight REAL,
        actual_weight REAL,
        accuracy_delta REAL,
        prediction_status TEXT DEFAULT 'pending',
        logged_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
""")

conn.commit()

# Verify
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cur.fetchall()]
print("Tables after creation:", tables)

conn.close()
print("[OK] session_audit table created")
