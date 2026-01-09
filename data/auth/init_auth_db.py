"""Initialize authentication database with schema and test user"""
import sqlite3
from pathlib import Path

def init_auth_db():
    db_path = Path(__file__).parent / "app_users.db"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_data_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    # Create model_quality table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            compound TEXT NOT NULL,
            session_count INTEGER DEFAULT 0,
            model_mape REAL,
            rule_mape REAL,
            model_enabled BOOLEAN DEFAULT 0,
            last_updated TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, compound)
        )
    """)
    
    # Create session_audit table
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
    
    # Check if User2 already exists
    cur.execute("SELECT user_id FROM users WHERE username = 'User2'")
    if cur.fetchone() is None:
        # Add User2 test account
        repo_root = Path(__file__).parent.parent.parent
        user2_path = repo_root / "users" / "User2"
        
        cur.execute(
            "INSERT INTO users (username, password, user_data_path) VALUES (?, ?, ?)",
            ("User2", "password", str(user2_path))
        )
        user_id = cur.lastrowid
        
        # Add model_quality rows
        compounds = ["squat", "bench_press", "lat_pulldown", "seated_row"]
        for compound in compounds:
            cur.execute(
                "INSERT INTO model_quality (user_id, compound) VALUES (?, ?)",
                (user_id, compound)
            )
        
        conn.commit()
        print("[OK] Database initialized with User2 test account")
    else:
        # User2 exists, ensure User2 has all CSVs
        cur.execute("SELECT user_id, user_data_path FROM users WHERE username = 'User2'")
        user_id, user2_path = cur.fetchone()
        user2_path = Path(user2_path)
        
        # Create empty CSV files if they don't exist
        compounds = ["squat", "bench_press", "lat_pulldown", "seated_row"]
        for compound in compounds:
            csv_path = user2_path / f"User2_{compound}_history.csv"
            if not csv_path.exists():
                csv_path.write_text("weight,reps,rpe,load_delta\n")
        
        print("[OK] Database already initialized (ensured CSVs exist)")
    
    conn.close()

if __name__ == "__main__":
    init_auth_db()
