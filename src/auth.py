"""User authentication and account management"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = Path(__file__).parent.parent / "data" / "auth" / "app_users.db"

class AuthError(Exception):
    """Custom exception for auth errors"""
    pass

def login(username: str, password: str) -> Tuple[int, str]:
    """
    Authenticate user and return (user_id, user_data_path)
    
    Raises AuthError if credentials invalid or path doesn't exist
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute(
        "SELECT user_id, password, user_data_path FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        raise AuthError(f"User '{username}' not found")
    
    user_id, stored_password, user_data_path = row
    
    if stored_password != password:
        raise AuthError("Invalid password")
    
    # Verify user path exists
    path = Path(user_data_path)
    if not path.exists():
        raise AuthError(f"User data path does not exist: {user_data_path}")
    
    # Update last_login
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    return user_id, user_data_path

def register(username: str, password: str) -> Tuple[int, str]:
    """
    Register new user and create user directory structure
    
    Returns (user_id, user_data_path)
    Raises AuthError if username already exists
    """
    # Check username doesn't exist
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    if cur.fetchone() is not None:
        conn.close()
        raise AuthError(f"Username '{username}' already exists")
    conn.close()
    
    # Create user directory
    repo_root = Path(__file__).parent.parent
    user_data_path = repo_root / "users" / username
    user_data_path.mkdir(parents=True, exist_ok=True)
    
    # Create empty CSV files for each compound
    compounds = ["squat", "bench_press", "lat_pulldown", "seated_row"]
    for compound in compounds:
        csv_path = user_data_path / f"{username}_{compound}_history.csv"
        csv_path.write_text("weight,reps,rpe,load_delta\n")
    
    # Create personalization.json
    personalization = {
        "scaling_factors": {c: 1.0 for c in compounds},
        "baseline_offsets": {c: 0.0 for c in compounds},
        "trend_modifiers": {c: {"trend": 0.0} for c in compounds},
        "calibration_meta": {
            c: {"last_calibrated_size": 0, "runs": 0}
            for c in compounds
        }
    }
    
    pers_path = user_data_path / "personalization.json"
    with open(pers_path, "w") as f:
        json.dump(personalization, f, indent=2)
    
    # Create plots directory
    plots_dir = user_data_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Insert user into database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO users (username, password, user_data_path) VALUES (?, ?, ?)",
        (username, password, str(user_data_path))
    )
    user_id = cur.lastrowid
    
    # Add model_quality rows for each compound
    for compound in compounds:
        cur.execute(
            "INSERT INTO model_quality (user_id, compound) VALUES (?, ?)",
            (user_id, compound)
        )
    
    conn.commit()
    conn.close()
    
    return user_id, str(user_data_path)

def get_user_id(username: str) -> Optional[int]:
    """Get user_id from username"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None
