"""Session logging and accuracy tracking"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

def log_session(
    user_id: int,
    user_data_path: str,
    compound: str,
    weight: float,
    reps: int,
    rpe: float,
    deviation_reason: str,
    recommended_weight: float,
    prediction_source: str
) -> int:
    """
    Log a new session to CSV and database
    
    Returns: prediction_id for later accuracy tracking
    """
    user_data_path = Path(user_data_path)
    
    # 1. Append to CSV
    csv_path = user_data_path / f"{Path(user_data_path).name}_{compound}_history.csv"
    
    # Calculate load_delta
    df_existing = pd.read_csv(csv_path)
    if len(df_existing) > 0:
        last_weight = df_existing.iloc[-1]['weight']
        load_delta = weight - last_weight
    else:
        load_delta = 0.0
    
    new_row = pd.DataFrame({
        'weight': [weight],
        'reps': [reps],
        'rpe': [rpe],
        'load_delta': [load_delta]
    })
    
    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
    df_updated.to_csv(csv_path, index=False)
    
    # 2. Log to database with PENDING status
    db_path = Path(__file__).parent.parent / "data" / "user_data.db"
    conn = sqlite3.connect(db_path)
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO session_audit 
        (user_id, compound, weight, reps, rpe, deviation_reason, 
         prediction_source, recommended_weight, prediction_status, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, compound, weight, reps, rpe, deviation_reason,
        prediction_source, recommended_weight, 'pending', datetime.now()
    ))
    
    prediction_id = cur.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id

def compute_accuracy_for_pending_predictions(user_id: int, user_data_path: str):
    """
    When user logs a new session, check if there are pending predictions
    from the previous session and mark them complete with accuracy
    """
    user_data_path = Path(user_data_path)
    db_path = Path(__file__).parent.parent / "data" / "user_data.db"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all compounds with predictions for this user
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT compound FROM session_audit WHERE user_id = ? AND prediction_status = 'pending'
    """, (user_id,))
    
    compounds = [row[0] for row in cur.fetchall()]
    
    for compound in compounds:
        # Get most recent pending prediction
        cur.execute("""
            SELECT id, recommended_weight, logged_at 
            FROM session_audit 
            WHERE user_id = ? AND compound = ? AND prediction_status = 'pending'
            ORDER BY logged_at DESC
            LIMIT 1
        """, (user_id, compound))
        
        pending_pred = cur.fetchone()
        if not pending_pred:
            continue
        
        # Get most recent actual session (just logged)
        csv_path = user_data_path / f"{Path(user_data_path).name}_{compound}_history.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
        
        actual_weight = df.iloc[-1]['weight']
        accuracy_delta = actual_weight - pending_pred['recommended_weight']
        
        # Mark prediction as COMPLETE and store accuracy
        cur.execute("""
            UPDATE session_audit 
            SET prediction_status = 'complete', actual_weight = ?, accuracy_delta = ?
            WHERE id = ?
        """, (actual_weight, accuracy_delta, pending_pred['id']))
    
    conn.commit()
    conn.close()

def get_session_count(user_data_path: str, compound: str) -> int:
    """Get number of sessions logged for a compound"""
    user_data_path = Path(user_data_path)
    csv_path = user_data_path / f"{Path(user_data_path).name}_{compound}_history.csv"
    
    if not csv_path.exists():
        return 0
    
    df = pd.read_csv(csv_path)
    return len(df)

def get_last_session(user_data_path: str, compound: str) -> Optional[Tuple[float, int, float]]:
    """
    Get last session (weight, reps, rpe) for a compound
    Returns: (weight, reps, rpe) or None if no data
    """
    user_data_path = Path(user_data_path)
    csv_path = user_data_path / f"{Path(user_data_path).name}_{compound}_history.csv"
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    
    last_row = df.iloc[-1]
    return (float(last_row['weight']), int(last_row['reps']), float(last_row['rpe']))
