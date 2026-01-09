"""Track model quality and determine when to enable ML predictions"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "user_data.db"
AUTH_DB_PATH = Path(__file__).parent.parent / "data" / "auth" / "app_users.db"

def calculate_mape(actual: list, predicted: list) -> float:
    """Calculate Mean Absolute Percentage Error"""
    if len(actual) == 0:
        return None
    
    errors = []
    for a, p in zip(actual, predicted):
        if a != 0:
            errors.append(abs((a - p) / a))
    
    return sum(errors) / len(errors) if errors else None

def update_model_quality(user_id: int, compound: str):
    """
    Calculate MAPE on recent 'normal' sessions only
    Compare to rule-based MAPE
    Auto-enable model if it's better
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    cur = conn.cursor()
    
    # Get recent predictions with COMPLETED status and 'normal' deviation reason
    cur.execute("""
        SELECT 
            recommended_weight as predicted,
            actual_weight as actual,
            prediction_source,
            logged_at
        FROM session_audit
        WHERE user_id = ? 
          AND compound = ? 
          AND prediction_status = 'complete'
          AND deviation_reason = 'normal'
        ORDER BY logged_at DESC
        LIMIT 15
    """, (user_id, compound))
    
    records = cur.fetchall()
    
    if len(records) < 10:
        # Not enough data to evaluate
        conn.close()
        return False
    
    # Split into model and rule-based predictions
    model_preds = [r['predicted'] for r in records if r['prediction_source'] == 'model']
    model_actuals = [r['actual'] for r in records if r['prediction_source'] == 'model']
    
    rule_preds = [r['predicted'] for r in records if r['prediction_source'] == 'rule_based']
    rule_actuals = [r['actual'] for r in records if r['prediction_source'] == 'rule_based']
    
    model_mape = None
    rule_mape = None
    
    if len(model_preds) >= 5:
        model_mape = calculate_mape(model_actuals, model_preds)
    
    if len(rule_preds) >= 5:
        rule_mape = calculate_mape(rule_actuals, rule_preds)
    
    # Determine if model should be enabled
    model_enabled = False
    if model_mape is not None and rule_mape is not None:
        # Enable if model beats rule-based by 15% AND is under 10% error
        if model_mape < rule_mape * 0.85 and model_mape < 0.10:
            model_enabled = True
    
    # Update model_quality table
    auth_conn = sqlite3.connect(AUTH_DB_PATH)
    auth_cur = auth_conn.cursor()
    
    auth_cur.execute("""
        UPDATE model_quality
        SET session_count = ?,
            model_mape = ?,
            rule_mape = ?,
            model_enabled = ?,
            last_updated = ?
        WHERE user_id = ? AND compound = ?
    """, (
        len(records),
        model_mape,
        rule_mape,
        1 if model_enabled else 0,
        datetime.now(),
        user_id,
        compound
    ))
    
    auth_conn.commit()
    auth_conn.close()
    conn.close()
    
    return model_enabled

def is_model_enabled(user_id: int, compound: str) -> bool:
    """Check if model is enabled for this user/compound"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    cur = conn.cursor()
    
    cur.execute(
        "SELECT model_enabled FROM model_quality WHERE user_id = ? AND compound = ?",
        (user_id, compound)
    )
    
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        return False
    
    return bool(row[0])

def get_session_count(user_id: int, compound: str) -> int:
    """Get session count from model_quality table"""
    conn = sqlite3.connect(AUTH_DB_PATH)
    cur = conn.cursor()
    
    cur.execute(
        "SELECT session_count FROM model_quality WHERE user_id = ? AND compound = ?",
        (user_id, compound)
    )
    
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        return 0
    
    return row[0] or 0
