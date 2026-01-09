import sqlite3
from pathlib import Path

db_path = Path("data/user_data.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== PREDICTIONS TABLE ===")
cur = conn.cursor()
cur.execute("SELECT user_id, compound, session_index, predicted_raw, predicted_adjusted, source, created_at FROM predictions LIMIT 3")
for row in cur:
    raw_str = f"{row['predicted_raw']:.2f}" if row['predicted_raw'] is not None else "NULL"
    adj_str = f"{row['predicted_adjusted']:.2f}" if row['predicted_adjusted'] is not None else "NULL"
    print(f"{row['user_id']} | {row['compound']} | session {row['session_index']} | raw: {raw_str} | adj: {adj_str} | src: {row['source']}")

print("\n=== CALIBRATIONS TABLE ===")
cur = conn.cursor()
cur.execute("SELECT user_id, compound, a, b, last_calibrated_size, runs FROM calibrations")
for row in cur:
    print(f"{row['user_id']} | {row['compound']} | a={row['a']:.3f}, b={row['b']:.3f} | last_size: {row['last_calibrated_size']} | runs: {row['runs']}")

conn.close()
