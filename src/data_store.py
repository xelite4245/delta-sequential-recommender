"""Lightweight SQLite persistence for per-user predictions and calibration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional


PRED_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    compound TEXT NOT NULL,
    session_index INTEGER,
    predicted_raw REAL,
    predicted_adjusted REAL,
    actual REAL,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CALIB_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calibrations (
    user_id TEXT NOT NULL,
    compound TEXT NOT NULL,
    a REAL,
    b REAL,
    last_calibrated_size INTEGER,
    runs INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, compound)
);
"""


class DataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(PRED_TABLE_SQL)
            cur.execute(CALIB_TABLE_SQL)
            conn.commit()

    def log_predictions(
        self,
        *,
        user_id: str,
        compound: str,
        raw: Optional[Iterable[float]],
        adjusted: Iterable[float],
        actual: Optional[Iterable[float]] = None,
        source: str = "ml",
    ) -> None:
        raw_list = list(raw) if raw is not None else [None] * len(adjusted)
        adj_list = list(adjusted)
        act_list = list(actual) if actual is not None else [None] * len(adj_list)
        with self._connect() as conn:
            cur = conn.cursor()
            for idx, (r, a, act) in enumerate(zip(raw_list, adj_list, act_list)):
                cur.execute(
                    """
                    INSERT INTO predictions (user_id, compound, session_index, predicted_raw, predicted_adjusted, actual, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, compound, idx, r, a, act, source),
                )
            conn.commit()

    def upsert_calibration(
        self,
        *,
        user_id: str,
        compound: str,
        a: float,
        b: float,
        last_calibrated_size: int,
        runs: int,
    ) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO calibrations (user_id, compound, a, b, last_calibrated_size, runs)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, compound) DO UPDATE SET
                    a = excluded.a,
                    b = excluded.b,
                    last_calibrated_size = excluded.last_calibrated_size,
                    runs = excluded.runs,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, compound, a, b, last_calibrated_size, runs),
            )
            conn.commit()
