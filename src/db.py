import sqlite3
from typing import Iterable, Optional, Tuple, Any
from .config import PATHS

def connect() -> sqlite3.Connection:
    return sqlite3.connect(PATHS.DB)

def init_db() -> None:
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                carbs_g REAL,
                med_name TEXT,
                med_units REAL,
                glucose_mgdl REAL,
                notes TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")

def insert_events(rows: Iterable[Tuple[Any, ...]]) -> None:
    with connect() as con:
        con.executemany(
            """
            INSERT INTO events (timestamp, event_type, carbs_g, med_name, med_units, glucose_mgdl, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

def fetch_all_events() -> list[tuple]:
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            """
            SELECT timestamp, event_type, carbs_g, med_name, med_units, glucose_mgdl, notes
            FROM events
            ORDER BY timestamp ASC
            """
        )
        return cur.fetchall()
