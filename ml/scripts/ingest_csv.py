import sys
import pandas as pd
from src.db import insert_events, init_db

def main(csv_path: str) -> None:
    init_db()
    df = pd.read_csv(csv_path)
    required = ["timestamp", "event_type", "carbs_g", "med_name", "med_units", "glucose_mgdl", "notes"]
    for c in required:
        if c not in df.columns:
            df[c] = None

    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["timestamp"]),
            str(r["event_type"]),
            None if pd.isna(r["carbs_g"]) else float(r["carbs_g"]),
            None if pd.isna(r["med_name"]) else str(r["med_name"]),
            None if pd.isna(r["med_units"]) else float(r["med_units"]),
            None if pd.isna(r["glucose_mgdl"]) else float(r["glucose_mgdl"]),
            None if pd.isna(r["notes"]) else str(r["notes"]),
        ))

    insert_events(rows)
    print(f"Ingested {len(rows)} rows from {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/ingest_csv.py <path_to_csv>")
    main(sys.argv[1])
