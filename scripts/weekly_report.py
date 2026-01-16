import pandas as pd
from src.config import PATHS
from src.reports import weekly_summary

if __name__ == "__main__":
    df = pd.read_csv(PATHS.PROCESSED)
    text = weekly_summary(df)

    PATHS.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = PATHS.OUTPUTS / "weekly_report.txt"
    out.write_text(text, encoding="utf-8")

    print(f"Wrote report: {out}")
    print("\n" + text)
