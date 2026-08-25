import pandas as pd

def weekly_summary(df_examples: pd.DataFrame) -> str:
    """
    Create a simple weekly insight summary (non-medical).
    """
    if df_examples.empty:
        return "No data available."

    df = df_examples.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    spike_rate = df["label_spike"].mean() * 100.0
    avg_glucose = df["glucose_mgdl"].mean()
    worst_hour = df.groupby("hour")["label_spike"].mean().sort_values(ascending=False).head(1)
    worst_hour_str = f"{int(worst_hour.index[0])}:00" if len(worst_hour) else "N/A"

    top_carbs = df.sort_values("last_meal_carbs", ascending=False).head(5)[["timestamp", "last_meal_carbs", "glucose_mgdl"]]

    lines = []
    lines.append("Weekly Insights (Informational Only)")
    lines.append(f"- Avg glucose: {avg_glucose:.1f} mg/dL")
    lines.append(f"- Spike-event rate (>= threshold): {spike_rate:.1f}%")
    lines.append(f"- Hour with highest spike rate: {worst_hour_str}")
    lines.append("")
    lines.append("Top high-carb-linked readings (for awareness):")
    for _, r in top_carbs.iterrows():
        lines.append(f"  - {r['timestamp']}: carbs={r['last_meal_carbs']:.0f}g, glucose={r['glucose_mgdl']:.0f} mg/dL")

    lines.append("")
    lines.append("Disclaimer: This report provides informational pattern insights only and is not medical advice.")
    return "\n".join(lines)
