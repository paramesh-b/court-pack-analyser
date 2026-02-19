import csv
import os
from datetime import datetime

LOG_FILE = "data/analysis_log.csv"

HEADERS = [
    "timestamp", "claimant", "hire_company", "vehicle_category",
    "region", "hire_duration_days", "daily_rate_charged",
    "benchmark_daily_rate", "rate_deviation_pct",
    "risk_level", "recommendation", "total_claim"
]

def log_result(result: dict):
    """Append analysis result to the audit log CSV."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{k: result.get(k, "") for k in HEADERS if k != "timestamp"}
        })