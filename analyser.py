
import os
import json
import re
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_hire_rates(csv_path: str = "data/hire_rates.csv") -> pd.DataFrame:
    """Load benchmark hire rates from CSV."""
    return pd.read_csv(csv_path)

def get_benchmark_rate(vehicle_category: str, region: str, rates_df: pd.DataFrame) -> float:
    """Find the benchmark daily rate for a given vehicle and region."""
    match = rates_df[
        (rates_df["vehicle_category"].str.contains(vehicle_category, case=False, na=False)) &
        (rates_df["region"].str.contains(region, case=False, na=False))
    ]
    if not match.empty:
        return match.iloc[0]["daily_rate_gbp"]
    # fallback to national rate
    match = rates_df[
        rates_df["vehicle_category"].str.contains(vehicle_category, case=False, na=False)
    ]
    if not match.empty:
        return match.iloc[0]["daily_rate_gbp"]
    return None

def extract_claim_details(text: str) -> dict:
    """Use Groq LLM to extract structured data from court pack text."""
    prompt = f"""
    You are an expert motor insurance claims analyst.
    Extract the following information from this court pack document.
    Return ONLY a JSON object with these exact keys:
    - vehicle_category (e.g. Small, Medium, Large, SUV, Luxury, Van)
    - hire_duration_days (integer)
    - daily_rate_charged (float, in GBP)
    - total_claim_amount (float, in GBP)
    - region (National or London)
    - hire_company (string)
    - claimant_name (string)

    Document:
    {text}

    Return only valid JSON, nothing else.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        rate_match = re.search(r'daily.*?£(\d+\.?\d*)', text, re.IGNORECASE)
        duration_match = re.search(r'(\d+)\s*days?', text, re.IGNORECASE)
        total_match = re.search(r'total.*?£(\d+\.?\d*)', text, re.IGNORECASE)
        return {
            "vehicle_category": "Medium",
            "hire_duration_days": int(duration_match.group(1)) if duration_match else 0,
            "daily_rate_charged": float(rate_match.group(1)) if rate_match else 0,
            "total_claim_amount": float(total_match.group(1)) if total_match else 0,
            "region": "National",
            "hire_company": "Unknown",
            "claimant_name": "Unknown"
        }

def analyse_claim(text: str) -> dict:
    """Full pipeline: extract details, compare rates, generate risk score."""
    rates_df = load_hire_rates()
    details = extract_claim_details(text)

    vehicle = details.get("vehicle_category", "Medium")
    region = details.get("region", "National")
    daily_rate = details.get("daily_rate_charged", 0)
    duration = details.get("hire_duration_days", 0)

    benchmark = get_benchmark_rate(vehicle, region, rates_df)

    if benchmark:
        deviation = ((daily_rate - benchmark) / benchmark) * 100
    else:
        deviation = 0

    # Risk scoring
    if deviation > 50:
        risk = "HIGH"
        recommendation = "Strongly recommend challenging this claim."
    elif deviation > 20:
        risk = "MEDIUM"
        recommendation = "Rate appears inflated. Consider negotiation."
    else:
        risk = "LOW"
        recommendation = "Rate appears reasonable. Low priority."

    return {
        "claimant": details.get("claimant_name", "Unknown"),
        "hire_company": details.get("hire_company", "Unknown"),
        "vehicle_category": vehicle,
        "region": region,
        "hire_duration_days": duration,
        "daily_rate_charged": daily_rate,
        "benchmark_daily_rate": benchmark,
        "rate_deviation_pct": round(deviation, 2),
        "risk_level": risk,
        "recommendation": recommendation,
        "total_claim": details.get("total_claim_amount", 0)
    }