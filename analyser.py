"""
Court Pack Analyser — RAG-powered extraction pipeline
------------------------------------------------------
Flow:
  1. extractor.py  → raw text from PDF
  2. validate_document() → confirm it is a court pack before proceeding
  3. rag_pipeline  → chunk → embed (all-MiniLM-L6-v2) → FAISS index
  4. retrieve()    → targeted context per extraction field
  5. Groq LLaMA    → structured JSON from retrieved context only
  6. benchmarking  → rate deviation + risk scoring
"""

import os
import json
import re
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from rag_pipeline import CourtPackRAG

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Document validation
# ---------------------------------------------------------------------------

COURT_PACK_KEYWORDS = [
    "hire", "claim", "vehicle", "daily rate", "claimant",
    "credit hire", "insurer", "accident", "solicitor", "invoice",
    "replacement", "hire period", "hire company", "court"
]

def validate_document(text: str) -> tuple[bool, str]:
    """
    Check whether the uploaded document is a court pack.
    Returns (is_valid: bool, reason: str).
    """
    text_lower = text.lower()
    matched = [kw for kw in COURT_PACK_KEYWORDS if kw in text_lower]

    if len(matched) >= 3:
        return True, f"Document recognised as a court pack ({', '.join(matched[:5])} detected)."
    else:
        return False, (
            "This does not appear to be a court pack document. "
            "Expected keywords such as 'hire', 'claim', 'vehicle', 'daily rate' were not found. "
            "Please upload a valid motor insurance court pack PDF."
        )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def load_hire_rates(csv_path: str = "data/hire_rates.csv") -> pd.DataFrame:
    """Load benchmark hire rates from CSV."""
    return pd.read_csv(csv_path)


def get_benchmark_rate(
    vehicle_category: str, region: str, rates_df: pd.DataFrame
) -> float | None:
    """Find the benchmark daily rate for a vehicle/region combination."""
    match = rates_df[
        rates_df["vehicle_category"].str.contains(vehicle_category, case=False, na=False)
        & rates_df["region"].str.contains(region, case=False, na=False)
    ]
    if not match.empty:
        return match.iloc[0]["daily_rate_gbp"]

    # Fallback: vehicle match only
    match = rates_df[
        rates_df["vehicle_category"].str.contains(vehicle_category, case=False, na=False)
    ]
    return match.iloc[0]["daily_rate_gbp"] if not match.empty else None


# ---------------------------------------------------------------------------
# RAG-powered extraction
# ---------------------------------------------------------------------------

def build_extraction_prompt(context: dict) -> str:
    """Build a structured LLM prompt from RAG-retrieved field contexts."""
    return f"""You are an expert motor insurance claims analyst.

Using ONLY the retrieved document sections below, extract the claim information.
Return ONLY a valid JSON object with these exact keys:
  - vehicle_category  (one of: Small, Medium, Large, SUV, Luxury, Van)
  - hire_duration_days  (integer)
  - daily_rate_charged  (float, GBP)
  - total_claim_amount  (float, GBP)
  - region  (National or London)
  - hire_company  (string)
  - claimant_name  (string)

--- HIRE RATE ---
{context['hire_rate']}

--- VEHICLE ---
{context['vehicle']}

--- DURATION ---
{context['duration']}

--- CLAIMANT ---
{context['claimant']}

--- REGION ---
{context['region']}

--- HIRE COMPANY ---
{context['hire_company']}

--- TOTAL AMOUNT ---
{context['total_amount']}

Return only valid JSON, nothing else.
"""


def _regex_fallback(full_text: str) -> dict:
    """Last-resort regex extraction when LLM JSON parsing fails."""
    rate_match     = re.search(r'daily.*?£(\d+\.?\d*)', full_text, re.IGNORECASE)
    duration_match = re.search(r'(\d+)\s*days?', full_text, re.IGNORECASE)
    total_match    = re.search(r'total.*?£(\d+\.?\d*)', full_text, re.IGNORECASE)
    return {
        "vehicle_category":   "Medium",
        "hire_duration_days": int(duration_match.group(1)) if duration_match else 0,
        "daily_rate_charged": float(rate_match.group(1))   if rate_match     else 0.0,
        "total_claim_amount": float(total_match.group(1))  if total_match    else 0.0,
        "region":             "National",
        "hire_company":       "Unknown",
        "claimant_name":      "Unknown",
    }


def extract_claim_details(text: str) -> tuple[dict, int]:
    """
    RAG-powered extraction pipeline.

    Steps:
      1. Index the document in FAISS (chunk + embed)
      2. Retrieve targeted context for each claim field
      3. Send structured prompt to Groq LLaMA
      4. Parse JSON response; fall back to regex on failure

    Returns:
      (extracted_fields dict, number_of_chunks indexed)
    """
    rag = CourtPackRAG()
    n_chunks = rag.index_document(text)
    context  = rag.retrieve_claim_context(k=2)
    prompt   = build_extraction_prompt(context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw), n_chunks
    except json.JSONDecodeError:
        return _regex_fallback(text), n_chunks


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyse_claim(text: str) -> dict:
    """
    End-to-end pipeline:
      validate → RAG extraction → benchmark lookup → risk scoring → recommendation.

    Raises ValueError if document is not a valid court pack.
    """
    # Step 1: Validate document
    is_valid, reason = validate_document(text)
    if not is_valid:
        raise ValueError(reason)

    # Step 2: Extract + benchmark + score
    rates_df          = load_hire_rates()
    details, n_chunks = extract_claim_details(text)

    vehicle    = details.get("vehicle_category", "Medium")
    region     = details.get("region", "National")
    daily_rate = details.get("daily_rate_charged", 0.0)
    duration   = details.get("hire_duration_days", 0)
    benchmark  = get_benchmark_rate(vehicle, region, rates_df)

    deviation = ((daily_rate - benchmark) / benchmark * 100) if benchmark else 0.0

    if deviation > 50:
        risk           = "HIGH"
        recommendation = "Strongly recommend challenging this claim."
    elif deviation > 20:
        risk           = "MEDIUM"
        recommendation = "Rate appears inflated. Consider negotiation."
    else:
        risk           = "LOW"
        recommendation = "Rate appears reasonable. Low priority."

    return {
        "claimant":             details.get("claimant_name", "Unknown"),
        "hire_company":         details.get("hire_company", "Unknown"),
        "vehicle_category":     vehicle,
        "region":               region,
        "hire_duration_days":   duration,
        "daily_rate_charged":   daily_rate,
        "benchmark_daily_rate": benchmark,
        "rate_deviation_pct":   round(deviation, 2),
        "risk_level":           risk,
        "recommendation":       recommendation,
        "total_claim":          details.get("total_claim_amount", 0.0),
        "rag_chunks_indexed":   n_chunks,
    }