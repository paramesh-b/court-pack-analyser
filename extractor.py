import pdfplumber
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def load_sample_text() -> str:
    """Return a mock court pack for testing without a real PDF."""
    return """
    CREDIT HIRE CLAIM - COURT PACK SUMMARY

    Claimant: Mr. John Smith
    Date of Accident: 15th January 2026
    Vehicle Damaged: Ford Focus (Medium)
    Region: National

    Credit Hire Company: FastHire Ltd
    Replacement Vehicle: Ford Mondeo (Large)
    Hire Start Date: 16th January 2026
    Hire End Date: 6th February 2026
    Total Hire Duration: 21 days

    Daily Hire Rate Charged: £95.00
    Total Hire Claim Amount: £1,995.00

    Additional Charges:
    - Collision Damage Waiver: £12.00/day
    - Delivery/Collection Fee: £75.00

    Total Invoice Amount: £2,322.00

    Solicitor Reference: JS/2026/0142
    Insurer: SafeDrive Insurance
    """