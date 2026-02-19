from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

doc = SimpleDocTemplate(
    "sample_docs/sample_court_pack.pdf",
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

styles = getSampleStyleSheet()
content = []

paragraphs = [
    ("CREDIT HIRE CLAIM — COURT PACK", styles['Title']),
    ("Claim Reference: CH/2026/00891", styles['Normal']),
    ("Court: Manchester County Court", styles['Normal']),
    ("Date Prepared: 10th February 2026", styles['Normal']),
    ("CLAIMANT DETAILS", styles['Heading2']),
    ("Claimant: Mrs. Sarah Johnson", styles['Normal']),
    ("Address: 42 Birchwood Avenue, Manchester, M14 6PQ", styles['Normal']),
    ("Date of Accident: 2nd January 2026", styles['Normal']),
    ("Vehicle Damaged: Toyota Yaris (Small)", styles['Normal']),
    ("CREDIT HIRE DETAILS", styles['Heading2']),
    ("Credit Hire Company: QuickHire Solutions Ltd", styles['Normal']),
    ("Replacement Vehicle Provided: Ford Focus (Medium)", styles['Normal']),
    ("Hire Start Date: 3rd January 2026", styles['Normal']),
    ("Hire End Date: 23rd January 2026", styles['Normal']),
    ("Total Hire Duration: 20 days", styles['Normal']),
    ("Region: National", styles['Normal']),
    ("FINANCIAL SUMMARY", styles['Heading2']),
    ("Daily Hire Rate Charged: £110.00 per day", styles['Normal']),
    ("Total Hire Charge: £2,200.00", styles['Normal']),
    ("Collision Damage Waiver: £15.00 per day", styles['Normal']),
    ("Delivery and Collection Fee: £85.00", styles['Normal']),
    ("Total Invoice Amount: £2,585.00", styles['Normal']),
    ("SOLICITOR DETAILS", styles['Heading2']),
    ("Solicitor: Brown & Partners LLP", styles['Normal']),
    ("Reference: SJ/2026/0089", styles['Normal']),
    ("Defendant Insurer: SafeRoad Insurance plc", styles['Normal']),
    ("NOTES", styles['Heading2']),
    ("The claimant contends that the hire rate is the basic hire rate available in the market. The defendant disputes the rate as significantly above benchmark rates for the vehicle category and region.", styles['Normal']),
]

for text, style in paragraphs:
    content.append(Paragraph(text, style))
    content.append(Spacer(1, 0.3*cm))

doc.build(content)
print("Sample PDF created: sample_docs/sample_court_pack.pdf")