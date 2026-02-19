from extractor import load_sample_text, extract_text_from_pdf
from analyser import extract_claim_details

# Ground truth — what the correct answers should be
ground_truth = [
    {
        "file": "Sample Court Pack (Mock)",
        "text": load_sample_text(),
        "expected": {
            "claimant_name": "Mr. John Smith",
            "hire_company": "FastHire Ltd",
            "vehicle_category": "Medium",
            "hire_duration_days": 21,
            "daily_rate_charged": 95.0,
            "region": "National"
        }
    },
    {
        "file": "Sample PDF (Generated)",
        "text": extract_text_from_pdf("sample_docs/sample_court_pack.pdf"),
        "expected": {
            "claimant_name": "Mrs. Sarah Johnson",
            "hire_company": "QuickHire Solutions Ltd",
            "vehicle_category": "Small",
            "hire_duration_days": 20,
            "daily_rate_charged": 110.0,
            "region": "National"
        }
    }
]

total_fields = 0
correct_fields = 0

for test in ground_truth:
    print(f"\nTesting: {test['file']}")
    extracted = extract_claim_details(test["text"])
    expected = test["expected"]

    for field, expected_value in expected.items():
        total_fields += 1
        extracted_value = extracted.get(field)
        
        # Normalise for comparison
        if isinstance(expected_value, str):
            match = str(extracted_value).strip().lower() == str(expected_value).strip().lower()
        else:
            match = extracted_value == expected_value

        status = "✅" if match else "❌"
        print(f"  {status} {field}: expected={expected_value}, got={extracted_value}")
        if match:
            correct_fields += 1

accuracy = (correct_fields / total_fields) * 100
print(f"\n{'='*50}")
print(f"Total fields tested: {total_fields}")
print(f"Correct: {correct_fields}")
print(f"Extraction Accuracy: {accuracy:.1f}%")