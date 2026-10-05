import json
from pathlib import Path
from src.validation.claim_schema import ClaimRecord

REQUIRED_FIELDS = [
    "claim_id",
    "policy_number",
    "patient_name",
    "hospital_name",
    "diagnosis",
    "admission_date",
    "total_claim_amount",
    "claim_status"
]

# Validate one extracted claim record for basic ML readiness
def validate_claim_record(record: ClaimRecord) -> list[str]:
    issues = []
    for field_name in REQUIRED_FIELDS:
        value = getattr(record, field_name)

        if value is None or value == "":
            issues.append(f"Missing required field: {field_name}")

    if record.total_claim_amount is not None and record.total_claim_amount <=0:
        issues.append("total_claim_amount must be greater than 0")

    if record.approved_amount is not None and record.approved_amount <0:
        issues.append("approved amount cannot be negative")

    if(
        record.total_claim_amount is not None
        and record.approved_amount is not None
        and record.approved_amount > record.total_claim_amount
    ):
        issues.append("approved_amount cannot be greater than total_claim_amount")

    return issues

# Validate all extracted claim records and build a summary report
def validate_claim_records(claim_records: list[ClaimRecord]) -> dict:
    seen_claim_ids = set()
    record_issues = []

    for index, record in enumerate(claim_records, start = 1):
        issues = validate_claim_record(record)

        if record.claim_id:
            if record.claim_id in seen_claim_ids:
                issues.append("Duplicate claim_id found: {record.claim_id}")
            seen_claim_ids.add(record.claim_id)

        if issues:
            record_issues.append(
                {
                    "row_number": index,
                    "claim_id": record.claim_id,
                    "issues": issues
                }
            )
    invalid_records = len(record_issues)
    total_records = len(claim_records)

    return {
        "total_records": total_records,
        "valid_records": total_records - invalid_records,
        "invalid_records": invalid_records,
        "issues": record_issues,
        "is_ml_ready": invalid_records == 0 and total_records > 0
    }

# Save date quality report
def save_data_quality_report(output_dir:Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath_path = output_dir / "data_quality_report.json"
    outpath_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Data quality report saved to: {outpath_path}")
    return outpath_path






