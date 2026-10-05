from pathlib import Path
from pydantic import BaseModel, Field
import json

# Define the structured claim fields which we need to extract for ML-Ready data
class ClaimRecord(BaseModel):
    claim_id: str | None = Field(default=None, description="Unique claim number")
    policy_number: str | None = Field(default=None, description="Insurance policy ID")
    patient_name: str | None = Field(default=None, description="Name of the patient")
    hospital_name: str | None = Field(default=None, description="Hospital name")
    diagnosis: str | None = Field(default=None, description="Primary diagnosis")
    admission_date: str | None = Field(default=None, description="Date of admission")
    discharge_date: str | None = Field(default=None, description="Date of discharge")
    claim_type: str | None = Field(default=None, description="Cashless or reimbursement")
    total_claim_amount: float | None = Field(
        default=None,
        description="Total amount claimed or billed",
    )
    approved_amount: float | None = Field(
        default=None,
        description="Approved or payable claim amount when available",
    )
    claim_status: str | None = Field(
        default=None,
        description="Claim status such as approved, rejected, or pending",
    )

# Get Claim Schema. It will return JSON schema for the claim record model
def get_claim_schema() -> dict:
    return ClaimRecord.model_json_schema()

# Save the claim extraction as JSON file
def save_claim_schema(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "claim_schema.json"
    output_path.write_text(json.dumps(get_claim_schema(), indent=2), encoding="utf-8")

    print(f"Claim schema saved to: {output_path}")
    return output_path
