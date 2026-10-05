import json
from pathlib import Path
import pandas as pd

from src.config.settings import AppSettings, validate_openai_settings
from src.validation.claim_schema import ClaimRecord

# Return the JSON path used to store extracted claim records.
def get_claim_records_path(output_dir: Path) -> Path:
    return output_dir / "claim_records.json"

# Load previosly extracted claim records from cache
def load_existing_claim_records(output_dir: Path) -> dict[str, ClaimRecord]:
    records_path = get_claim_records_path(output_dir)

    if not records_path.exists():
        return {}

    saved_records = json.loads(records_path.read_text(encoding="utf-8"))
    cache = {}

    for saved_record in saved_records:
        claim_record = ClaimRecord.model_validate(saved_record)

        if claim_record.claim_id:
            cache[claim_record.claim_id] = claim_record

    return cache

# Prompt Enginnering. Build prompt message for structured claim extraction.
def build_extraction_prompt(claim_text: str) -> list[tuple[str, str]]:
    system_message = (
        "You extract structured claim data from document text. Return only a "
        "valid JSON object. Do not include markdown, explanations, or extra "
        "text. If a field is missing, use null."
    )
    user_message = f"""
Extract the following fields from the claim document text:
- claim_id
- policy_number
- patient_name
- hospital_name
- diagnosis
- admission_date
- discharge_date
- claim_type
- total_claim_amount
- approved_amount
- claim_status

For amount fields, return numbers only. For example, Rs. 187,500/- should be
187500.

Claim document text:
{claim_text}
""".strip()

    return [
        ("system", system_message),
        ("user", user_message),
    ]

# Parse the model response as JSON
def parse_json_response(response_text: str) -> dict:
    return json.loads(response_text)

# Extract one validated ClaimRecord from cleaned claim text

def extract_claim_record_from_text(
        settings: AppSettings,
        claim_text: str
) -> ClaimRecord | None:
    if not validate_openai_settings(settings):
        print("OPENAI_API_KEY is not set. Skipping structured extraction." )
        return None

    from langchain_openai import ChatOpenAI

    chat_model = ChatOpenAI(
        model = settings.chat_model,
        api_key= settings.openai_api_key,
        temperature=0
    )
    response = chat_model.invoke(build_extraction_prompt(claim_text))
    extracted_data = parse_json_response(response.content)

    return ClaimRecord.model_validate(extracted_data)

# Extract structured records from all cleaned text files
def extract_claim_records(settings: AppSettings) -> list[ClaimRecord]:
    cleaned_claims_dir = settings.processed_data_dir / "cleaned_claims"
    existing_records = load_existing_claim_records(settings.output_data_dir)
    claim_records = []

    if not cleaned_claims_dir.exists():
        print(f"No Cleaned Claims folder found in {cleaned_claims_dir}")
        return claim_records

    cleaned_files = sorted(cleaned_claims_dir.glob("*.txt"))

    if not cleaned_files:
        print(f"No Cleaned claim files foind in {cleaned_claims_dir}")
        return claim_records

    for cleaned_file in cleaned_files:
        claim_id = cleaned_file.stem

        if claim_id in existing_records:
            print(f"Using cached structured record for: {claim_id}")
            claim_records.append(existing_records[claim_id])
            continue

        claim_text = cleaned_file.read_text(encoding="utf-8")
        claim_record = extract_claim_record_from_text(
            settings=settings,
            claim_text=claim_text)

        if claim_record:
            claim_records.append(claim_record)
            print(f"Extracted structured record for: {cleaned_file.stem}")

    return claim_records

# Save extracted claim records into JSON
def save_claim_records(output_dir: Path, claim_records: list[ClaimRecord]) -> Path | None:
    if not claim_records:
        print("No claim records available to save.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = get_claim_records_path(output_dir)
    records = [record.model_dump() for record in claim_records]
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Claim records saved to: {output_path}")
    return output_path

# Save claim records as CSV
def save_claim_records_csv(
        output_dir: Path,
        claim_records: list[ClaimRecord]
) -> Path | None:
    if not claim_records:
        print("No claim records available to save for CSV export.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "claim_dataset.csv"
    records = [record.model_dump() for record in claim_records]
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_path, index=False)

    print(f"Claim dataset CSV saved to: {output_path}")
    return output_path
