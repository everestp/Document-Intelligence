from pathlib import Path
from typing import Any
import pandas as pd

from src.validation.claim_schema import ClaimRecord
from src.validation.data_quality import REQUIRED_FIELDS

# Get field Type
def get_field_type(field_info: Any) -> str:
    return str(field_info.annotation).replace(" | None", "")

# Get the example value
def get_example_value(field_name: str, claim_records: list[ClaimRecord]) -> Any:
    for record in claim_records:
        value = getattr(record, field_name)

        if value is not None and value != "":
            return value

    return None

# Build Data dictionary
def build_data_dictionary(claim_records: list[ClaimRecord]) -> list[dict]:
    dictionary_rows = []
    for field_name, field_info in ClaimRecord.model_fields.items():
        dictionary_rows.append(
            {
                "column_name": field_name,
                "data_type": get_field_type(field_info),
                "required": field_name in REQUIRED_FIELDS,
                "description": field_info.description or "",
                "example": get_example_value(
                    field_name=field_name,
                    claim_records=claim_records
                )
            }
        )
    return dictionary_rows

# Save Data dictionary
def save_data_dictionary(
        output_dir: Path,
        claim_records: list[ClaimRecord]
) -> Path | None:
    if not claim_records:
        print("No claim records available for data dictionary creation.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_dictionary.csv"
    dictionary_rows = build_data_dictionary(claim_records=claim_records)
    pd.DataFrame(dictionary_rows).to_csv(output_path, index=False)

    print(f"Data dictionary saved to: {output_path}")
    return output_path
    