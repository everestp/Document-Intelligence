import json
from pathlib import Path

# Build a final summary of the document intelligence pipeline run
def build_processing_summary(
        claim_count: int,
        data_quality_report: dict,
        output_files: dict[str, Path | None]
) -> dict:
    return {
        "total_claims_processed": claim_count,
        "data_quality_passed": data_quality_report.get("is_ml_ready", False),
        "ml_ready": data_quality_report.get("is_ml_ready", False),
        "valid_records": data_quality_report.get("valid_records", 0),
        "invalid_records":  data_quality_report.get("invalid_records", 0),
        "output_files": {
            name: str(path) if path else None for name, path in output_files.items()
        }
    }

# Save processing summary
def save_processing_summary(
        output_dir: Path,
        summary: dict
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processing_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Processing summary saved to: {output_path}")
    return output_path
