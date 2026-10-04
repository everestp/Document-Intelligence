import re
from pathlib import Path


COMMON_TEXT_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "×": "x",
    "â€”": "-",
    "â€“": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€�": '"',
    "Ã—": "x",
    "Â": "",
}


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving document structure.

    This function fixes common PDF encoding artifacts and normalizes whitespace.
    It keeps important markers such as page headings and document boundaries so
    later steps can still understand where the text came from.
    """
    cleaned_text = text

    for old_value, new_value in COMMON_TEXT_REPLACEMENTS.items():
        cleaned_text = cleaned_text.replace(old_value, new_value)

    # Normalize tabs/spaces without removing line breaks used as structure.
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    # Reduce large blank areas to a maximum of one blank line.
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # Remove trailing whitespace from every line.
    cleaned_lines = [line.strip() for line in cleaned_text.splitlines()]

    return "\n".join(cleaned_lines).strip()


def clean_combined_claims(processed_dir: Path) -> list[Path]:
    """
    Clean every combined claim text file.

    Input files are read from data/processed/combined_claims. Cleaned files are
    saved into data/processed/cleaned_claims with the same claim ID filename.
    """
    combined_claims_dir = processed_dir / "combined_claims"
    cleaned_claims_dir = processed_dir / "cleaned_claims"
    cleaned_claims_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files = []

    if not combined_claims_dir.exists():
        print(f"No combined claims folder found in {combined_claims_dir}")
        return cleaned_files

    combined_files = sorted(combined_claims_dir.glob("*.txt"))

    if not combined_files:
        print(f"No combined claim files found in {combined_claims_dir}")
        return cleaned_files

    for combined_file in combined_files:
        output_path = cleaned_claims_dir / combined_file.name

        if output_path.exists():
            print(f"Skipping already cleaned claim text: {output_path}")
            cleaned_files.append(output_path)
            continue

        raw_text = combined_file.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)

        output_path.write_text(cleaned_text, encoding="utf-8")
        cleaned_files.append(output_path)

        print(f"Cleaned claim text created: {output_path}")

    return cleaned_files
