import re
from pathlib import Path
import fitz

# Extract the Claim Id
def extract_claim_id(pdf_path: Path) -> str:
    match = re.search(r"CLM\d+", str(pdf_path), re.IGNORECASE)

    if match:
        return match.group(0).upper()
    
    return "UNKNOWN_CLAIM"

# Extract text from PDF
def extract_text_from_pdf(pdf_path: Path) -> str:
    text_by_page = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()
            text_by_page.append(f"\n {page_number} ---\n{page_text}")

    return "\n".join(text_by_page).strip()

# Process PDFs
# data/processed/extracted_text/CLM2024001847/01_Claim_Form_CLM2024001847.txt
def process_pdfs(raw_dir: Path, processed_dir: Path) -> list[Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_output_dir = processed_dir / "extracted_text"
    pdf_files = sorted(raw_dir.rglob("*.pdf"))
    output_files = []

    if not pdf_files:
        print(f"No PDF files found in {raw_dir}") 
        return output_files
    
    for pdf_path in pdf_files:
        # Group each PDF output by the claim id found in the path
        claim_id = extract_claim_id(pdf_path)
        claim_output_dir = text_output_dir / claim_id
        claim_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = claim_output_dir / f"{pdf_path.stem}.txt"

        if output_path.exists():
            print(f"Skipping already extracted PDF: {pdf_path.name} ({claim_id})")
            output_files.append(output_path)
            continue
        
        extract_text = extract_text_from_pdf(pdf_path)
        output_path.write_text(extract_text, encoding="utf-8")
        output_files.append(output_path)

        print(f"Processed: {pdf_path.name} ({claim_id})")
        print(f"Saved text to: {output_path}")
    
    return output_files

# Combine Claim Texts
def combine_claim_texts(processed_dir: Path) -> list[Path]:
    extracted_text_dir = processed_dir / "extracted_text"
    combined_output_dir = processed_dir / "combined_claims"
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    combined_files = []

    if not extracted_text_dir.exists():
        print(f"No extracted text folder found in {extracted_text_dir}") 
        return combined_files
    
    claim_dirs = sorted(path for path in extracted_text_dir.iterdir() if path.is_dir())

    if not claim_dirs:
        print(f"No Claim folders found in {extracted_text_dir}")
        return combined_files
    
    for claim_dir in claim_dirs:
        text_files = sorted(claim_dir.glob("*.txt"))

        if not text_files:
            continue

        output_path = combined_output_dir / f"{claim_dir.name}.txt"

        if output_path.exists():
            print(f"Skipping already combined claim text: {output_path}")
            combined_files.append(output_path)
            continue

        combined_parts = [f"CLAIM ID: {claim_dir.name}"]

        for text_file in text_files:
            # Keep document boundaries
            document_text = text_file.read_text(encoding="utf-8")
            combined_parts.append(
                "\n".join(
                    [
                        "",
                        "=" * 80,
                        f"DOCUMENT: {text_file.stem}",
                        "=" * 80,
                        document_text,
                    ]
                )
            )
        output_path.write_text("\n".join(combined_parts).strip(), encoding="utf-8")    
        combined_files.append(output_path)
        print(f"Combined claim text created: {output_path}")

    return combined_files