from pathlib import Path
from src.ingestion.pdf_loader import combine_claim_texts, process_pdfs
from src.preprocessing.text_cleaner import clean_combined_claims

def main():
    """Run the document pipeline from raw PDFs to cleaned claim text"""
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    # Step 1: Extract tect from each pdf and save one text file per document:
    print("Starting PDF ingestion...")
    output_files = process_pdfs(raw_dir=raw_dir, processed_dir=processed_dir)
    print(f"PDF ingestion completed. Files created: {len(output_files)}")

    # Step 2: Combine all document text files that belong to the same claim
    print("Combining claim documents...")
    combined_files = combine_claim_texts(processed_dir=processed_dir)
    print(f"Claim document combination completed. Files created: {len(combined_files)}")

    # Step 3: Clean the combined claim text so it is easier to chunk and query
    print("Cleaning combined claim text")
    cleaned_files = clean_combined_claims(processed_dir=processed_dir)
    print(f"Text cleaning completed. Files created: {len(cleaned_files)}")

if __name__ == "__main__":
    main()