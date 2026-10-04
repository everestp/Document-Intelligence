from pathlib import Path

from src.config.settings import load_settings
from src.ingestion.pdf_loader import combine_claim_texts, process_pdfs
from src.preprocessing.text_chunker import chunk_cleaned_claims, validate_chunk_files
from src.preprocessing.text_cleaner import clean_combined_claims
from src.rag.chunk_loader import load_all_chunks, save_chunk_manifest
from src.rag.vector_store import build_vector_store, print_search_results, search_vector_store

def main():
    """Run the document pipeline from raw PDFs to cleaned claim text"""
    settings = load_settings()
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

    # Step 4: Split Cleaned claim files into smaller chunks for RAG retrieval
    print("Creating RAG-ready text chunks...")
    chunk_files = chunk_cleaned_claims(processed_dir=processed_dir)
    print(f"Text chunking completed. Files created: {len(chunk_files)}")

    # Step 5: Validate chunk quality before embeddings..
    print("Validating text chunks...")
    report_files = validate_chunk_files(processed_dir=processed_dir)
    print(f"Chunk validation completed. Reports created: {len(report_files)}")

    # Step 6: Load chunk JSON files as the input for embeddings and vector store
    print("Loading chunks for RAG...")
    chunks = load_all_chunks(processed_dir=processed_dir)
    manifest_path = save_chunk_manifest(processed_dir=processed_dir, chunks=chunks)
    print(f"Rag chunk loading completed. Manifest created: {manifest_path is not None}")

    # Step 7: Converting chunks to embeddings and storing in chroma db
    print("Building Vector Store")
    vector_store_created = build_vector_store(settings=settings, chunks=chunks)
    print(f"Vector store created. Created or Updated : {vector_store_created}")

    # Step 8: Run a sample semantic search against the vector store
    sample_query = "What is the total claim amount?"
    search_results = search_vector_store(settings=settings, query=sample_query)
    print_search_results(query=sample_query, results=search_results)

if __name__ == "__main__":
    main()
