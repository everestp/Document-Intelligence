from pathlib import Path

from src.config.settings import load_settings
from src.ingestion.pdf_loader import combine_claim_texts, process_pdfs
from src.preprocessing.text_chunker import chunk_cleaned_claims, validate_chunk_files
from src.preprocessing.text_cleaner import clean_combined_claims
from src.rag.chunk_loader import load_all_chunks, save_chunk_manifest
from src.rag.vector_store import build_vector_store, print_search_results, search_vector_store
from src.rag.qa_pipeline import print_rag_answer, run_rag_question, save_rag_responses
from src.validation.claim_schema import save_claim_schema
from src.validation.data_quality import save_data_quality_report, validate_claim_records
from src.validation.data_dictionary import save_data_dictionary
from src.extraction.claim_extractor import (
    extract_claim_records,
    save_claim_records,
    save_claim_records_csv
)
from src.validation.processing_summary import (
    build_processing_summary,
    save_processing_summary
)

def main():
    """Run the document pipeline from raw PDFs to cleaned claim text"""
    settings = load_settings()
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    output_dir = settings.output_data_dir

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
    claim_id = "CLM2024001847"
    sample_query = "What is the total claim amount?"
    search_results = search_vector_store(
        settings=settings,
        query=sample_query,
        top_k=3,
        claim_id=claim_id
    )
    print_search_results(query=sample_query, results=search_results)

    # Step 9: Ask multiple grounded RAG questions using retrieval + LLM
    print("Generating RAG Answers")
    sample_questions = [
        "What is the total claim amount?",
        "What diagnosis is mentioned in the claim documents?",
        "What is the policy number?",
        "What is the patient's passport number?",
        "Who is Bill Gates?",
    ]

    rag_responses = []
    for question in sample_questions:
        rag_answer = run_rag_question(
            settings=settings,
            question=question,
            claim_id=claim_id,
            use_cache=True
        )
        print_rag_answer(rag_response=rag_answer)

        if rag_answer:
            rag_responses.append(rag_answer)

    # Step 10: Save RAG answers to cache
    print("Saving RAG answers.....")
    rag_answers_path = save_rag_responses(output_dir=output_dir, rag_responses=rag_responses)

    # Step 11: Save the structured extraction schema for ML-Ready claim records
    claim_schema_path = save_claim_schema(output_dir=output_dir)

    # Step 12: Extract Validated structured claim records from cleaned text
    print("Extracting structured claim records...")
    claim_records = extract_claim_records(settings=settings)
    claim_records_path = save_claim_records(
        output_dir=output_dir,
        claim_records=claim_records
    )

    # Step 13: Export validated claim records as in ML-Ready CSV Dataset
    print("Exporting claim records to csv...")
    claim_dataset_path = save_claim_records_csv(
        output_dir=output_dir,
        claim_records=claim_records
    )

    # Step 14: Validate extracted records before calling the CSV ML-Ready
    print("Running data quality checks...")
    data_quality_report = validate_claim_records(claim_records=claim_records)
    data_quality_report_path = save_data_quality_report(
        output_dir=output_dir,
        report=data_quality_report
    )

    # Step 15: Create a data dictionary for the ML Ready CSV Columns
    print("Creating data dictionary...")
    data_dictionary_path = save_data_dictionary(
        output_dir=output_dir,
        claim_records=claim_records
    )

    # Step 16: Save one final summary of the pipeline outputs and quality status
    print("Saving processing summary")
    processing_summary = build_processing_summary(
        claim_count=len(claim_records),
        data_quality_report=data_quality_report,
        output_files={
            "rag_answers": rag_answers_path,
            "claim_schema": claim_schema_path,
            "claim_records": claim_records_path,
            "claims_dataset": claim_dataset_path,
            "data_quality_report": data_quality_report_path,
            "data_dictionary": data_dictionary_path
        }
    )
    save_processing_summary(output_dir=output_dir, summary=processing_summary)

if __name__ == "__main__":
    main()
