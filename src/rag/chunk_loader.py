import json
from pathlib import Path

REQUIRED_RAG_FIELDS = {
    "claim_id",
    "chunk_id",
    "chunk_index",
    "source_file",
    "text",
}

# Loading the chunk file
def load_chunk_file(chunk_file: Path) -> list[dict]:
    return json.loads(chunk_file.read_text(encoding="utf-8"))

# Validating whether chunk is fine or not
def validate_rag_chunk(chunk_record: dict) -> bool:
    has_required_fields = REQUIRED_RAG_FIELDS.issubset(chunk_record)
    has_text = bool(chunk_record.get("text", "").strip())
    return has_required_fields and has_text

# Load all chunks
def load_all_chunks(processed_dir: Path) -> list[dict]:
    chunks_dir = processed_dir / "chunks"
    all_chunks = []

    if not chunks_dir.exists():
        print(f"No chunks folder found in {chunks_dir}")
        return all_chunks

    chunk_files = sorted(chunks_dir.glob("*_chunks.json"))

    if not chunk_files:
        print(f"No chunk JSON files found in {chunks_dir}")
        return all_chunks

    for chunk_file in chunk_files:
        chunk_records = load_chunk_file(chunk_file)

        for chunk_record in chunk_records:
            if validate_rag_chunk(chunk_record):
                all_chunks.append(chunk_record)

    print(f"Loaded {len(all_chunks)} valid chunks from {len(chunk_files)} file(s)")
    return all_chunks

# Build Chunk Manifest
def build_chunk_manifest(chunks: list[dict]) -> dict:
    claim_ids = sorted({chunk["claim_id"] for chunk in chunks})
    chunk_counts_by_claim = {}
    for chunk in chunks:
        claim_id = chunk["claim_id"]
        chunk_counts_by_claim[claim_id] = chunk_counts_by_claim.get(claim_id, 0) + 1

    return {
        "total_chunks": len(chunks),
        "total_claims": len(claim_ids),
        "claim_ids": claim_ids,
        "chunk_counts_by_claim": chunk_counts_by_claim
    }

# Save the Chunk Manifest
def save_chunk_manifest(processed_dir: Path, chunks: list[dict]) -> Path | None:
    if not chunks:
        print("No chunks available for manifest creation")
        return None

    rag_inputs_dir = processed_dir / "rag_inputs"
    rag_inputs_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_chunk_manifest(chunks)
    output_path = rag_inputs_dir / "chunk_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Chunk manifest created: {output_path}")
    return output_path
