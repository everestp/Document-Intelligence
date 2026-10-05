from pathlib import Path
from src.config.settings import AppSettings, validate_openai_settings
import chromadb

COLLECTION_NAME = "claim_document_chunks"

# Create embedding client
def create_embedding_client(settings: AppSettings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key
    )

# Create or load a persistent chroma db collection
def create_chroma_collection(processed_dir: Path):
    vector_store_dir = processed_dir / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(vector_store_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

# Build chunk metadata
def build_chunk_metadata(chunk: dict) -> dict:
    return {
        "claim_id": chunk["claim_id"],
        "chunk_index": chunk["chunk_index"],
        "source_file": chunk["source_file"],
        "text_length": chunk.get("text_length", len(chunk["text"]))
    }

# Getting Existing Vector Ids
def get_existing_vector_ids(collection, chunk_ids: list[str]) -> list[str]:
    if not chunk_ids:
        return set()
    existing_items = collection.get(ids=chunk_ids, include=[])
    return set(existing_items.get("ids", []))

# Check whether every expected chunk id already exists in the vector store
def vector_store_has_all_chunks(collection, chunk_ids: list[str]) -> bool:
    existing_ids = get_existing_vector_ids(collection=collection, chunk_ids=chunk_ids)
    return set(chunk_ids).issubset(existing_ids)

# Create embeddings for loaded chunks and save them into chroma db
def build_vector_store(settings: AppSettings, chunks:list[str]) -> bool:
    if not chunks:
        print("No chunks available for vector store creation")
        return False

    if not validate_openai_settings(settings):
        print("OPENAI_API_KEY is not set. Skipping Vector Store creation.")
        return False

    collection = create_chroma_collection(settings.processed_data_dir)

    # Extract Ids
    ids = [chunk["chunk_id"] for chunk in chunks]

    if vector_store_has_all_chunks(collection=collection, chunk_ids=ids):
        print(f"Skipping vector store rebuild. {len(ids)} chunks already indexed.")
        return False

    embedding_client = create_embedding_client(settings)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [build_chunk_metadata(chunk) for chunk in chunks]
    embeddings = embedding_client.embed_documents(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
        )

    print(f"Vector store updated with {len(ids)} chunks.")
    return True

# Searching Vector Store
def search_vector_store(
        settings: AppSettings,
        query: str,
        top_k: int = 3,
        claim_id: str | None = None
    ) -> list[dict]:

    if not validate_openai_settings(settings):
        print("OPENAI_API_KEY is not set. Skipping Vector search.")
        return []

    embedding_client = create_embedding_client(settings)
    collection = create_chroma_collection(settings.processed_data_dir)
    query_embedding = embedding_client.embed_query(query)

    where_filter = {"claim_id": claim_id} if claim_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):
        retrieved_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return retrieved_chunks
# Print search results
def print_search_results(query: str, results: list[dict]) -> None:
    if not results:
        print("No search results found")
        return

    print(f"Search query: {query}")

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        preview = result["text"][:300].replace("\n", " ")

        print(f"\nResult {index}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Claim ID: {metadata.get('claim_id')}")
        print(f"Source file: {metadata.get('source_file')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print(f"Cosine distance: {result['distance']}")
        print(f"Preview: {preview}...")
