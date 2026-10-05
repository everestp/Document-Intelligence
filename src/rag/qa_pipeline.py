import json
from pathlib import Path
from  src.config.settings import AppSettings, validate_openai_settings
from src.rag.vector_store import search_vector_store

# Normalize a question before using it as cache key
def normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())

# Return the JSON file path used for Cached RAG Answers
def get_rag_cache_path(output_dir: Path) -> Path:
    return output_dir / "rag_answers.json"

# Load previosly saved RAG answers as qa cache
def load_rag_cache(output_dir: Path) -> dict:
    cache_path = get_rag_cache_path(output_dir)

    if not cache_path.exists():
        return {}

    saved_responses = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        normalize_question(response["question"]): response
        for response in saved_responses
    }

# Get cached RAG Response
def get_cached_rag_response(output_dir: Path, question: str) -> dict | None:
    cache = load_rag_cache(output_dir)
    cached_response = cache.get(normalize_question(question))

    if cached_response:
        print(f"Using cached RAG answer for question: {question}")
    return cached_response

# Build Source References
def build_source_references(search_results: list[dict]) -> list[dict]:
    sources = []

    for index, result in enumerate(search_results, start=1):
        metadata = result["metadata"]
        sources.append(
            {
                "source_number": index,
                "chunk_id": result["chunk_id"],
                "claim_id": metadata.get("claim_id"),
                "source_file": metadata.get("source_file"),
                "chunk_index": metadata.get("chunk_index"),
                "cosine_distance": result["distance"]
            }
        )
    return sources

# Build the context from results
def build_context_from_results(search_results: list[dict]) -> str:
    context_blocks = []

    for index, result in enumerate(search_results, start=1):
        metadata = result["metadata"]
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Claim ID: {metadata.get('claim_id')}",
                    f"Source file: {metadata.get('source_file')}",
                    f"Chunk index: {metadata.get('chunk_index')}",
                    "Text:",
                    result["text"],
                ]
            )
        )

    return "\n\n".join(context_blocks)

# Check whether context is relevant
def has_relevant_context(search_results: list[dict], max_distance: float) -> bool:
    if not search_results:
        return False

    best_distance = search_results[0]["distance"]
    return best_distance <= max_distance

# Building no context response
def build_no_context_response(question: str, search_results: list[dict]) -> dict:
    return {
        "question": question,
        "answer": "No relevant document context was found for this question",
        "sources": build_source_references(search_results)
    }

# Build RAG prompt for grounded Q&A
def build_rag_prompt(question: str, context: str) -> list[tuple[str, str]]:
    system_message = (
        "You are a document intelligence assistant. Answer the user's question "
        "using only the provided document context. If the answer is not present "
        "in the context, say that the document context does not contain enough "
        "information. Keep the answer concise and mention source numbers used."
    )
    user_message = f"Question: {question}\n\nDocument context:\n{context}"

    return [
        ("system", system_message),
        ("user", user_message)
    ]

# Answer question with context
def answer_question_with_context(
        settings: AppSettings,
        question: str,
        search_results: list[dict]) -> dict| None:

    if not search_results:
        print("No retrieved chunks available for RAG answer generation.")
        return None

    if not validate_openai_settings(settings):
        print("OPENAI_API_KEY is not set. Skipping RAG answer generation.")
        return None

    from langchain_openai import ChatOpenAI
    context = build_context_from_results(search_results)
    messages = build_rag_prompt(question=question, context=context)
    chat_model = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0
    )
    response = chat_model.invoke(messages)

    return {
        "question": question,
        "answer": response.content,
        "sources": build_source_references(search_results)
    }

# Run the full flow for one RAG question
def run_rag_question(
        settings: AppSettings,
        question: str,
        top_k: int = 3,
        use_cache: bool = True,
        claim_id: str | None = None
) -> dict | None:
    if use_cache:
        cached_response = get_cached_rag_response(
            output_dir=settings.output_data_dir,
            question=question
        )

        if cached_response:
            return cached_response

    search_results = search_vector_store(
        settings=settings,
        query=question,
        top_k=top_k,
        claim_id=claim_id
    )

    if not has_relevant_context(
        search_results=search_results,
        max_distance=settings.retrieval_distance_threshold
    ):
        print("Skipping LLM call because retrieved context is above the "
              f"distance threshold: {settings.retrieval_distance_threshold}"
              )
        return build_no_context_response(
            question=question,
            search_results=search_results
        )

    return answer_question_with_context(
        settings=settings,
        question=question,
        search_results=search_results
    )

# Print RAG answers
def print_rag_answer(rag_response: dict | None) -> None:
    if not rag_response:
        print("RAG answer was not generated.")
        return

    print(f"\nRAG question: {rag_response['question']}")
    print("RAG answer:")
    print(rag_response["answer"])
    print("\nSources:")
    for source in rag_response["sources"]:
        print(
            " - "
            f"Source {source['source_number']}: "
            f"{source['chunk_id']} "
            f"(claim={source['claim_id']}), "
            f"(chunk={source['chunk_index']}), "
            f"(distance={source['cosine_distance']})"
        )

# Save RAG Response
def save_rag_responses(output_dir: Path, rag_responses: list[dict]) -> Path | None:
    if not rag_responses:
        print("No RAG responses available to save.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = get_rag_cache_path(output_dir)
    response_cache = load_rag_cache(output_dir)

    for response in rag_responses:
        response_cache[normalize_question(response['question'])] = response

    merged_responses = list(response_cache.values())
    output_path.write_text(json.dumps(merged_responses, indent=2), encoding="utf-8")

    print(f"RAG answers saved to: {output_path}")
    return output_path
