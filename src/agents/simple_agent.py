import json
from pathlib import Path
import re

from src.agents.document_tools import (
    ask_claim_documents,
    filter_claim_by_status,
    get_data_quality_status,
    summarize_pipeline_outputs,
    get_claim_record_by_id
)

from src.config.settings import AppSettings

# Extract claim id
def extract_claim_id(user_request: str) -> str | None:
    match = re.search(r"CLM\d+", user_request, re.IGNORECASE)
    return match.group(0).upper() if match else None

# Detect agent intent. We are detecting which tool should handle which request
def detect_agent_intent(user_request: str) -> str:
    normalized_request = user_request.lower()

    if "quality" in normalized_request or "ml-ready" in normalized_request:
        return "data_quality"

    if "summary" in normalized_request or "processed" in normalized_request:
        return "pipeline_summary"

    if "rejected" in normalized_request:
        return "filter_rejected"

    if "pending" in normalized_request:
        return "filter_pending"

    if "settled" in normalized_request:
        return "filter_settled"

    if "claim id" in normalized_request or "claim_id" in normalized_request:
        return "claim_record_by_id"

    return "document_question"

# Run document agent
def run_document_agent(settings: AppSettings, user_request: str) -> dict:
    intent = detect_agent_intent(user_request)

    if intent == "data_quality":
        result = get_data_quality_status(settings=settings)
        tool_name = "get_data_quality_status"

    elif intent == "pipeline_summary":
        result = summarize_pipeline_outputs(settings=settings)
        tool_name = "summarize_pipeline_outputs"

    elif intent == "filter_rejected":
        result = filter_claim_by_status(settings=settings, status="Rejected")
        tool_name = "filter_claims_by_status"

    elif intent == "filter_pending":
        result = filter_claim_by_status(settings=settings, status="Pending")
        tool_name = "filter_claims_by_status"

    elif intent == "filter_settled":
        result = filter_claim_by_status(settings=settings, status="Settled")
        tool_name = "filter_claims_by_status"

    elif intent == "claim_record_by_id":
        claim_id = extract_claim_id(user_request)

        if claim_id:
            result = get_claim_record_by_id(
                settings=settings,
                claim_id=claim_id
            )
            tool_name = "get_claim_record_by_id"
        else:
            result = ask_claim_documents(
                settings=settings,
                question=user_request
            )
            tool_name = "ask_claim_documents"

    else:
        result = ask_claim_documents(
            settings=settings,
            question=user_request
        )
        tool_name = "ask_claim_documents"

    return {
        "user_request": user_request,
        "intent": intent,
        "tool_used": tool_name,
        "result": result,
    }

# Print agent response
def print_agent_response(agent_response: dict) -> None:
    print(f"\nAgent Request: {agent_response['user_request']}")
    print(f"Detected intent: {agent_response['intent']}")
    print(f"Tool used: {agent_response['tool_used']}")
    print(f"Result: {agent_response['result']}")

# Save agent responses
def save_agent_responses(
        output_dir: Path,
        agent_responses: list[dict]
) -> Path | None:
    if not agent_responses:
        print("No agent responses available to save")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agent_responses.json"
    output_path.write_text(json.dumps(agent_responses, indent=2), encoding="utf-8")

    print(f"Agent responses saved to: {output_path}")
    return output_path
