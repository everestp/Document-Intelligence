import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_openai import ChatOpenAI

from src.agents.document_tools import (
    ask_claim_documents,
    filter_claim_by_status,
    get_claim_fields,
    get_claim_record_by_id,
    get_claim_record_by_patient,
    get_data_quality_status,
    query_claim_dataset,
    summarize_pipeline_outputs,
)
from src.config.settings import AppSettings, validate_openai_settings


MAX_QUERY_LIMIT = 20
DEFAULT_QUERY_LIMIT = 10
UNSAFE_REQUEST_KEYWORDS = {
    "delete",
    "drop",
    "remove",
    "update",
    "insert",
    "overwrite",
    "truncate",
    "modify",
}

ALLOWED_TOOLS = {
    "summarize_pipeline_outputs",
    "get_data_quality_status",
    "filter_claim_by_status",
    "get_claim_fields",
    "get_claim_record_by_id",
    "get_claim_record_by_patient",
    "query_claim_dataset",
    "ask_claim_documents",
}

ALLOWED_CLAIM_STATUSES = {"Settled", "Rejected", "Pending"}
ALLOWED_CLAIM_FIELDS = {
    "claim_id",
    "policy_number",
    "patient_name",
    "hospital_name",
    "diagnosis",
    "admission_date",
    "discharge_date",
    "claim_type",
    "total_claim_amount",
    "approved_amount",
    "claim_status",
}


def build_tool_selection_prompt(user_request: str) -> str:
    """
    Build the prompt that asks the LLM to choose one safe project tool.

    The LLM only returns a JSON decision. Python code still executes the tool,
    which keeps the workflow controlled and easy to explain in the course.
    """
    return f"""
You are a document intelligence tool selector.
Choose exactly one tool for the user request.
The tools are read-only. If the request asks to delete, update, overwrite,
insert, drop, truncate, or modify records, do not choose a data tool.

Allowed tools:
1. summarize_pipeline_outputs
   Use for pipeline summaries, processed claim counts, and output artifacts.
2. get_data_quality_status
   Use for data quality, validation, ML-ready status, and required fields.
3. query_claim_dataset
   Use for structured CSV questions that need selected columns, filters,
   sorting, limits, claim IDs, patient names, statuses, diagnosis, policy
   numbers, claim amounts, or exact field lookups. Examples:
   "what is the policy number",
   "what is the policy number for claim CLM2024002193",
   "what is the diagnosis for claim CLM2024001847",
   "show rejected claims above 100000",
   "show policy number and diagnosis for settled claims",
   "top 3 claims by total amount",
   "claims where approved amount is less than total amount".
4. ask_claim_documents
   Use for natural-language questions that need evidence from claim documents.
   Do not use this for structured CSV field lookups.

Return only valid JSON in this exact shape:
{{
  "tool_name": "one_allowed_tool_name",
  "arguments": {{}},
  "reason": "short reason"
}}

For ask_claim_documents, arguments must be:
{{
  "question": "the original user question"
}}

For query_claim_dataset, arguments must be:
{{
  "query_plan": {{
    "select": ["claim_id", "policy_number"],
    "filters": [
      {{"column": "claim_status", "operator": "equals", "value": "Rejected"}}
    ],
  "sort_by": "total_claim_amount",
  "sort_order": "desc",
  "limit": 10
  }}
}}

For column-to-column comparisons, use value_column:
{{
  "column": "approved_amount",
  "operator": "less_than",
  "value_column": "total_claim_amount"
}}

Supported query operators:
equals, not_equals, contains, greater_than, less_than,
greater_than_or_equal, less_than_or_equal

Valid CSV fields:
claim_id, policy_number, patient_name, hospital_name, diagnosis,
admission_date, discharge_date, claim_type, total_claim_amount,
approved_amount, claim_status

Query safety:
- The maximum query limit is {MAX_QUERY_LIMIT}.
- If the user does not request a limit, use {DEFAULT_QUERY_LIMIT}.
- Never create a plan that modifies data.
- Amount fields in this project represent Indian Rupees.
- Do not assume USD or dollars.

User request:
{user_request}
""".strip()


def clean_json_response(response_text: str) -> str:
    """
    Remove simple Markdown code fences from an LLM JSON response.

    The prompt asks for plain JSON.
    """
    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.removeprefix("```json").removesuffix("```")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```").removesuffix("```")

    return cleaned_text.strip()


def parse_tool_decision(response_text: str) -> dict:
    """
    Parse the LLM response into a tool decision dictionary.

    If the model returns invalid JSON, this function returns a safe fallback
    decision that asks the RAG pipeline to answer the original request.
    """
    return json.loads(clean_json_response(response_text))


def validate_tool_decision(decision: dict, user_request: str) -> dict:
    """
    Validate and normalize the selected tool before execution.

    This prevents the LLM from choosing tools that are not part of the course
    project and fills required arguments when possible.
    """
    tool_name = decision.get("tool_name")
    arguments = decision.get("arguments") or {}
    reason = decision.get("reason", "Selected by LLM tool router.")

    if tool_name not in ALLOWED_TOOLS:
        return {
            "tool_name": "ask_claim_documents",
            "arguments": {"question": user_request},
            "reason": "Fallback used because the selected tool was not allowed.",
        }

    if tool_name == "filter_claim_by_status":
        status = str(arguments.get("status", "")).title()

        if status not in ALLOWED_CLAIM_STATUSES:
            return {
                "tool_name": "ask_claim_documents",
                "arguments": {"question": user_request},
                "reason": "Fallback used because the claim status was not supported.",
            }

        arguments = {"status": status}

    if tool_name == "get_claim_fields":
        fields = arguments.get("fields") or []

        if isinstance(fields, str):
            fields = [fields]

        selected_fields = [
            str(field)
            for field in fields
            if str(field) in ALLOWED_CLAIM_FIELDS
        ]

        if "claim_id" not in selected_fields:
            selected_fields.insert(0, "claim_id")

        if len(selected_fields) == 1:
            selected_fields.append("policy_number")

        arguments = {"fields": selected_fields}

    if tool_name == "get_claim_record_by_id":
        claim_id = str(arguments.get("claim_id") or "").strip()

        if not claim_id:
            return {
                "tool_name": "get_claim_fields",
                "arguments": {"fields": ["claim_id", "policy_number"]},
                "reason": "Fallback used because no claim ID was provided.",
            }

        arguments = {"claim_id": claim_id}

    if tool_name == "get_claim_record_by_patient":
        patient_name = str(arguments.get("patient_name") or "").strip()

        if not patient_name:
            return {
                "tool_name": "get_claim_fields",
                "arguments": {"fields": ["claim_id", "patient_name"]},
                "reason": "Fallback used because no patient name was provided.",
            }

        arguments = {"patient_name": patient_name}

    if tool_name == "query_claim_dataset":
        query_plan = validate_query_plan(
            query_plan=arguments.get("query_plan") or {},
            user_request=user_request,
        )
        arguments = {"query_plan": query_plan}

    if tool_name == "ask_claim_documents":
        question = str(arguments.get("question") or user_request)
        arguments = {"question": question}

    if tool_name in {"summarize_pipeline_outputs", "get_data_quality_status"}:
        arguments = {}

    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "reason": reason,
    }


def is_unsafe_request(user_request: str) -> bool:
    """
    Detect requests that try to modify or delete project data.

    The course agent is intentionally read-only, so unsafe requests are blocked
    before tool selection or query execution.
    """
    normalized_request = user_request.lower()

    return any(
        keyword in normalized_request
        for keyword in UNSAFE_REQUEST_KEYWORDS
    )


def build_blocked_agent_response(user_request: str) -> dict:
    """
    Return a safe response for unsupported write or destructive requests.

    This keeps the agent helpful while making its read-only boundary explicit.
    """
    final_answer = (
        "I can only read and analyze the claims dataset. "
        "I cannot modify, delete, overwrite, or insert records."
    )

    return {
        "user_request": user_request,
        "tool_decision": {
            "tool_name": "blocked_request",
            "arguments": {},
            "reason": "The request attempted a write or destructive action.",
        },
        "tool_used": "blocked_request",
        "final_answer": final_answer,
        "result": {"error": final_answer},
    }


def get_agent_audit_path(settings: AppSettings) -> Path:
    """
    Return the file path where LLM agent audit entries are stored.

    """
    return settings.output_data_dir / "agent_audit_trail.json"


def load_agent_audit_trail(settings: AppSettings) -> list[dict]:
    """
    Load existing agent audit entries from disk.

    If the audit file does not exist yet, an empty list is returned.
    """
    audit_path = get_agent_audit_path(settings=settings)

    if not audit_path.exists():
        return []

    return json.loads(audit_path.read_text(encoding="utf-8"))


def extract_query_plan(agent_response: dict) -> dict | None:
    """
    Extract the dynamic query plan from an agent response when available.

    Only query-based agent requests have a query plan, so other tools return
    None for this audit field.
    """
    result = agent_response.get("result")

    if isinstance(result, dict):
        return result.get("query_plan")

    return None


def extract_row_count(agent_response: dict) -> int | None:
    """
    Extract the number of returned rows from a tool result when available.

    This helps audit how much data the dynamic query returned.
    """
    result = agent_response.get("result")

    if isinstance(result, dict) and "row_count" in result:
        return result["row_count"]

    if isinstance(result, list):
        return len(result)

    return None


def extract_agent_errors(agent_response: dict) -> list[str]:
    """
    Extract error messages from an agent result for audit logging.

    The audit trail keeps errors as a list so future lectures can add multiple
    validation or execution issues without changing the shape of the log.
    """
    result = agent_response.get("result")

    if isinstance(result, dict) and "error" in result:
        return [str(result["error"])]

    if isinstance(result, list):
        return [
            str(item["error"])
            for item in result
            if isinstance(item, dict) and "error" in item
        ]

    return []


def build_agent_audit_entry(agent_response: dict) -> dict:
    """
    Build a compact audit entry from a full agent response.

    The audit entry captures observability fields without duplicating the entire
    raw result payload from the demo response file.
    """
    tool_used = agent_response.get("tool_used")
    safety_status = "blocked" if tool_used == "blocked_request" else "allowed"

    return {
        "request_id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "user_request": agent_response.get("user_request"),
        "safety_status": safety_status,
        "tool_used": tool_used,
        "query_plan": extract_query_plan(agent_response=agent_response),
        "row_count": extract_row_count(agent_response=agent_response),
        "final_answer": agent_response.get("final_answer"),
        "errors": extract_agent_errors(agent_response=agent_response),
    }


def append_agent_audit_entry(settings: AppSettings, agent_response: dict) -> Path:
    """
    Append one agent execution entry to the audit trail JSON file.

    The file is rewritten as a JSON array.
    """
    settings.output_data_dir.mkdir(parents=True, exist_ok=True)
    audit_path = get_agent_audit_path(settings=settings)
    audit_entries = load_agent_audit_trail(settings=settings)
    audit_entries.append(build_agent_audit_entry(agent_response=agent_response))
    audit_path.write_text(json.dumps(audit_entries, indent=2), encoding="utf-8")

    return audit_path


def validate_query_plan(query_plan: dict, user_request: str) -> dict:
    """
    Validate an LLM-generated query plan before dataframe execution.

    Only known CSV columns and supported operators are allowed, which keeps the
    dynamic query tool flexible without letting the model run arbitrary code.
    """
    supported_operators = {
        "equals",
        "not_equals",
        "contains",
        "greater_than",
        "less_than",
        "greater_than_or_equal",
        "less_than_or_equal",
    }
    selected_fields = query_plan.get("select") or list(ALLOWED_CLAIM_FIELDS)

    if isinstance(selected_fields, str):
        selected_fields = [selected_fields]

    selected_fields = [
        field for field in selected_fields if field in ALLOWED_CLAIM_FIELDS
    ]

    if not selected_fields:
        selected_fields = ["claim_id", "policy_number", "claim_status"]

    validated_filters = []

    for filter_rule in query_plan.get("filters", []):
        if not isinstance(filter_rule, dict):
            continue

        column = filter_rule.get("column")
        operator = filter_rule.get("operator", "equals")

        if column not in ALLOWED_CLAIM_FIELDS:
            continue

        if operator not in supported_operators:
            operator = "equals"

        validated_filters.append(
            {
                "column": column,
                "operator": operator,
                "value": filter_rule.get("value"),
                "value_column": filter_rule.get("value_column")
                if filter_rule.get("value_column") in ALLOWED_CLAIM_FIELDS
                else None,
            }
        )

    sort_by = query_plan.get("sort_by")
    sort_order = str(query_plan.get("sort_order", "asc")).lower()
    limit = query_plan.get("limit")

    if sort_by not in ALLOWED_CLAIM_FIELDS:
        sort_by = None

    if sort_order not in {"asc", "desc"}:
        sort_order = "asc"

    has_explicit_limit = any(
        keyword in user_request.lower()
        for keyword in ["top", "first", "last", "limit", "only one", "single"]
    )

    if not isinstance(limit, int) or limit < 1:
        limit = DEFAULT_QUERY_LIMIT

    limit = min(limit, MAX_QUERY_LIMIT)

    if limit == 1 and not validated_filters and not has_explicit_limit:
        limit = DEFAULT_QUERY_LIMIT

    return {
        "select": selected_fields,
        "filters": validated_filters,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "limit": limit,
    }


def select_tool_with_llm(settings: AppSettings, user_request: str) -> dict:
    """
    Ask the chat model to select the best allowed tool for the request.

    The returned decision is validated before any tool is executed.
    """
    if not validate_openai_settings(settings=settings):
        return {
            "tool_name": "ask_claim_documents",
            "arguments": {"question": user_request},
            "reason": "OPENAI_API_KEY is missing, so the default RAG tool was selected.",
        }

    chat_model = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    prompt = build_tool_selection_prompt(user_request=user_request)
    response = chat_model.invoke(prompt)

    try:
        raw_decision = parse_tool_decision(response.content)
    except json.JSONDecodeError:
        raw_decision = {
            "tool_name": "ask_claim_documents",
            "arguments": {"question": user_request},
            "reason": "Fallback used because the LLM returned invalid JSON.",
        }

    return validate_tool_decision(
        decision=raw_decision,
        user_request=user_request,
    )


def execute_tool_decision(settings: AppSettings, decision: dict) -> dict | list | None:
    """
    Execute the validated tool decision.

    Tool execution stays in normal Python code so the project remains explicit,
    debuggable.
    """
    tool_name = decision["tool_name"]
    arguments = decision.get("arguments", {})

    if tool_name == "summarize_pipeline_outputs":
        return summarize_pipeline_outputs(settings=settings)

    if tool_name == "get_data_quality_status":
        return get_data_quality_status(settings=settings)

    if tool_name == "filter_claim_by_status":
        return filter_claim_by_status(
            settings=settings,
            status=arguments["status"],
        )

    if tool_name == "get_claim_fields":
        return get_claim_fields(
            settings=settings,
            fields=arguments["fields"],
        )

    if tool_name == "get_claim_record_by_id":
        return get_claim_record_by_id(
            settings=settings,
            claim_id=arguments["claim_id"],
        )

    if tool_name == "get_claim_record_by_patient":
        return get_claim_record_by_patient(
            settings=settings,
            patient_name=arguments["patient_name"],
        )

    if tool_name == "query_claim_dataset":
        return query_claim_dataset(
            settings=settings,
            query_plan=arguments["query_plan"],
        )

    return ask_claim_documents(
        settings=settings,
        question=arguments.get("question", ""),
    )


def build_final_answer_prompt(
    user_request: str,
    tool_name: str,
    tool_result: dict | list | None,
) -> str:
    """
    Build the prompt that turns raw tool output into a final user answer.

    The answer generator should not create new facts; it can only summarize the
    records, row counts, reports, or RAG answer returned by the selected tool.
    """
    tool_result_json = json.dumps(tool_result, indent=2)

    return f"""
You are a document intelligence assistant.
Answer the user's request using only the tool result below.

Rules:
- Do not invent facts that are not present in the tool result.
- If the tool result contains records, summarize the matching records clearly.
- If the result is empty, say no matching records were found.
- Keep the answer concise and business-friendly.
- Mention key claim IDs when they are available.
- Amount fields such as total_claim_amount and approved_amount are in Indian Rupees.
- Format amount values using ₹.
- Never use $, USD, or dollars unless the tool result explicitly says USD.

User request:
{user_request}

Tool used:
{tool_name}

Tool result:
{tool_result_json}
""".strip()


def build_fallback_final_answer(tool_result: dict | list | None) -> str:
    """
    Create a simple final answer when the chat model is unavailable.

    This keeps the agent usable in local demos even if OPENAI_API_KEY is not
    configured.
    """
    if isinstance(tool_result, dict):
        if "answer" in tool_result:
            return str(tool_result["answer"])

        if "row_count" in tool_result:
            return f"Found {tool_result['row_count']} matching record(s)."

        if "is_ml_ready" in tool_result:
            return f"ML-ready status: {tool_result['is_ml_ready']}"

    if isinstance(tool_result, list):
        return f"Found {len(tool_result)} matching record(s)."

    return "No result was returned by the selected tool."


def generate_final_answer(
    settings: AppSettings,
    user_request: str,
    tool_name: str,
    tool_result: dict | list | None,
) -> str:
    """
    Generate a natural-language final answer from the selected tool output.

    The raw tool result is still saved in the response for auditability, while
    this answer is the user-facing explanation.
    """
    if not validate_openai_settings(settings=settings):
        return build_fallback_final_answer(tool_result=tool_result)

    chat_model = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    prompt = build_final_answer_prompt(
        user_request=user_request,
        tool_name=tool_name,
        tool_result=tool_result,
    )
    response = chat_model.invoke(prompt)

    return response.content.strip()


def run_llm_document_agent(settings: AppSettings, user_request: str) -> dict:
    """
    Run an LLM-based agent workflow for one user request.

    The workflow has three visible steps: select a tool, execute it, and return
    a traceable response for the course demo.
    """
    if is_unsafe_request(user_request=user_request):
        blocked_response = build_blocked_agent_response(user_request=user_request)
        append_agent_audit_entry(
            settings=settings,
            agent_response=blocked_response,
        )
        return blocked_response

    decision = select_tool_with_llm(
        settings=settings,
        user_request=user_request,
    )
    result = execute_tool_decision(
        settings=settings,
        decision=decision,
    )
    final_answer = generate_final_answer(
        settings=settings,
        user_request=user_request,
        tool_name=decision["tool_name"],
        tool_result=result,
    )

    agent_response = {
        "user_request": user_request,
        "tool_decision": decision,
        "tool_used": decision["tool_name"],
        "final_answer": final_answer,
        "result": result,
    }
    append_agent_audit_entry(
        settings=settings,
        agent_response=agent_response,
    )

    return agent_response


def print_llm_agent_response(agent_response: dict) -> None:
    """
    Print the LLM agent response in a readable format.

    This shows both the selected tool and the reason returned by the model.
    """
    decision = agent_response["tool_decision"]

    print(f"\nLLM agent request: {agent_response['user_request']}")
    print(f"Tool selected: {agent_response['tool_used']}")
    print(f"Arguments: {decision.get('arguments', {})}")
    print(f"Reason: {decision.get('reason')}")
    print(f"Final answer: {agent_response['final_answer']}")
    print(f"Result: {agent_response['result']}")


def save_llm_agent_responses(
    output_dir: Path,
    agent_responses: list[dict],
) -> Path | None:
    """
    Save LLM agent demo responses to a JSON file.

    The saved artifact lets devs inspect how the model selected tools and
    what each tool returned.
    """
    if not agent_responses:
        print("No LLM agent responses available to save.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_agent_responses.json"
    output_path.write_text(json.dumps(agent_responses, indent=2), encoding="utf-8")

    print(f"LLM agent responses saved to: {output_path}")
    return output_path
