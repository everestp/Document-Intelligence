import json
from pathlib import Path
import pandas as pd

from src.config.settings import AppSettings
from src.rag.qa_pipeline import run_rag_question

MAX_QUERY_LIMIT = 20
DEFAULT_QUERY_LIMIT = 10

# Convert pandas missing values into JSON-friendly Python Values
def normalize_record_value(value):
    if pd.isna(value):
        return None
    return value

# Normalize one CSV claim record before returning it from an agent tool
def  normalize_claim_record(record: dict) -> dict:
    return {
        column_name: normalize_record_value(value)
        for column_name, value in record.items()
    }

# Load JSON File
def load_json_file(file_path: Path) -> dict | list:
    return json.loads(file_path.read_text(encoding="utf-8"))

# Load claim dataset
def load_claim_dataset(settings: AppSettings) -> pd.DataFrame | None:
    dataset_path = settings.output_data_dir / "claim_dataset.csv"

    if not dataset_path.exists():
        return None

    return pd.read_csv(dataset_path)

# Summarize Pipeline Outputs
def summarize_pipeline_outputs(settings: AppSettings) -> dict:
    summary_path = settings.output_data_dir / "processing_summary.json"

    if not summary_path.exists():
        return{"error": "processing_summary.json was not found"}

    return load_json_file(summary_path)

# Get data quality report status
def get_data_quality_status(settings: AppSettings) -> dict:
    report_path = settings.output_data_dir / "data_quality_report.json"

    if not report_path.exists():
        return{"error": "data_quality_report.json was not found"}

    return load_json_file(report_path)

# Filter claim by status
def filter_claim_by_status(settings: AppSettings, status: str) -> list[dict]:
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return [{"error": "claim_dataset.csv was not found"}]

    filtered = dataframe[
        dataframe["claim_status"].astype(str).str.lower() == status.lower()
    ]

    return [
        normalize_claim_record(record)
        for record in filtered.to_dict(orient="records")
    ]

# Get Claim Fields
def get_claim_fields(settings: AppSettings, fields: list[str]) -> list[dict]:
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return[{"error": "claim_dataset.csv was not found"}]

    available_fields = set(dataframe.columns)
    selected_fields = [field for field in fields if field in available_fields]

    if not selected_fields:
        return [
            {
                "error": "No requested fields were found in claim_dataset.csv",
                "available_fields": list(dataframe.columns)
            }
        ]

    return [
        normalize_claim_record(record)
        for record in dataframe[selected_fields].to_dict(orient="records")
    ]

# Get Claim Record by Id
def get_claim_record_by_id(settings: AppSettings, claim_id: str) -> dict:
    dataframe = load_claim_dataset(settings=settings)
    if dataframe is None:
        return[{"error": "claim_dataset.csv was not found"}]

    filtered = dataframe[
        dataframe["claim_id"].str.lower() == claim_id.lower()
    ]

    if filtered.empty:
        return {"error": f"No claim record found for claim_id={claim_id}"}

    return normalize_claim_record(filtered.iloc[0].to_dict())

# Get Claim Record by Patient
def get_claim_record_by_patient(settings: AppSettings, patient_name: str) -> list[dict]:
    dataframe = load_claim_dataset(settings=settings)
    if dataframe is None:
        return[{"error": "claim_dataset.csv was not found"}]

    filtered = dataframe[
            dataframe["patient_name"].str.contains(patient_name, case=False, na=False)
        ]
    if filtered.empty:
        return {"error": f"No claim record found for patient_name={patient_name}"}

    return [
        normalize_claim_record(record)
        for record in filtered.to_dict(orient="records")
    ]

def normalize_query_operator(operator: str) -> str:
    operator_map = {
        "=": "equals",
        "==": "equals",
        "eq": "equals",
        "equals": "equals",
        "not_equals": "not_equals",
        "!=": "not_equals",
        ">": "greater_than",
        "gt": "greater_than",
        "greater_than": "greater_than",
        "<": "less_than",
        "lt": "less_than",
        "less_than": "less_than",
        ">=": "greater_than_or_equal",
        "gte": "greater_than_or_equal",
        "greater_than_or_equal": "greater_than_or_equal",
        "<=": "less_than_or_equal",
        "lte": "less_than_or_equal",
        "less_than_or_equal": "less_than_or_equal",
        "contains": "contains",
    }
    return operator_map.get(operator, "equals")

# Apply one validated filter rule to the claims dataframe
def apply_query_filter(dataframe: pd.DataFrame, filter_rule: dict) -> pd.DataFrame:
    column = filter_rule["column"]
    operator = normalize_query_operator(filter_rule.get("operator", "equals"))
    value = filter_rule.get("value")
    value_column = filter_rule.get("value_column")

    if column not in dataframe.columns:
        return dataframe

    series = dataframe[column]
    comparison_value = dataframe[value_column] if value_column in dataframe.columns else value

    if operator == "contains":
        return dataframe[
            series.astype(str).str.contains(str(comparison_value), case=False, na=False)
        ]

    if pd.api.types.is_numeric_dtype(series):
        numeric_value = pd.to_numeric(comparison_value, errors="coerce")

        if operator == "greater_than":
            return dataframe[series > numeric_value]
        if operator == "less_than":
            return dataframe[series < numeric_value]
        if operator == "greater_than_or_equal":
            return dataframe[series >= numeric_value]
        if operator == "less_than_or_equal":
            return dataframe[series <= numeric_value]
        if operator == "not_equals":
            return dataframe[series != numeric_value]

        return dataframe[series == numeric_value]

    normalized_series = series.astype(str).str.lower()
    normalized_value = comparison_value.astype(str).str.lower() if value_column in dataframe.columns else str(comparison_value).lower()
    if operator == "not_equals":
        return dataframe[normalized_series != normalized_value]

    return dataframe[normalized_series == normalized_value]

def query_claim_dataset(settings: AppSettings, query_plan: dict) -> dict:
    """
    Execute a validated dynamic query plan against the ML-ready claims CSV.

    This is the smarter structured-data tool: instead of one hardcoded function
    per question, the LLM creates a query plan and Python safely applies it.
    """
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        return {"error": "claim_dataset.csv was not found"}

    available_columns = list(dataframe.columns)
    guarded_query_plan = query_plan.copy()
    selected_columns = guarded_query_plan.get("select") or available_columns
    selected_columns = [
        column for column in selected_columns if column in available_columns
    ]

    if not selected_columns:
        selected_columns = available_columns

    filtered_dataframe = dataframe.copy()

    for filter_rule in guarded_query_plan.get("filters", []):
        if isinstance(filter_rule, dict):
            filtered_dataframe = apply_query_filter(
                dataframe=filtered_dataframe,
                filter_rule=filter_rule,
            )

    sort_by = guarded_query_plan.get("sort_by")

    if sort_by in available_columns:
        sort_order = str(guarded_query_plan.get("sort_order", "asc")).lower()
        filtered_dataframe = filtered_dataframe.sort_values(
            by=sort_by,
            ascending=sort_order != "desc",
        )

    limit = guarded_query_plan.get("limit")

    if not isinstance(limit, int) or limit < 1:
        limit = DEFAULT_QUERY_LIMIT

    limit = min(limit, MAX_QUERY_LIMIT)
    guarded_query_plan["limit"] = limit
    filtered_dataframe = filtered_dataframe.head(limit)

    records = [
        normalize_claim_record(record)
        for record in filtered_dataframe[selected_columns].to_dict(orient="records")
    ]

    return {
        "query_plan": guarded_query_plan,
        "row_count": len(records),
        "max_query_limit": MAX_QUERY_LIMIT,
        "records": records,
    }


def ask_claim_documents(settings: AppSettings, question: str) -> dict | None:
    """
    Ask a natural-language question against the claim documents.

    This tool delegates to the RAG question-answering pipeline built in the
    previous module.
    """
    return run_rag_question(settings=settings, question=question)
