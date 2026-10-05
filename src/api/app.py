from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from main import main as run_full_pipeline
from src.agents.document_tools import (
    get_claim_record_by_id,
    load_claim_dataset,
    normalize_claim_record
)

from src.agents.llm_agent import load_agent_audit_trail, run_llm_document_agent
from src.api.schemas import AgentQuestionRequest, PipelineRunResponse, RagQuestionRequest
from src.config.settings import load_settings
from src.rag.qa_pipeline import run_rag_question

app = FastAPI(
    title="AI Document Inteligence API",
    description="Fast API layer for the PDF-to-ML-Ready data document intelligence project.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Loading the JSON File
def read_json_artifact(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))

# Get the output file status
def get_output_file_status() -> dict:
    settings = load_settings()
    output_dir = settings.output_data_dir
    output_files = {
        "rag_answers": output_dir / "rag_answers.json",
        "claim_records": output_dir / "claim_records.json",
        "claim_dataset": output_dir / "claim_dataset.csv",
        "data_quality_report": output_dir / "data_quality_report.json",
        "data_dictionary" : output_dir / "data_dictionary.csv",
        "processing_summary": output_dir / "processing_summary.json",
        "agent_audit_trail": output_dir / "agent_audit_trail.json"
    }

    return {
        name:{
            "exists": file_path.exists(),
            "path": str(file_path)
        }
        for name, file_path in output_files.items()
    }

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "AI Document Intelligence API"}

@app.get("/")
def api_home() -> dict:
    return {
        "service": "AI Document Intelligence API",
        "docs": "/docs",
        "health": "/health",
        "pipeline_status": "/pipeline/status"
    }

@app.get("/pipeline/status")
def get_pipeline_status() -> dict:
    settings = load_settings()
    summary_path = settings.output_data_dir / "processing_summary.json"
    summary = read_json_artifact(summary_path)

    return {
        "project_root": str(settings.project_root),
        "output_files": get_output_file_status(),
        "processing_summary": summary
    }

# Run the full document intelligence pipeline from the API
@app.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline() -> PipelineRunResponse:
    try:
        run_full_pipeline()
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex

    return PipelineRunResponse(
        success=True,
        message="Pipeline completed successfully."
    )

@app.post("/rag/ask")
def ask_rag_question(request: RagQuestionRequest) -> dict:
    settings = load_settings()
    response = run_rag_question(
        settings=settings,
        question=request.question,
        top_k= request.top_k,
        use_cache=request.use_cache
    )

    if response is None:
        raise HTTPException(status_code = 503, detail="Rag response was not created")

    return response

# Ask the LLM document agent through the api
@app.post("/agent/ask")
def ask_agent(request: AgentQuestionRequest) -> dict:
    settings = load_settings()
    return run_llm_document_agent(
        settings=settings,
        user_request=request.question
    )

# Get the claims
@app.get("/claims")
def get_claims() -> dict:
    settings = load_settings()
    dataframe = load_claim_dataset(settings=settings)

    if dataframe is None:
        raise HTTPException(status_code=404, detail="claim_dataset.csv was not found")

    records = [
        normalize_claim_record(record)
        for record in dataframe.to_dict(orient="records")
    ]

    return {
        "row_count": len(records),
        "records": records
    }

# Get the claims with Id
@app.get("/claims/{claim_id}")
def get_claim_by_id(claim_id: str) -> dict:
    settings = load_settings()
    record = get_claim_record_by_id(settings=settings, claim_id=claim_id)

    if "error" in record:
        raise HTTPException(status_code=404, detail=record["error"])

    return record

# Run audit trail
@app.get("/audit-trail")
def get_audit_trail() -> dict:
    settings = load_settings()
    entries = load_agent_audit_trail(settings=settings)

    return {
        "entry_count": len(entries),
        "entries": entries
    }
