import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Place for storing all reusable project settings in one place
@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    output_data_dir: Path
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    retrieval_distance_threshold: float

# Load AppSettings
def load_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    return AppSettings(
        project_root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        output_data_dir=project_root / "data" / "output",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        retrieval_distance_threshold=float(
            os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "1.25")
        )
    )

# Validate Open AI Settings
def validate_openai_settings(settings: AppSettings) -> bool:
    return bool(settings.openai_api_key)
