from pydantic import BaseModel, Field

# Represent the request body for asking a grounded RAG question.
class RagQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default = 3, ge=1, le=5)
    use_cache: bool = True

# Represent the request body for asking the LLM document agent
class AgentQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)

# Represent the response returned after triggering the pipeline
class PipelineRunResponse(BaseModel):
    success: bool
    message: str
