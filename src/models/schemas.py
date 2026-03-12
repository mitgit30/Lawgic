from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None


class CitationItem(BaseModel):
    citation: str
    section: str
    reference_type: str
    score: float | None = None


class ClauseAnalysis(BaseModel):
    clause_id: int
    clause_text: str
    clause_type: str
    entities: list[dict] = Field(default_factory=list)
    confidence: float | None = None


class RiskAssessment(BaseModel):
    clause_id: int
    clause_text: str
    clause_type: str
    risk_level: str
    triggers: list[str]


class UploadResponse(BaseModel):
    filename: str
    total_clauses: int
    summary: str
    clauses: list[ClauseAnalysis]
    risks: list[RiskAssessment]


class AskRequest(BaseModel):
    question: str
    clause_text: str | None = None
    top_k: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: list[RetrievedChunk]
    citations: list[CitationItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
