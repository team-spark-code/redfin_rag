from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .query import Strategy

class SourceDoc(BaseModel):
    doc_id: str
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None

class AnswerPayload(BaseModel):
    text: str
    bullets: Optional[List[str]] = None
    format: str = "markdown"

class DataPayload(BaseModel):
    answer: AnswerPayload
    persona: str
    strategy: Strategy
    sources: List[SourceDoc] = []

class MetaPayload(BaseModel):
    user: Dict[str, str]  # {"user_id": "...", "session_id": "..."}
    request: Dict[str, Any]
    pipeline: Dict[str, Any]

class QueryResponseV1(BaseModel):
    version: str = "v1"
    data: DataPayload
    meta: MetaPayload
