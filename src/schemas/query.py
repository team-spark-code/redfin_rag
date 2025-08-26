from pydantic import BaseModel, Field
from typing import Optional, Literal

Strategy = Literal["auto", "stuff", "map_refine", "cod"]

class QueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질의")
    persona: Optional[str] = Field(None, description="페르소나 (없으면 auto)")
    strategy: Strategy = "auto"
    top_k: int = 8
    fetch_k: int = 60
    lambda_mult: float = 0.25
    user_id: Optional[str] = Field("notuser", description="회원 ID 없으면 notuser")
