from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class NewsPublishRequest(BaseModel):
    article_id: Optional[str] = None
    article_code: Optional[str] = None  # 기존 호환(남아있다면 유지)
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    top_k: int = 6
    publish: bool = True
    author_name: Optional[str] = None

    @property
    def article_key(self) -> Optional[str]:
        return self.article_id or self.article_code

class NewsPost(BaseModel):
    post_id: str = Field(default_factory=lambda: "")
    article_id: Optional[str] = None
    article_code: Optional[str] = None
    url: Optional[str] = None

    title: str
    dek: Optional[str] = None
    tldr: List[str] = Field(default_factory=list)
    body_md: str

    hero_image_url: Optional[str] = None
    author_name: Optional[str] = None

    category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    status: str = "published"  # "draft" | "published"
    model_meta: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: datetime = Field(default_factory=datetime.utcnow)
