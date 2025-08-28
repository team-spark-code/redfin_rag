from typing import List, Optional
from pydantic import BaseModel, Field, validator

class NewsLLMOut(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    tldr: List[str] = Field(default_factory=list)
    body_md: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    hero_image_url: Optional[str] = None
    author_name: Optional[str] = None

    @validator("tldr", "tags", "sources", pre=True)
    def _listify(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return [str(v)]

    @validator("title", "subtitle", "body_md", "hero_image_url", "author_name", pre=True)
    def _to_str(cls, v):
        return None if v is None else str(v)
