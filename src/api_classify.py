# src/api_classify.py
from __future__ import annotations
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from news_cat.hybrid import HybridClassifier, LABEL_KR

app = FastAPI(title="News Category Classifier (Hybrid)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 글로벌 싱글톤으로 모델 로드(프로세스 시작 시 1회)
clf: Optional[HybridClassifier] = None

class NewsItem(BaseModel):
    guid: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    pub_date: Optional[str] = None
    published_at: Optional[str] = None

class ClassifiedItem(BaseModel):
    guid: Optional[str] = None
    category_pred: str
    category_label_kr: str
    category_confidence: float
    category_rationale: str

@app.on_event("startup")
def _startup():
    global clf
    clf = HybridClassifier.from_env()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/classify", response_model=List[ClassifiedItem])
def classify(
    items: List[NewsItem] = Body(...),
    max_rows: int = Query(20, ge=1, le=1000),
    max_chars: int = Query(2000, ge=200, le=20000),
):
    assert clf is not None
    rows = [i.model_dump() for i in items[:max_rows]]
    out = clf.classify_rows(rows, max_chars=max_chars)
    # 최소 결과 스키마로 매핑
    resp: List[ClassifiedItem] = []
    for r in out:
        resp.append(ClassifiedItem(
            guid=r.get("guid"),
            category_pred=r["category_pred"],
            category_label_kr=LABEL_KR[r["category_pred"]],
            category_confidence=r["category_confidence"],
            category_rationale=r["category_rationale"],
        ))
    return resp
